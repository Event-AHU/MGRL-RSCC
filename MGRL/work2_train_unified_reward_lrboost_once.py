import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
import csv
from datetime import datetime
from collections import Counter

from data.LEVIR_CC.LEVIRCCbase import LEVIRCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.work2_decoder_cider import DecoderTransformer

from utils import accuracy
from utils import get_eval_score
import torch

from eval_func.bleu.bleu import Bleu
from eval_func.cider.cider import Cider
from kg_reward_utils import (
    build_kg_support_maps,
    KG_ENTITY_PHRASE_TO_TAGS,
    KG_ENTITY_PHRASE_LENGTHS,
    KG_RELATION_PHRASE_TO_TAGS,
    KG_RELATION_PHRASE_LENGTHS,
    KG_TAIL_PHRASE_TO_TAGS,
    KG_TAIL_PHRASE_LENGTHS,
    match_kg_entities_from_tokens,
    match_kg_relations_from_tokens,
    match_kg_tail_tags_from_tokens,
)
from lr_boost_utils import build_boost_step_plan, maybe_apply_lr_boost
from checkpoint_utils import should_save_checkpoint


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_token(token):
    return token.lower().strip(".,;:!?")


def set_f1_score(pred_set, gt_set, eps=1e-8):
    if len(pred_set) == 0 or len(gt_set) == 0:
        return 0.0

    hit = len(pred_set & gt_set)
    if hit == 0:
        return 0.0

    precision = hit / float(len(pred_set))
    recall = hit / float(len(gt_set))
    return 2.0 * precision * recall / (precision + recall + eps)


def triplet_similarity(pred_triplet, gt_triplet):
    head_score = 1.0 if pred_triplet[0] == gt_triplet[0] else 0.0
    relation_score = 1.0 if pred_triplet[1] == gt_triplet[1] else 0.0
    tail_score = 1.0 if pred_triplet[2] == gt_triplet[2] else 0.0
    return (head_score + relation_score + tail_score) / 3.0


def soft_triplet_f1(pred_triples, gt_triples, eps=1e-8):
    if len(pred_triples) == 0 or len(gt_triples) == 0:
        return 0.0

    pred_list = list(pred_triples)
    gt_list = list(gt_triples)

    precision = 0.0
    for pred_triplet in pred_list:
        precision += max(triplet_similarity(pred_triplet, gt_triplet) for gt_triplet in gt_list)
    precision /= float(len(pred_list))

    recall = 0.0
    for gt_triplet in gt_list:
        recall += max(triplet_similarity(gt_triplet, pred_triplet) for pred_triplet in pred_list)
    recall /= float(len(gt_list))

    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall + eps)


DETAIL_PHRASES = [
    ("new", "road"), ("new", "roads"),
    ("two", "roads"),
    ("cement", "road"), ("cement", "roads"),
    ("t-shaped", "road"), ("t-shaped", "roads"),
    ("vertical", "road"), ("vertical", "roads"),
    ("straight", "road"), ("straight", "roads"),
    ("winding", "road"), ("winding", "roads"),
    ("along", "the", "road"), ("along", "the", "roads"),
    ("of", "the", "road"), ("of", "the", "roads"),
    ("built", "along", "the", "road"), ("built", "along", "the", "roads"),
    ("sides", "of", "the", "road"), ("sides", "of", "the", "roads"),
    ("beside", "the", "road"), ("beside", "the", "roads"),
    ("small", "house"), ("small", "houses"),
    ("new", "house"), ("new", "houses"),
    ("two", "houses"), ("three", "houses"), ("four", "houses"),
    ("massive", "houses"),
    ("several", "houses"),
    ("detached", "house"),
    ("rows", "of", "houses"),
    ("row", "of", "houses"),
]


def _build_phrase_index(phrases):
    by_len = {}
    for p in phrases:
        by_len.setdefault(len(p), set()).add(tuple(w.lower() for w in p))
    return by_len


PHRASE_BY_LEN = _build_phrase_index(DETAIL_PHRASES)


def main(args):
    """
    Unified training script for flexible multi-reward SCST.
    Each reward can be enabled in a custom epoch window with its own weight.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.decoder_lr_boost_epoch is None) != (args.decoder_lr_boost_value is None):
        raise ValueError("decoder LR boost requires both --decoder_lr_boost_epoch and --decoder_lr_boost_value")
    if args.decoder_lr_boost_epoch is not None and args.decoder_lr_boost_epoch < 0:
        raise ValueError("--decoder_lr_boost_epoch must be >= 0")
    if args.decoder_lr_boost_value is not None and args.decoder_lr_boost_value <= 0:
        raise ValueError("--decoder_lr_boost_value must be > 0")

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    save_bleu4_threshold = getattr(args, "save_bleu4_threshold", 0.64)
    best_bleu4 = 0.4
    start_epoch = 0

    with open(os.path.join(args.list_path, args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    id2word = {v: k for k, v in word_vocab.items()}

    cider_scorer = Cider()
    bleu_scorer = Bleu(4)
    print("[INFO] CIDEr scorer loaded from local eval_func/cider.")
    print("[INFO] BLEU scorer loaded from local eval_func/bleu.")

    kg_data = build_kg_support_maps(args.kg_csv)
    head_relation_to_tails = kg_data["head_relation_to_tails"]
    print(
        "[INFO] KG resource loaded: "
        f"{len(kg_data['triples'])} normalized triples, "
        f"{len(kg_data['triple_counter'])} unique triples, "
        f"relations={dict(kg_data['relation_counter'])}"
    )

    if (
        args.kg_entity_weight < 0 or
        args.kg_relation_weight < 0 or
        args.kg_tail_weight < 0 or
        args.kg_triplet_weight < 0 or
        args.kg_supported_bonus_weight < 0
    ):
        raise ValueError("KG reward weights must be non-negative")
    if (
        args.kg_entity_weight == 0 and
        args.kg_relation_weight == 0 and
        args.kg_tail_weight == 0 and
        args.kg_triplet_weight == 0 and
        args.kg_supported_bonus_weight == 0
    ):
        raise ValueError("At least one KG reward component weight must be > 0")

    log_path = os.path.join(args.savepath, args.log_file)
    is_new_log = (not os.path.exists(log_path))
    log_f = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_f)
    print(
        "[INFO] Checkpoint save policy: "
        f"BLEU-4 must be > {save_bleu4_threshold:.4f}, "
        f"savepath={args.savepath}"
    )
    if is_new_log:
        log_writer.writerow([
            "time", "split", "epoch", "global_step",
            "loss", "loss_ce", "loss_rl", "top5_acc",
            "rg", "rs", "adv_mean", "adv_std", "pos_ratio", "logp_sum_mean", "rl_loss",
            "bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL", "cider"
        ])
        if args.log_flush:
            log_f.flush()

    def tokens_to_words(token_ids):
        words = []
        for tid in token_ids:
            w = id2word.get(int(tid), None)
            if w is None:
                continue
            if w in ["<START>", "<END>", "<NULL>"]:
                continue
            words.append(w)
        return words

    def tokens_to_sentence(token_ids):
        return " ".join(tokens_to_words(token_ids))

    def ids_to_normalized_tokens(token_ids):
        toks = []
        for w in tokens_to_words(token_ids):
            nw = normalize_token(w)
            if nw:
                toks.append(nw)
        return toks

    def tokens_to_sentence_list(batch_token_ids):
        return [tokens_to_sentence(x) for x in batch_token_ids]

    def reward_active(epoch, start, end):
        if epoch < start:
            return False
        if end >= 0 and epoch > end:
            return False
        return True

    def _ids_to_tokens_list(token_ids):
        toks = []
        for tid in token_ids:
            w = id2word.get(int(tid), None)
            if w is None:
                continue
            if w in ["<START>", "<END>", "<NULL>"]:
                continue
            toks.append(w.lower())
        return toks

    def _match_detail_phrases(tokens):
        hits = set()
        L = len(tokens)
        for n, phrase_set in PHRASE_BY_LEN.items():
            if L < n:
                continue
            for i in range(L - n + 1):
                g = tuple(tokens[i:i + n])
                if g in phrase_set:
                    hits.add(g)
        return hits

    def _gt_detail_set_from_5refs(refs_tokens_5, min_vote=1):
        c = Counter()
        for toks in refs_tokens_5:
            for h in _match_detail_phrases(toks):
                c[h] += 1
        return {p for p, k in c.items() if k >= min_vote}

    no_change_phrases = {
        "there is no difference",
        "the scene is the same as before",
        "the two scenes seem identical",
        "nothing has changed",
        "no change has occurred",
        "almost nothing has changed",
        "no change",
    }

    def normalize_sentence(text):
        toks = []
        for tok in text.lower().strip().split():
            tok = tok.strip(".,;:!?")
            if tok:
                toks.append(tok)
        return " ".join(toks)

    def is_no_change_sentence(text):
        return normalize_sentence(text) in no_change_phrases

    def is_no_change_tokens(token_ids):
        return is_no_change_sentence(tokens_to_sentence(token_ids))

    def gt_no_change_mask(gt_tokens_all):
        gt_tokens_all = gt_tokens_all.detach().cpu().tolist()
        out = []
        for gt5 in gt_tokens_all:
            flags = [is_no_change_tokens(gt) for gt in gt5]
            out.append(all(flags))
        return torch.tensor(out, device=device, dtype=torch.bool)

    def pred_no_change_mask(gen_seqs):
        gen_seqs = gen_seqs.detach().cpu().tolist()
        out = [is_no_change_tokens(seq) for seq in gen_seqs]
        return torch.tensor(out, device=device, dtype=torch.bool)

    def bleu_reward(gen_seqs, gt_tokens_all):
        gen_seqs = gen_seqs.detach().cpu().tolist()
        gt_tokens_all = gt_tokens_all.detach().cpu().tolist()

        res = []
        gts = []

        for g, gt5 in zip(gen_seqs, gt_tokens_all):
            res.append([tokens_to_sentence(g)])
            gts.append([tokens_to_sentence(gt) for gt in gt5])

        score, scores = bleu_scorer.compute_score(gts, res)
        bleu_avg = np.mean(np.array(scores), axis=0)
        return torch.tensor(bleu_avg, device=device, dtype=torch.float)

    def match_tag_spans(tokens, phrase_to_tags, phrase_lengths):
        spans = []
        length = len(tokens)

        for n in phrase_lengths:
            if length < n:
                continue
            for i in range(length - n + 1):
                phrase = tuple(tokens[i:i + n])
                tags = phrase_to_tags.get(phrase, None)
                if not tags:
                    continue
                for tag in tags:
                    spans.append((tag, i, i + n))

        return spans

    def select_nearest_left_spans(anchor_start, spans, keep_topk=1):
        if len(spans) == 0:
            return []

        left_spans = [span for span in spans if span[2] <= anchor_start]
        if len(left_spans) > 0:
            left_spans.sort(key=lambda span: (anchor_start - span[2], -(span[2] - span[1]), span[1]))
            return left_spans[:keep_topk]

        fallback = sorted(
            spans,
            key=lambda span: (abs(anchor_start - span[1]), abs(anchor_start - span[2]), span[1])
        )
        return fallback[:keep_topk]

    def select_nearest_right_spans(anchor_end, spans, keep_topk=2):
        if len(spans) == 0:
            return []

        right_spans = [span for span in spans if span[1] >= anchor_end]
        if len(right_spans) > 0:
            right_spans.sort(key=lambda span: (span[1] - anchor_end, -(span[2] - span[1]), span[1]))
            return right_spans[:keep_topk]

        fallback = sorted(
            spans,
            key=lambda span: (abs(span[1] - anchor_end), abs(span[2] - anchor_end), span[1])
        )
        return fallback[:keep_topk]

    def infer_open_candidate_triples(tokens):
        entity_spans = match_tag_spans(tokens, KG_ENTITY_PHRASE_TO_TAGS, KG_ENTITY_PHRASE_LENGTHS)
        relation_spans = match_tag_spans(tokens, KG_RELATION_PHRASE_TO_TAGS, KG_RELATION_PHRASE_LENGTHS)
        tail_spans = match_tag_spans(tokens, KG_TAIL_PHRASE_TO_TAGS, KG_TAIL_PHRASE_LENGTHS)

        triples = set()

        for relation_tag, rel_start, rel_end in relation_spans:
            head_candidates = select_nearest_left_spans(rel_start, entity_spans, keep_topk=1)
            tail_candidates = select_nearest_right_spans(rel_end, tail_spans, keep_topk=2)

            if len(head_candidates) == 0 or len(tail_candidates) == 0:
                continue

            for head_tag, _head_start, _head_end in head_candidates:
                for tail_tag, _tail_start, _tail_end in tail_candidates:
                    triples.add((head_tag, relation_tag, tail_tag))

        return triples

    def infer_supported_triples(candidate_triples):
        triples = set()

        for head_tag, relation_tag, tail_tag in candidate_triples:
            allowed_tails = head_relation_to_tails.get((head_tag, relation_tag), None)
            if not allowed_tails:
                continue
            if tail_tag in allowed_tails:
                triples.add((head_tag, relation_tag, tail_tag))

        return triples

    def build_sentence_kg_structure(token_ids):
        tokens = ids_to_normalized_tokens(token_ids)
        entity_tags = match_kg_entities_from_tokens(tokens)
        relation_tags = match_kg_relations_from_tokens(tokens)
        tail_tags = match_kg_tail_tags_from_tokens(tokens)
        open_triplets = infer_open_candidate_triples(tokens)
        supported_triplets = infer_supported_triples(open_triplets)
        return entity_tags, relation_tags, tail_tags, open_triplets, supported_triplets

    def kg_caption_score(pred_seq, gt_seq):
        pred_entities, pred_relations, pred_tails, pred_open_triples, pred_supported_triples = build_sentence_kg_structure(pred_seq)
        gt_entities, gt_relations, gt_tails, gt_open_triples, gt_supported_triples = build_sentence_kg_structure(gt_seq)

        if (
            len(gt_entities) == 0 and
            len(gt_relations) == 0 and
            len(gt_tails) == 0 and
            len(gt_open_triples) == 0
        ):
            return None

        f1_entity = set_f1_score(pred_entities, gt_entities)
        f1_relation = set_f1_score(pred_relations, gt_relations)
        f1_tail = set_f1_score(pred_tails, gt_tails)
        f1_open_triplet = soft_triplet_f1(pred_open_triples, gt_open_triples)
        f1_supported_triplet = soft_triplet_f1(pred_supported_triples, gt_supported_triples)

        score = (
            args.kg_entity_weight * f1_entity +
            args.kg_relation_weight * f1_relation +
            args.kg_tail_weight * f1_tail +
            args.kg_triplet_weight * f1_open_triplet +
            args.kg_supported_bonus_weight * f1_supported_triplet
        )
        return score

    def kg_reward_batch(seqs, token_all):
        seqs_cpu = seqs.detach().cpu().tolist()
        token_all_cpu = token_all.detach().cpu().tolist()

        rewards = np.zeros(len(seqs_cpu), dtype=np.float32)

        for i, (pred_seq, gt5) in enumerate(zip(seqs_cpu, token_all_cpu)):
            best_score = None

            for gt_seq in gt5:
                score = kg_caption_score(pred_seq, gt_seq)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score

            rewards[i] = 0.0 if best_score is None else best_score

        return torch.tensor(rewards, device=device, dtype=torch.float)

    def cider_reward(gen_seqs, gt_tokens_all):
        gen_seqs = gen_seqs.detach().cpu().tolist()
        gt_tokens_all = gt_tokens_all.detach().cpu().tolist()

        res = []
        gts = []

        for g, gt5 in zip(gen_seqs, gt_tokens_all):
            gen_sent = tokens_to_sentence(g)
            res.append([gen_sent])

            gt_sents = []
            for gt in gt5:
                gt_sents.append(tokens_to_sentence(gt))
            gts.append(gt_sents)

        _score, scores = cider_scorer.compute_score(gts, res)
        return torch.tensor(scores, device=device, dtype=torch.float)

    def change_state_reward(gen_seqs, gt_tokens_all):
        gt_is_no_change = gt_no_change_mask(gt_tokens_all)
        pred_is_no_change = pred_no_change_mask(gen_seqs)

        reward = torch.empty(gen_seqs.size(0), device=device, dtype=torch.float)
        both_no_change = gt_is_no_change & pred_is_no_change
        gt_no_pred_change = gt_is_no_change & (~pred_is_no_change)
        gt_change_pred_change = (~gt_is_no_change) & (~pred_is_no_change)
        gt_change_pred_no = (~gt_is_no_change) & pred_is_no_change

        reward[both_no_change] = args.reward_nochange_nochange
        reward[gt_no_pred_change] = args.reward_nochange_change
        reward[gt_change_pred_change] = args.reward_change_change
        reward[gt_change_pred_no] = args.reward_change_nochange
        return reward

    def detail_reward_batch(seqs, token_all):
        seqs_cpu = seqs.detach().cpu().tolist()
        token_all_cpu = token_all.detach().cpu().tolist()

        out = np.zeros(len(seqs_cpu), dtype=np.float32)
        for i in range(len(seqs_cpu)):
            pred_tokens = _ids_to_tokens_list(seqs_cpu[i])
            refs_tokens_5 = [_ids_to_tokens_list(token_all_cpu[i][j]) for j in range(5)]
            gt_set = _gt_detail_set_from_5refs(refs_tokens_5, min_vote=args.detail_min_vote)

            if len(gt_set) == 0:
                out[i] = 0.0
                continue

            pred_set = _match_detail_phrases(pred_tokens)
            hit = len(pred_set & gt_set)
            out[i] = hit / float(len(gt_set))

        return torch.tensor(out, device=device, dtype=torch.float)

    def total_reward(gen_seqs, gt_tokens_all, epoch):
        reward = torch.zeros(gen_seqs.size(0), device=device, dtype=torch.float)
        total_weight = 0.0

        if reward_active(epoch, args.bleu_reward_start_epoch, args.bleu_reward_end_epoch) and args.bleu_reward_weight > 0:
            reward = reward + args.bleu_reward_weight * bleu_reward(gen_seqs, gt_tokens_all)
            total_weight += args.bleu_reward_weight

        if reward_active(epoch, args.cider_reward_start_epoch, args.cider_reward_end_epoch) and args.cider_reward_weight > 0:
            reward = reward + args.cider_reward_weight * cider_reward(gen_seqs, gt_tokens_all)
            total_weight += args.cider_reward_weight

        if reward_active(epoch, args.change_reward_start_epoch, args.change_reward_end_epoch) and args.change_reward_weight > 0:
            reward = reward + args.change_reward_weight * change_state_reward(gen_seqs, gt_tokens_all)
            total_weight += args.change_reward_weight

        if reward_active(epoch, args.detail_reward_start_epoch, args.detail_reward_end_epoch) and args.detail_reward_weight > 0:
            reward = reward + args.detail_reward_weight * detail_reward_batch(gen_seqs, gt_tokens_all)
            total_weight += args.detail_reward_weight

        if reward_active(epoch, args.kg_reward_start_epoch, args.kg_reward_end_epoch) and args.kg_reward_weight > 0:
            reward = reward + args.kg_reward_weight * kg_reward_batch(gen_seqs, gt_tokens_all)
            total_weight += args.kg_reward_weight

        if total_weight == 0.0:
            return torch.zeros(gen_seqs.size(0), device=device, dtype=torch.float)
        return reward / total_weight

    train_loader = data.DataLoader(
        LEVIRCCDataset(args.data_folder, args.list_path, 'train',
                       args.token_folder, args.vocab_file,
                       args.max_length, args.allow_unk),
        batch_size=args.train_batchsize, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = data.DataLoader(
        LEVIRCCDataset(args.data_folder, args.list_path, 'val',
                       args.token_folder, args.vocab_file,
                       args.max_length, args.allow_unk),
        batch_size=args.val_batchsize, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    def run_validation(epoch, global_step, compute_val_rewards=True):
        decoder.eval()
        encoder_trans.eval()
        encoder.eval()

        references = []
        hypotheses = []
        rg_list = []
        rs_list = []

        with torch.no_grad():
            for bidx, (imgA, imgB, token_all, token_all_len, token, token_len, name) in enumerate(val_loader):
                if args.val_max_batches > 0 and bidx >= args.val_max_batches:
                    break

                imgA = imgA.to(device)
                imgB = imgB.to(device)
                token_all = token_all.to(device)

                feat1, feat2 = encoder(imgA, imgB)
                feat1, feat2 = encoder_trans(feat1, feat2)

                seq = decoder.sample(feat1, feat2, k=1)

                img_token = token_all.squeeze(0).detach().cpu().tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {
                    word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {
                    word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)

                if compute_val_rewards and hasattr(decoder, "decode"):
                    seq_greedy, _ = decoder.decode(feat1, feat2, mode="greedy")
                    r_greedy = total_reward(seq_greedy, token_all, epoch)
                    rg_list.append(r_greedy.mean().item())

                    seq_sample, _logp_sample = decoder.decode(
                        feat1, feat2,
                        mode="sample",
                        temperature=args.rl_temperature,
                        topk=args.rl_topk
                    )
                    r_sample = total_reward(seq_sample, token_all, epoch)
                    rs_list.append(r_sample.mean().item())

        score_dict = get_eval_score(references, hypotheses)

        val_rg = float(np.mean(rg_list)) if len(rg_list) > 0 else None
        val_rs = float(np.mean(rs_list)) if len(rs_list) > 0 else None
        return score_dict, val_rg, val_rs

    if args.checkpoint is None:
        encoder = Encoder(args.network)
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(
            params=encoder.parameters(),
            lr=args.encoder_lr
        ) if args.fine_tune_encoder else None

        encoder_trans = AttentiveEncoder(
            n_layers=args.n_layers,
            feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
            heads=args.n_heads,
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            dropout=args.dropout
        )
        encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
            lr=args.encoder_lr
        )

        decoder = DecoderTransformer(
            encoder_dim=args.encoder_dim,
            feature_dim=args.feature_dim,
            vocab_size=len(word_vocab),
            max_lengths=args.max_length,
            word_vocab=word_vocab,
            n_head=args.n_heads,
            n_layers=args.decoder_n_layers,
            dropout=args.dropout
        )
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=args.decoder_lr
        )

    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_bleu4 = checkpoint['bleu-4']

        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_trans = checkpoint['encoder_trans']
        encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']

        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=args.encoder_lr
            )

    encoder = encoder.to(device)
    encoder_trans = encoder_trans.to(device)
    decoder = decoder.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        encoder_optimizer, step_size=5, gamma=0.5
    ) if args.fine_tune_encoder else None

    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        encoder_trans_optimizer, step_size=5, gamma=0.5)

    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        decoder_optimizer, step_size=5, gamma=0.5)

    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    decoder_lr_boost_applied = (
        args.decoder_lr_boost_epoch is not None and
        args.decoder_lr_boost_epoch < start_epoch
    )

    if args.decoder_lr_boost_epoch is not None:
        print(
            "[INFO] Decoder LR boost armed: "
            f"target_epoch={args.decoder_lr_boost_epoch}, "
            f"boost_value={args.decoder_lr_boost_value}, "
            f"pre_step_epoch={build_boost_step_plan(args.decoder_lr_boost_epoch)}"
        )
        if decoder_lr_boost_applied:
            print(
                "[INFO] Decoder LR boost already lies before resumed start epoch; "
                f"start_epoch={start_epoch}, boost will be skipped."
            )

    try:
        for epoch in range(start_epoch, args.num_epochs):
            before_lrs = [group["lr"] for group in decoder_optimizer.param_groups]
            was_applied = decoder_lr_boost_applied
            decoder_lr_boost_applied = maybe_apply_lr_boost(
                decoder_optimizer,
                current_epoch=epoch,
                boost_epoch=args.decoder_lr_boost_epoch,
                boost_value=args.decoder_lr_boost_value,
                already_applied=decoder_lr_boost_applied,
            )
            if decoder_lr_boost_applied and not was_applied:
                after_lrs = [group["lr"] for group in decoder_optimizer.param_groups]
                print(
                    "[INFO] Applied one-time decoder LR boost: "
                    f"epoch={epoch}, old_lrs={before_lrs}, new_lrs={after_lrs}"
                )

            for _id, (imgA, imgB, token_all, token_all_len, token, token_len, name) in enumerate(train_loader):
                start_time = time.time()

                decoder.train()
                encoder.train()
                encoder_trans.train()

                decoder_optimizer.zero_grad()
                encoder_trans_optimizer.zero_grad()
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()

                imgA = imgA.to(device)
                imgB = imgB.to(device)

                token = token.squeeze(1).to(device)
                token_len = token_len.to(device)
                token_all = token_all.to(device)

                feat1, feat2 = encoder(imgA, imgB)
                feat1, feat2 = encoder_trans(feat1, feat2)

                scores, caps_sorted, decode_lengths, sort_ind = decoder(feat1, feat2, token, token_len)

                targets = caps_sorted[:, 1:]
                scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                loss_ce = criterion(scores_packed, targets_packed)
                loss = loss_ce
                loss_rl = torch.tensor(0.0, device=device)

                tr_rg = tr_rs = tr_adv_mean = tr_adv_std = tr_pos_ratio = tr_logp_sum_mean = tr_rl_loss_val = None

                if args.use_rl and epoch >= args.rl_start_epoch:

                    with torch.no_grad():
                        seq_greedy, _ = decoder.decode(feat1, feat2, mode="greedy")
                        r_greedy = total_reward(seq_greedy, token_all, epoch)

                    seq_sample, logp_sample = decoder.decode(
                        feat1, feat2,
                        mode="sample",
                        temperature=args.rl_temperature,
                        topk=args.rl_topk
                    )
                    r_sample = total_reward(seq_sample, token_all, epoch)

                    adv = (r_sample - r_greedy)

                    loss_rl = -(adv.detach() * logp_sample.sum(dim=1)).mean()

                    lam = args.ce_lambda
                    loss = lam * loss_ce + (1.0 - lam) * loss_rl

                    if index_i % args.print_freq == 0:
                        with torch.no_grad():
                            tr_rg = r_greedy.mean().item()
                            tr_rs = r_sample.mean().item()
                            tr_adv_mean = adv.mean().item()
                            tr_adv_std = adv.std(unbiased=False).item()
                            tr_pos_ratio = (adv > 0).float().mean().item()
                            tr_logp_sum_mean = logp_sample.sum(dim=1).mean().item()
                            tr_rl_loss_val = loss_rl.item()

                        print(f"[TRAIN-RL-UNIFIED] Rg:{tr_rg:.3f} Rs:{tr_rs:.3f} "
                              f"Adv:{tr_adv_mean:.3f}±{tr_adv_std:.3f} "
                              f"Pos:{tr_pos_ratio:.2f} LogPsum:{tr_logp_sum_mean:.1f} RLloss:{tr_rl_loss_val:.3f}")

                loss.backward()

                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
                    if encoder_optimizer is not None:
                        torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)

                decoder_optimizer.step()
                encoder_trans_optimizer.step()
                if encoder_optimizer is not None:
                    encoder_optimizer.step()

                hist[index_i, 0] = time.time() - start_time
                hist[index_i, 1] = loss.item()
                hist[index_i, 2] = accuracy(scores_packed, targets_packed, 5)
                index_i += 1

                if index_i % args.print_freq == 0:
                    train_top5 = float(np.mean(hist[index_i - args.print_freq:index_i - 1, 2]))
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss: {3:.4f}\t'
                          'CE: {4:.4f}\t'
                          'RL: {5:.4f}\t'
                          'Top-5 Acc: {6:.3f}'.format(
                              epoch, index_i, args.num_epochs * len(train_loader),
                              loss.item(), loss_ce.item(), loss_rl.item(),
                              train_top5
                          ))

                    log_writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        "train", epoch, index_i,
                        float(loss.item()), float(loss_ce.item()), float(loss_rl.item()), train_top5,
                        "" if tr_rg is None else tr_rg,
                        "" if tr_rs is None else tr_rs,
                        "" if tr_adv_mean is None else tr_adv_mean,
                        "" if tr_adv_std is None else tr_adv_std,
                        "" if tr_pos_ratio is None else tr_pos_ratio,
                        "" if tr_logp_sum_mean is None else tr_logp_sum_mean,
                        "" if tr_rl_loss_val is None else tr_rl_loss_val,
                        "", "", "", "", "", "", ""
                    ])
                    if args.log_flush:
                        log_f.flush()

                if args.val_step_freq > 0 and (index_i % args.val_step_freq == 0):
                    t0 = time.time()
                    score_dict, val_rg, val_rs = run_validation(epoch, index_i, compute_val_rewards=True)
                    dt = time.time() - t0

                    Bleu_1 = score_dict['Bleu_1']
                    Bleu_2 = score_dict['Bleu_2']
                    Bleu_3 = score_dict['Bleu_3']
                    Bleu_4 = score_dict['Bleu_4']
                    Meteor = score_dict['METEOR']
                    Rouge = score_dict['ROUGE_L']
                    cider = score_dict['CIDEr']

                    print(f"[VAL@step] step={index_i} time={dt:.1f}s "
                          f"BLEU4={Bleu_4:.4f} CIDEr={cider:.4f} "
                          f"ValRg(kg)={val_rg if val_rg is not None else 'NA'} "
                          f"ValRs(kg)={val_rs if val_rs is not None else 'NA'}")

                    log_writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        "val", epoch, index_i,
                        "", "", "", "",
                        "" if val_rg is None else val_rg,
                        "" if val_rs is None else val_rs,
                        "", "", "", "", "",
                        Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, cider
                    ])
                    if args.log_flush:
                        log_f.flush()

                    decoder.train()
                    encoder.train()
                    encoder_trans.train()

            t0 = time.time()
            score_dict, val_rg, val_rs = run_validation(epoch, index_i, compute_val_rewards=True)
            val_time = time.time() - t0

            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            cider = score_dict['CIDEr']

            print('Validation:\n'
                  'Time: {0:.3f}\t'
                  'BLEU-1: {1:.4f}\t'
                  'BLEU-2: {2:.4f}\t'
                  'BLEU-3: {3:.4f}\t'
                  'BLEU-4: {4:.4f}\t'
                  'Meteor: {5:.4f}\t'
                  'Rouge: {6:.4f}\t'
                  'Cider(metric): {7:.4f}\t'
                  'ValRg(kg): {8}\t'
                  'ValRs(kg): {9}\t'
                  .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, cider,
                          "NA" if val_rg is None else f"{val_rg:.4f}",
                          "NA" if val_rs is None else f"{val_rs:.4f}"))

            log_writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                "val_epoch_end", epoch, index_i,
                "", "", "", "",
                "" if val_rg is None else val_rg,
                "" if val_rs is None else val_rs,
                "", "", "", "", "",
                Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, cider
            ])
            if args.log_flush:
                log_f.flush()

            decoder_lr_scheduler.step()
            encoder_trans_lr_scheduler.step()
            if encoder_lr_scheduler is not None:
                encoder_lr_scheduler.step()

            if should_save_checkpoint(Bleu_4, save_bleu4_threshold):
                best_bleu4 = max(Bleu_4, best_bleu4)
                print('Save Model')
                state = {
                    'encoder_dict': encoder.state_dict(),
                    'encoder_trans_dict': encoder_trans.state_dict(),
                    'decoder_dict': decoder.state_dict()
                }
                model_name = (
                    str(args.data_name) + '_batchsize_' + str(args.train_batchsize) + '_' +
                    str(args.network) + '_epoch_' + str(epoch) + '_Bleu4_' +
                    str(round(10000 * Bleu_4)) + '.pth'
                )
                torch.save(state, os.path.join(args.savepath, model_name))
            else:
                print(
                    "[INFO] Skip save: "
                    f"BLEU-4={Bleu_4:.4f} <= threshold={save_bleu4_threshold:.4f}"
                )

    finally:
        log_f.close()
        print(f"[INFO] Log saved to: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LEVIR-CC CE + multi-reward SCST with validation')

    parser.add_argument('--data_folder', default=os.path.join(PROJECT_DIR, 'datasets', 'LEVIR_CC', 'images'))
    parser.add_argument('--list_path', default=os.path.join(PROJECT_DIR, 'datasets', 'LEVIR_CC'))
    parser.add_argument('--token_folder', default=os.path.join(PROJECT_DIR, 'datasets', 'LEVIR_CC', 'tokens'))
    parser.add_argument('--vocab_file', default='vocab')
    parser.add_argument('--max_length', type=int, default=41)
    parser.add_argument('--allow_unk', type=int, default=1)
    parser.add_argument('--data_name', default="LEVIR_CC")

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--print_freq', type=int, default=100)

    parser.add_argument('--fine_tune_encoder', type=bool, default=True)
    parser.add_argument('--train_batchsize', type=int, default=32)
    parser.add_argument('--network', default='resnet101')
    parser.add_argument('--encoder_dim', default=2048)
    parser.add_argument('--feat_size', default=16)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=1e-5)
    parser.add_argument('--decoder_lr_boost_epoch', type=int, default=30,
                        help='epoch to apply a one-time decoder LR boost before training starts for that epoch')
    parser.add_argument('--decoder_lr_boost_value', type=float, default=2e-5,
                        help='decoder LR value used for the one-time boost')
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--val_batchsize', type=int, default=1)
    parser.add_argument('--savepath', default=os.path.join(PROJECT_DIR, 'checkpoints', 'levir_mgrl'))

    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=2048)
    parser.add_argument('--feature_dim', type=int, default=2048)

    parser.add_argument('--use_rl', action='store_true', help='enable SCST RL fine-tuning')
    parser.add_argument('--rl_start_epoch', type=int, default=8, help='start SCST at this epoch')
    parser.add_argument('--ce_lambda', type=float, default=0.9, help='loss = lam*CE + (1-lam)*RL')
    parser.add_argument('--rl_topk', type=int, default=3)
    parser.add_argument('--rl_temperature', type=float, default=0.8)

    parser.add_argument('--kg_csv', type=str,
                        default=os.path.join(PROJECT_DIR, 'datasets', 'LEVIR_CC', 'levir_kg4_three_cleaned.csv'),
                        help='path to the GBK-encoded LEVIR KG csv')
    parser.add_argument('--kg_entity_weight', type=float, default=0.5,
                        help='weight for entity F1 in the KG reward')
    parser.add_argument('--kg_relation_weight', type=float, default=0.2,
                        help='weight for relation F1 in the KG reward')
    parser.add_argument('--kg_tail_weight', type=float, default=0.2,
                        help='weight for tail/context F1 in the KG reward')
    parser.add_argument('--kg_triplet_weight', type=float, default=0.1,
                        help='weight for open candidate triplet F1 in the KG reward')
    parser.add_argument('--kg_supported_bonus_weight', type=float, default=0.0,
                        help='extra bonus weight for KG-supported triplet F1')
    parser.add_argument('--reward_nochange_nochange', type=float, default=1.0,
                        help='GT no-change, Pred no-change')
    parser.add_argument('--reward_nochange_change', type=float, default=-0.5,
                        help='GT no-change, Pred change')
    parser.add_argument('--reward_change_change', type=float, default=1.0,
                        help='GT change, Pred change')
    parser.add_argument('--reward_change_nochange', type=float, default=-1.0,
                        help='GT change, Pred no-change')
    parser.add_argument('--detail_min_vote', type=int, default=1,
                        help='minimum number of GT refs that must contain a detail phrase')

    parser.add_argument('--bleu_reward_weight', type=float, default=0.3)
    parser.add_argument('--bleu_reward_start_epoch', type=int, default=12)
    parser.add_argument('--bleu_reward_end_epoch', type=int, default=20,
                        help='-1 means never turn off')

    parser.add_argument('--cider_reward_weight', type=float, default=0.7)
    parser.add_argument('--cider_reward_start_epoch', type=int, default=8)
    parser.add_argument('--cider_reward_end_epoch', type=int, default=-1,
                        help='-1 means never turn off')

    parser.add_argument('--change_reward_weight', type=float, default=0.3)
    parser.add_argument('--change_reward_start_epoch', type=int, default=8)
    parser.add_argument('--change_reward_end_epoch', type=int, default=10,
                        help='-1 means never turn off')

    parser.add_argument('--detail_reward_weight', type=float, default=0.3)
    parser.add_argument('--detail_reward_start_epoch', type=int, default=21)
    parser.add_argument('--detail_reward_end_epoch', type=int, default=25,
                        help='-1 means never turn off')

    parser.add_argument('--kg_reward_weight', type=float, default=0.3)
    parser.add_argument('--kg_reward_start_epoch', type=int, default=21)
    parser.add_argument('--kg_reward_end_epoch', type=int, default=30,
                        help='-1 means never turn off')

    parser.add_argument('--log_file', type=str, default='train_val_log.csv',
                        help='csv file name saved under savepath')
    parser.add_argument('--val_step_freq', type=int, default=0,
                        help='run validation every N training steps (0 = only at epoch end)')
    parser.add_argument('--val_max_batches', type=int, default=0,
                        help='limit val batches for faster eval (0 = use full val set)')
    parser.add_argument('--log_flush', action='store_true',
                        help='flush csv to disk each write (safer if training may crash)')

    args = parser.parse_args()
    main(args)
