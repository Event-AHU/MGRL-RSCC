import torch, os
from torch import nn
import math
from torch import Tensor
from typing import Optional
from torch.nn import functional as F


class resblock(nn.Module):
    '''
    module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 2), kernel_size=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 2), int(outchannel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 2), outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return F.relu(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))

    def forward(self, x):
        # fixed positional encoding
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Mesh_TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        enc_att, att_weight = self._mha_block(self_att_tgt, memory, memory_mask, memory_key_padding_mask)

        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x + tgt

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem,
                                            attn_mask=attn_mask,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=True)
        return self.dropout2(x), att_weight

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class StackTransformer(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = tgt
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer + SCST decode() support.
    """

    def __init__(self, encoder_dim, feature_dim, vocab_size, max_lengths,
                 word_vocab, n_head, n_layers, dropout):
        super(DecoderTransformer, self).__init__()

        print("decoder_n_layers=", n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout_rate = dropout

        self.Conv1 = nn.Conv2d(encoder_dim * 2, feature_dim, kernel_size=1)
        self.LN = resblock(feature_dim, feature_dim)

        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)

        decoder_layer = Mesh_TransformerDecoderLayer(
            feature_dim, n_head,
            dim_feedforward=feature_dim * 4,
            dropout=self.dropout_rate
        )
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim, max_len=max_lengths)

        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.init_weights()

    def init_weights(self):
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    # =========================================================
    # (NEW) 统一处理 memory，保证 forward/sample/decode 一致
    # =========================================================
    def _prepare_memory(self, x1, x2):
        """
        Prepare encoder memory tokens for decoder cross-attention.
        Return memory: (hw, B, dim)
        """
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1) + x_sam.unsqueeze(1)
        x = self.LN(self.Conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)  # (hw, B, dim)
        return x

    # =========================================================
    # 原本 forward (CE teacher forcing) 基本不变
    # =========================================================
    def forward(self, x1, x2, encoded_captions, caption_lengths):
        memory = self._prepare_memory(x1, x2)

        word_length = encoded_captions.size(1)
        device = encoded_captions.device

        mask = torch.triu(torch.ones(word_length, word_length, device=device) * float('-inf'), diagonal=1)

        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | (encoded_captions == self.word_vocab['<END>'])

        word_emb = self.vocab_embedding(encoded_captions)  # (B, L, D)
        word_emb = word_emb.transpose(1, 0)  # (L, B, D)
        word_emb = self.position_encoding(word_emb)

        pred = self.transformer(word_emb, memory, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
        pred = self.wdc(self.dropout(pred))  # (L, B, vocab)
        pred = pred.permute(1, 0, 2)  # (B, L, vocab)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind

    # =========================================================
    # sample() 依旧保留你原本接口，内部调用 greedy decode
    # =========================================================
    @torch.no_grad()
    def sample(self, x1, x2, k=1):
        seqs, _ = self.decode(x1, x2, mode="greedy")
        # 你原来的 sample 输出 list
        return seqs[0].cpu().tolist()

    # =========================================================
    # (NEW) decode(): SCST 必需。返回 seqs + log_probs
    # =========================================================
    def decode(self, x1, x2, mode="greedy", temperature=1.0, topk=None):
        memory = self._prepare_memory(x1, x2)
        device = memory.device
        B = memory.size(1)

        start_id = self.word_vocab['<START>']
        end_id   = self.word_vocab['<END>']
        null_id  = self.word_vocab['<NULL>']

        # seqs: (B, cur_len) 逐步增长，不做 inplace 写
        seqs = torch.full((B, 1), start_id, dtype=torch.long, device=device)

        chosen_log_probs = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(self.max_lengths - 1):
            cur_len = seqs.size(1)  # 当前长度

            # causal mask: (cur_len, cur_len)
            cur_mask = torch.triu(
                torch.ones(cur_len, cur_len, device=device) * float('-inf'),
                diagonal=1
            )

            # padding mask：这里 seqs 没有 NULL，但可以把 END 之后的 token 当 padding
            tgt_pad_mask = (seqs == end_id)  # (B, cur_len)

            word_emb = self.vocab_embedding(seqs).transpose(1, 0)  # (cur_len, B, D)
            word_emb = self.position_encoding(word_emb)

            pred = self.transformer(
                word_emb,
                memory,
                tgt_mask=cur_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )
            pred = self.wdc(self.dropout(pred))  # (cur_len, B, vocab)
            
            
            # print("pred.shape:", pred.shape)        # 期望 (cur_len, B, vocab)
            # tmp = pred[-1]
            # print("pred[-1].shape:", tmp.shape)     # 期望 (B, vocab)
            
            # 只取最后一个时间步
            # scores_t = pred[-1].transpose(0, 1)  # (B, vocab)

            # scores_t = scores_t / temperature
            # logp_t = F.log_softmax(scores_t, dim=-1)
            scores_t = pred[-1]  # (B, vocab) 这里已经是正确形状，不要 transpose
            logp_t = F.log_softmax(scores_t / temperature, dim=-1)  # (B, vocab)


            if mode == "greedy":
                next_token = torch.argmax(logp_t, dim=-1)
            elif mode == "sample":
                if topk is not None:
                    topk_val, topk_idx = torch.topk(logp_t, topk, dim=-1)
                    probs = torch.softmax(topk_val, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(1)
                    next_token = topk_idx[torch.arange(B, device=device), sampled]
                else:
                    probs = torch.softmax(logp_t, dim=-1)
                    next_token = torch.multinomial(probs, 1).squeeze(1)
            else:
                raise ValueError("mode must be 'greedy' or 'sample'")

            chosen_logp = logp_t[
                torch.arange(B, device=device), next_token
            ]  # (B,)
            chosen_log_probs.append(chosen_logp)

            # 已结束的样本强制输出 END（不影响 logp 的计算，因为 logp 已经取过了）
            next_token = torch.where(
                finished,
                torch.full_like(next_token, end_id),
                next_token
            )
            
            # print("logp_t.shape:", logp_t.shape)    # 期望 (B, vocab)
            # print("next_token.shape:", next_token.shape)


            seqs = torch.cat([seqs, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == end_id)
            if finished.all():
                break

        log_probs = (
            torch.stack(chosen_log_probs, dim=1)
            if len(chosen_log_probs) > 0
            else torch.zeros(B, 0, device=device)
        )
        return seqs, log_probs

