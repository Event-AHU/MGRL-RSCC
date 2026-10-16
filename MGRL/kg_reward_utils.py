"""
Utilities and resources for a KG-based auxiliary reward.

This module is meant to be imported by a future training script.
It prepares:
1. normalized relation groups
2. entity / tail lexicons tailored to LEVIR-CC style captions
3. helper functions to load the GBK-encoded KG CSV as normalized triples
"""

import csv
from collections import Counter, defaultdict

from location import LOCATION_LEXICON


KG_RELATION_GROUPS = {
    "appearance": [
        ("appear",),
        ("appears",),
        ("appears", "on"),
        ("appears", "in"),
        ("appears", "at"),
        ("shows", "up", "on"),
        ("show", "up", "in"),
    ],
    "construction": [
        ("built",),
        ("built", "on"),
        ("built", "at"),
        ("are", "built"),
        ("are", "constructed"),
        ("is", "constructed", "in"),
        ("is", "built", "with"),
        ("is", "built", "at"),
        ("is", "built", "in"),
        ("has", "been", "built", "near"),
        ("have", "been", "built", "along"),
        ("is", "built", "around"),
    ],
    "replacement": [
        ("replaces",),
        ("is", "replaced", "by"),
    ],
    "spatial": [
        ("next", "to"),
        ("in",),
        ("appears", "with"),
        ("appears", "surrounded", "by"),
    ],
    "state_change": [
        ("have",),
        ("becomes",),
    ],
}


KG_ENTITY_LEXICON = {
    "house": [
        ("house",), ("houses",),
        ("villa",), ("villas",),
        ("detached", "houses"),
        ("row", "of", "houses"),
        ("rows", "of", "houses"),
        ("many", "houses"),
        ("some", "houses"),
        ("several", "houses"),
        ("large", "house"),
        ("mansion",),
    ],
    "building": [
        ("building",), ("buildings",),
        ("two", "buildings"),
        ("three", "buildings"),
        ("small", "buildings"),
        ("large", "building"),
        ("big", "buildings"),
        ("some", "buildings"),
        ("several", "buildings"),
        ("rows", "of", "buildings"),
        ("cluster", "of", "buildings"),
        ("long", "building"),
        ("new", "building"),
        ("square", "buildings"),
        ("massive", "buildings"),
        ("factory",),
        ("tower",),
        ("mall",),
    ],
    "road": [
        ("road",), ("roads",),
        ("a", "road"),
        ("concrete", "roads"),
        ("asphalt", "road"),
        ("circular", "road"),
        ("ring", "road"),
        ("crossroad",), ("crossroads",),
        ("cross", "road"), ("cross", "roads"),
        ("crossing", "roads"),
        ("straight", "road"),
        ("winding", "road"),
        ("a", "t-shaped", "road"),
        ("turning", "road"),
        ("vertical", "wide", "roads"),
        ("original", "road"),
        ("dirt", "road"),
        ("reconstructed", "roads"),
        ("parallel", "roads"),
        ("new", "built", "roads"),
        ("bypasses",),
        ("roadside",),
    ],
    "path": [
        ("path",), ("paths",),
        ("footpath",),
        ("trail",), ("trails",),
        ("track",), ("tracks",),
    ],
    "parking_lot": [
        ("parking", "lot"),
        ("parking", "lots"),
    ],
    "trees": [
        ("tree",), ("trees",),
    ],
    "plants": [
        ("plant",), ("plants",),
        ("vegetation",),
    ],
    "forest": [
        ("forest",), ("forests",),
        ("wood",), ("woods",),
    ],
    "bare_land": [
        ("bare", "land"),
        ("bareland",),
        ("original", "bareland"),
        ("a", "wasteland"),
        ("wasteland",),
        ("a", "desert"),
        ("desert",),
        ("a", "clearing"),
        ("clearing",),
        ("empty", "space"),
    ],
    "grassland": [
        ("grassland",),
        ("grass",),
        ("field",), ("fields",),
        ("meadow",),
    ],
    "water": [
        ("pool",), ("pools",),
        ("lake",), ("lakes",),
        ("water",),
    ],
    "residential_area": [
        ("residential", "area"),
        ("old", "residential", "area"),
        ("village",),
    ],
    "playground": [
        ("playground",),
        ("court",), ("courts",),
    ],
    "storage": [
        ("tank",), ("tanks",),
        ("container",), ("containers",),
    ],
}


KG_TAIL_CONTEXT_LEXICON = {
    "road": [
        ("road",), ("roads",),
        ("a", "road"),
        ("corner", "of", "the", "road"),
        ("end", "of", "a", "road"),
        ("left", "side", "of", "the", "road"),
        ("right", "side", "of", "the", "road"),
    ],
    "path": [
        ("path",), ("paths",),
    ],
    "bare_land": [
        ("bare", "land"),
        ("bareland",),
        ("original", "bareland"),
        ("empty", "space"),
        ("a", "wasteland"),
        ("a", "desert"),
        ("a", "clearing"),
    ],
    "trees": [
        ("tree",), ("trees",),
        ("plants",),
    ],
    "forest": [
        ("forest",),
        ("woods",),
        ("edge", "of", "woods"),
        ("edge", "of", "the", "woods"),
        ("former", "forest"),
        ("clearing", "of", "the", "forest"),
    ],
    "grassland": [
        ("grassland",),
        ("grass",),
        ("field",), ("fields",),
    ],
    "parking_lot": [
        ("parking", "lot"),
        ("parking", "lots"),
    ],
    "house": [
        ("house",), ("houses",),
        ("villas",),
    ],
    "building": [
        ("building",), ("buildings",),
        ("two", "buildings"),
        ("cluster", "of", "buildings"),
    ],
    "residential_area": [
        ("old", "residential", "area"),
        ("residential", "area"),
    ],
    "water": [
        ("pool",),
        ("lake",),
    ],
}


KG_TAIL_CONTEXT_LEXICON.update({
    "top": LOCATION_LEXICON["top"] + [
        ("top", "of", "the", "scene"),
        ("top", "side", "of", "the", "scene"),
        ("top", "corner"),
    ],
    "bottom": LOCATION_LEXICON["bottom"] + [
        ("bottom", "of", "the", "scene"),
        ("bottom", "side", "of", "the", "scene"),
        ("bottom", "corner"),
    ],
    "left": LOCATION_LEXICON["left"] + [
        ("left",),
        ("left", "side"),
        ("left", "area"),
    ],
    "right": LOCATION_LEXICON["right"] + [
        ("right",),
        ("right", "side"),
        ("right", "scene"),
    ],
    "center": LOCATION_LEXICON["center"] + [
        ("center", "of", "the", "scene"),
        ("middle",),
        ("middle", "of", "the", "scene"),
    ],
    "top_left_corner": LOCATION_LEXICON["top_left_corner"] + [
        ("top", "left", "corner"),
        ("top", "left", "corner", "of", "the", "scene"),
    ],
    "top_right_corner": LOCATION_LEXICON["top_right_corner"] + [
        ("upper", "right"),
        ("top", "right", "of", "the", "scene"),
        ("upper", "right", "corner", "of", "the", "scene"),
    ],
    "bottom_left_corner": LOCATION_LEXICON["bottom_left_corner"] + [
        ("lower-left",),
        ("lower-left", "corner"),
        ("lower-left", "corner", "of", "the", "scene"),
        ("bottom", "left", "corner", "of", "the", "scene"),
    ],
    "bottom_right_corner": LOCATION_LEXICON["bottom_right_corner"] + [
        ("lower-right",),
        ("lower-right", "corner", "of", "the", "scene"),
        ("bottom", "right", "corner", "of", "the", "scene"),
    ],
    "both_sides": LOCATION_LEXICON["both_sides"] + [
        ("both", "sides", "of", "the", "roads"),
        ("sides", "of", "the", "road"),
    ],
    "along_road": LOCATION_LEXICON["along_road"],
    "beside_road": LOCATION_LEXICON["beside_road"],
})


KG_TAIL_HIERARCHY = {
    "top_left_corner": {"top", "left"},
    "top_right_corner": {"top", "right"},
    "bottom_left_corner": {"bottom", "left"},
    "bottom_right_corner": {"bottom", "right"},
}


def normalize_token(token):
    return token.lower().strip(".,;:!?")


def text_to_tokens(text):
    return [tok for tok in (normalize_token(t) for t in text.strip().split()) if tok]


def build_phrase_index(lexicon):
    phrase_to_tags = {}
    phrase_lengths = set()

    for tag, phrases in lexicon.items():
        for phrase in phrases:
            normalized_phrase = tuple(normalize_token(tok) for tok in phrase if normalize_token(tok))
            if len(normalized_phrase) == 0:
                continue
            phrase_to_tags.setdefault(normalized_phrase, set()).add(tag)
            phrase_lengths.add(len(normalized_phrase))

    return phrase_to_tags, sorted(phrase_lengths, reverse=True)


KG_ENTITY_PHRASE_TO_TAGS, KG_ENTITY_PHRASE_LENGTHS = build_phrase_index(KG_ENTITY_LEXICON)
KG_RELATION_PHRASE_TO_TAGS, KG_RELATION_PHRASE_LENGTHS = build_phrase_index(KG_RELATION_GROUPS)
KG_TAIL_PHRASE_TO_TAGS, KG_TAIL_PHRASE_LENGTHS = build_phrase_index(KG_TAIL_CONTEXT_LEXICON)


def match_tags_from_tokens(tokens, phrase_to_tags, phrase_lengths):
    tags = set()
    length = len(tokens)

    for n in phrase_lengths:
        if length < n:
            continue
        for i in range(length - n + 1):
            phrase = tuple(tokens[i:i + n])
            if phrase in phrase_to_tags:
                tags.update(phrase_to_tags[phrase])
    return tags


def expand_tail_tags(tags):
    expanded = set(tags)
    for tag in list(tags):
        if tag in KG_TAIL_HIERARCHY:
            expanded.update(KG_TAIL_HIERARCHY[tag])
    return expanded


def match_kg_entities_from_tokens(tokens):
    return match_tags_from_tokens(tokens, KG_ENTITY_PHRASE_TO_TAGS, KG_ENTITY_PHRASE_LENGTHS)


def match_kg_relations_from_tokens(tokens):
    return match_tags_from_tokens(tokens, KG_RELATION_PHRASE_TO_TAGS, KG_RELATION_PHRASE_LENGTHS)


def match_kg_tail_tags_from_tokens(tokens):
    tags = match_tags_from_tokens(tokens, KG_TAIL_PHRASE_TO_TAGS, KG_TAIL_PHRASE_LENGTHS)
    return expand_tail_tags(tags)


def match_kg_entities_from_text(text):
    return match_kg_entities_from_tokens(text_to_tokens(text))


def match_kg_relations_from_text(text):
    return match_kg_relations_from_tokens(text_to_tokens(text))


def match_kg_tail_tags_from_text(text):
    return match_kg_tail_tags_from_tokens(text_to_tokens(text))


def normalize_kg_row(head_text, relation_text, tail_text):
    head_tags = match_kg_entities_from_text(head_text)
    relation_tags = match_kg_relations_from_text(relation_text)
    tail_tags = match_kg_tail_tags_from_text(tail_text)
    return head_tags, relation_tags, tail_tags


def load_normalized_kg_triples(csv_path):
    """
    Load the LEVIR KG CSV and normalize each row into canonical triples.
    The source CSV is GBK-encoded.
    """
    triples = []

    with open(csv_path, "r", encoding="gbk", newline="") as f:
        rows = list(csv.reader(f))[1:]

    for head_text, relation_text, tail_text in rows:
        head_tags, relation_tags, tail_tags = normalize_kg_row(head_text, relation_text, tail_text)

        if len(head_tags) == 0 or len(relation_tags) == 0 or len(tail_tags) == 0:
            continue

        for head_tag in head_tags:
            for relation_tag in relation_tags:
                for tail_tag in tail_tags:
                    triples.append((head_tag, relation_tag, tail_tag))

    return triples


def build_kg_support_maps(csv_path):
    """
    Build reusable statistics for a future KG reward:
      - normalized triple counts
      - entity / relation counts
      - (head, relation) -> allowed tails
      - (relation, tail) -> allowed heads
    """
    triples = load_normalized_kg_triples(csv_path)

    triple_counter = Counter(triples)
    entity_counter = Counter()
    relation_counter = Counter()
    head_relation_to_tails = defaultdict(set)
    relation_tail_to_heads = defaultdict(set)

    for head_tag, relation_tag, tail_tag in triples:
        entity_counter[head_tag] += 1
        entity_counter[tail_tag] += 1
        relation_counter[relation_tag] += 1
        head_relation_to_tails[(head_tag, relation_tag)].add(tail_tag)
        relation_tail_to_heads[(relation_tag, tail_tag)].add(head_tag)

    return {
        "triples": triples,
        "triple_counter": triple_counter,
        "entity_counter": entity_counter,
        "relation_counter": relation_counter,
        "head_relation_to_tails": head_relation_to_tails,
        "relation_tail_to_heads": relation_tail_to_heads,
    }
