"""
Simplified location lexicon for LEVIR-CC.

Design choice:
1. keep only high-frequency location expressions
2. merge very similar variants to reduce complexity
3. keep a separate low-weight tag for "both sides" because the model often overuses it
"""

LOCATION_LEXICON = {
    # coarse absolute positions
    "top": [
        ("at", "the", "top"),
        ("on", "the", "top"),
        ("in", "the", "upper"),
        ("at", "the", "upper"),
    ],
    "bottom": [
        ("at", "the", "bottom"),
        ("on", "the", "bottom"),
    ],
    "left": [
        ("on", "the", "left"),
        ("at", "the", "left"),
    ],
    "right": [
        ("on", "the", "right"),
        ("at", "the", "right"),
    ],
    "center": [
        ("in", "the", "center"),
        ("at", "the", "center"),
        ("in", "the", "middle"),
        ("at", "the", "middle"),
    ],

    # corners: more specific than only top/bottom/left/right
    "top_left_corner": [
        ("top", "left", "corner"),
        ("upper", "left", "corner"),
    ],
    "top_right_corner": [
        ("top", "right", "corner"),
        ("upper", "right", "corner"),
    ],
    "bottom_left_corner": [
        ("bottom", "left", "corner"),
    ],
    "bottom_right_corner": [
        ("bottom", "right", "corner"),
    ],

    # relation-style positions that appear often in LEVIR-CC
    "along_road": [
        ("along", "the", "road"),
        ("along", "the", "roads"),
    ],
    "beside_road": [
        ("beside", "the", "road"),
        ("next", "to", "the", "road"),
        ("near", "the", "road"),
    ],
    "both_sides": [
        ("on", "both", "sides"),
        ("both", "sides", "of", "the", "road"),
        ("on", "both", "sides", "of", "the", "road"),
    ],

    # common scene-region phrases
    "on_bareland": [
        ("on", "the", "bareland"),
    ],
    "in_woods": [
        ("in", "the", "woods"),
        ("in", "the", "forest"),
    ],
}


LOCATION_TAG_WEIGHTS = {
    # specific corners deserve higher reward
    "top_left_corner": 2.0,
    "top_right_corner": 2.0,
    "bottom_left_corner": 2.0,
    "bottom_right_corner": 2.0,

    # coarse directions
    "top": 1.5,
    "bottom": 1.5,
    "left": 1.5,
    "right": 1.5,
    "center": 1.5,

    # relation / region phrases
    "along_road": 1.0,
    "beside_road": 1.0,
    "on_bareland": 1.0,
    "in_woods": 1.0,

    # keep low because it is frequent and easy to over-generate
    "both_sides": 0.5,
}
