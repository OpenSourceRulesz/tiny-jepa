"""tiny-Marble: prompt-to-world for tiny-4D.

Text -> attributes -> voxels. The attribute vector is the bridge:
words map to a small vocabulary of world-attributes (Marble-style),
and the generator conditions on attributes. Phase 1 uses keyword
parsing + procedural generation (real prompt-to-world today); Phase 2
will train the SlicePrior conditioned on attributes instead of style.

Like Marble: "a dark cavern with lava" -> a walkable 3D world.
Tiny: pure numpy, no API calls, runs on a laptop.
"""

import re

import numpy as np

from .voxel import (G, VoxelWorld, procedural_voxel, dream_voxel_world,
                    EMPTY, LAVA, WATER)

__all__ = ["parse_prompt", "dream_from_prompt", "ATTRIBUTES"]

# attribute vocabulary: the words we understand -> world features
ATTRIBUTES = {
    "style": ["rooms", "cavern", "arena"],
    "lava": False,
    "water": False,
    "dark": False,
    "pillars": False,
}

_KEYWORDS = [
    # (pattern, attribute, value)
    (r"\b(dungeon|rooms?|castle|fortress|maze)\b", "style", "rooms"),
    (r"\b(cavern|cave|grotto|tunnel)s?\b", "style", "cavern"),
    (r"\b(arena|open|field|plaza|courtyard)\b", "style", "arena"),
    (r"\b(lava|magma|fire|burning|volcan)\w*\b", "lava", True),
    (r"\b(water|lake|pool|river|ocean|pond)\b", "water", True),
    (r"\b(dark|night|spooky|creepy|gloom|shadow)\w*\b", "dark", True),
    (r"\b(pillars?|columns?)\b", "pillars", True),
]


def parse_prompt(text):
    """'a dark cavern with lava' -> {style: 'cavern', lava: True, ...}."""
    attrs = {"style": "arena", "lava": False, "water": False,
             "dark": False, "pillars": False}
    low = text.lower()
    for pat, key, val in _KEYWORDS:
        if re.search(pat, low):
            attrs[key] = val
    return attrs


def _add_pools(v, rng, material, n_pools):
    """Scatter glowing pools on the floor (z=1)."""
    for _ in range(n_pools):
        cx, cy = int(rng.integers(2, G - 2)), int(rng.integers(2, G - 2))
        r = int(rng.integers(1, 3))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    x, y = cx + dx, cy + dy
                    if 1 <= x < G - 1 and 1 <= y < G - 1:
                        if v[x, y, 1] == EMPTY:
                            v[x, y, 1] = material


def dream_from_prompt(text, seed=None):
    """'a dark cavern with lava' -> walkable VoxelWorld. Tiny Marble."""
    rng = np.random.default_rng(seed)
    attrs = parse_prompt(text)
    style_idx = {"rooms": 0, "cavern": 1, "arena": 2}[attrs["style"]]
    v, _ = procedural_voxel(style=style_idx,
                            seed=int(rng.integers(1e9)))
    if attrs["lava"]:
        _add_pools(v, rng, LAVA, int(rng.integers(2, 5)))
    if attrs["water"]:
        _add_pools(v, rng, WATER, int(rng.integers(2, 4)))
    if attrs["pillars"]:
        for _ in range(int(rng.integers(4, 8))):
            x, y = int(rng.integers(2, G - 2)), int(rng.integers(2, G - 2))
            h = int(rng.integers(4, 9))
            if (v[x, y, 1:3] == EMPTY).all():
                v[x, y, 1:1 + h] = 1  # WALL
    # clear a spawn pocket
    cx, cy = int(rng.integers(4, G - 4)), int(rng.integers(4, G - 4))
    v[cx - 2:cx + 3, cy - 2:cy + 3, 1:7] = EMPTY
    ambient = 0.55 if attrs["dark"] else 1.0
    world = VoxelWorld(v, spawn=(cx, cy),
                       seed=int(rng.integers(1e9)), ambient=ambient)
    world.prompt = text
    world.attrs = attrs
    return world
