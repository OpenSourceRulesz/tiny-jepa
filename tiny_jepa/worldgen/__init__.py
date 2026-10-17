"""tiny-world: a tiny generative world model.

Dream up brand-new playable worlds with a small diffusion prior over
16x16 tile maps, rendered deterministically to 32x32 frames — the
"World Labs for your laptop" cheat: generate the *world description*,
not pixels, so every view is consistent by construction.

    from tiny_jepa.worldgen import dream_world, WorldPrior

    world = dream_world(style=1)          # a dreamed maze, playable
    world = dream_world()                 # any style, procedural fallback

Train your own prior on procedural maps:

    from tiny_jepa.worldgen import WorldPrior, procedural_map
    import numpy as np
    maps = np.stack([procedural_map(s, seed=i)
                     for s in range(3) for i in range(1000)])
    styles = np.array([s for s in range(3) for _ in range(1000)])
    prior = WorldPrior().fit(maps, styles)
    prior.save("tiny_jepa/checkpoints/worldgen_v1.npz")
"""

from .maps import GRID, STYLES, N_STYLES, procedural_map, reachable
from .diffusion import WorldPrior
from .env import GeneratedWorld, dream_world, detect_pellet
from .roam import (BigWorld, dream_big_world, WorldMemory, ViewCompletion,
                   BTILE, BCANVAS, VIEW, VTILES)

__all__ = [
    "GRID", "STYLES", "N_STYLES", "procedural_map", "reachable",
    "WorldPrior", "GeneratedWorld", "dream_world", "detect_pellet",
    "BigWorld", "dream_big_world", "WorldMemory", "ViewCompletion",
    "BTILE", "BCANVAS", "VIEW", "VTILES",
]

# 3D prompt-to-world moved to tiny-marble (separate PyPI package):
# pip install tiny-marble — "a dark cavern with lava" -> walkable 3D world
