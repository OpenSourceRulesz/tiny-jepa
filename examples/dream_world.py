"""tiny-world demo: dream up new worlds, play in them zero-shot.

1. Samples one world per style from the diffusion prior and saves a
   side-by-side strip (these worlds never existed before).
2. Drops the hierarchical JEPA planner into a dreamed maze with NO
   retraining and records it hunting pellets -> dream_play.gif.

Usage:  .venv/bin/python examples/dream_world.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from tiny_jepa.worldgen import WorldPrior, dream_world, detect_pellet, STYLES
from tiny_jepa.hjepa import HierarchicalJEPA

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "tiny_jepa", "checkpoints")


def upscale(frames, scale=4):
    out = []
    for f in frames:
        img = Image.fromarray((f * 255).astype(np.uint8), "L")
        out.append(img.resize((32 * scale, 32 * scale), Image.NEAREST))
    return out


def main():
    prior = WorldPrior.load(os.path.join(CKPT, "worldgen_v1.npz"))

    # 1. a strip of dreamed worlds, one per style
    strip = []
    for s, name in enumerate(STYLES):
        w = dream_world(prior=prior, style=s, seed=100 + s)
        strip.append(w.render())
        print(f"dreamed {name}: {w.eaten} pellets so far (nobody's played)")
    strip_img = Image.new("L", (32 * 3 * 4, 32 * 4))
    for i, img in enumerate(upscale(strip)):
        strip_img.paste(img, (i * 32 * 4, 0))
    strip_path = os.path.join(HERE, "dream_worlds.png")
    strip_img.save(strip_path)
    print("saved", strip_path)

    # 2. zero-shot play: hierarchical planner in a dreamed rooms world.
    # Perception is physics-grounded (detect_pellet): the learned head
    # never saw walls in training, so we don't ask it to. No retraining
    # anywhere -- the planner meets this world for the first time.
    hier = HierarchicalJEPA.load(
        os.path.join(CKPT, "orby_v1.npz"),
        os.path.join(CKPT, "orby_hier_v1_slow.npz"))
    world = dream_world(prior=prior, style=0, seed=7)
    hier.reset_episode()
    frames = [world.render()]
    for _ in range(64):
        f = world.render()
        a = hier.plan(f, world.agent,
                      pellet_xy=detect_pellet(f, world.agent))
        frame, _ = world.step(a)
        frames.append(frame)
    print(f"zero-shot in a dreamed rooms-world: {world.eaten} pellets "
          f"/ 64 steps")
    gif_path = os.path.join(HERE, "dream_play.gif")
    upscale(frames)[0].save(
        gif_path, save_all=True, append_images=upscale(frames)[1:],
        duration=120, loop=0)
    print("saved", gif_path)


if __name__ == "__main__":
    main()
