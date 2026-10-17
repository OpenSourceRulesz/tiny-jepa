"""tiny-inspatio demo: roam a dreamed big world, watch the model's
persistent world-memory fill in, and see it hallucinate unseen views.

Left panel: what the agent's 32x32 camera sees.
Right panel: what the MODEL believes is there -- solid walls are
remembered geometry (the implicit spatiotemporal cache), lighter walls
are generative hallucinations of unseen tiles anchored by the known.

Every 24 steps the camera jumps somewhere new: a pure novel-view
prediction, InSpatio-WorldFM style, tiny.

Usage:  .venv/bin/python examples/roam_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from tiny_jepa.worldgen import WorldPrior, dream_big_world, detect_pellet
from tiny_jepa.worldgen.roam import (
    WorldMemory, ViewCompletion, VIEW, BTILE, VTILES)
from tiny_jepa.hjepa import HierarchicalJEPA

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "tiny_jepa", "checkpoints")


def up4(f):
    img = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8), "L")
    return img.resize((VIEW * 4, VIEW * 4), Image.NEAREST)


def main():
    prior = WorldPrior.load(os.path.join(CKPT, "worldgen_v1.npz"))
    vc = ViewCompletion.load(
        os.path.join(CKPT, "roam_completion_v1.npz"))
    hier = HierarchicalJEPA.load(
        os.path.join(CKPT, "orby_v1.npz"),
        os.path.join(CKPT, "orby_hier_v1_slow.npz"))

    world = dream_big_world(prior=prior, style=0, seed=21)
    mem = WorldMemory()
    hier.reset_episode()
    rng = np.random.default_rng(0)

    gif_frames = []
    for t in range(120):
        frame, (x0, y0), av = world.viewport()
        mem.observe(world.map, x0, y0)
        pellet_xy = detect_pellet(frame, av)
        a = hier.plan(frame, av, pellet_xy=pellet_xy)
        world.step(a)

        novel = (t % 24 == 23)
        if novel:  # jump the camera somewhere unseen
            cx0 = int(np.clip(x0 + int(rng.choice([-16, 16])), 0, 32))
            cy0 = int(np.clip(y0 + int(rng.choice([-16, 16])), 0, 32))
            # ground-truth novel view (for the left panel)
            full = world.render_full()
            actual = full[cy0:cy0 + VIEW, cx0:cx0 + VIEW]
        else:
            cx0, cy0 = x0, y0
            actual = frame
        partial, mask = mem.anchored_view(cx0, cy0)
        completed = vc.predict(partial, mask, style=0)
        belief = mem.render_belief(cx0, cy0, completed)

        pair = Image.new("L", (VIEW * 4 * 2, VIEW * 4))
        pair.paste(up4(actual), (0, 0))
        pair.paste(up4(belief), (VIEW * 4, 0))
        gif_frames.append(pair)

    print(f"roamed 120 steps, {world.eaten} pellets, "
          f"map coverage {mem.coverage():.0%}")
    out = os.path.join(HERE, "roam_belief.gif")
    gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:],
                        duration=150, loop=0)
    print("saved", out)


if __name__ == "__main__":
    main()
