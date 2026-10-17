"""tiny-Marble demo: prompt-to-world.

Type words -> walk through the 3D world they describe.
"a dark cavern with lava" -> a dark cavern, with lava, that you
can walk through in first person.

Usage: .venv/bin/python examples/prompt_demo.py ["your prompt here"]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from tiny_jepa.worldgen import dream_from_prompt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a dark cavern with lava"
    world = dream_from_prompt(prompt, seed=7)
    print(f"prompt: '{prompt}' -> {world.attrs}")

    # find a good starting view (slight downward pitch to see floor)
    world.pitch = -0.25
    best, bv = -1, 0
    for y in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        world.yaw = float(y)
        f = world.render()
        if f.var() > best:
            best, bv = f.var(), float(y)
    world.yaw = bv

    frames = []
    for t in range(96):
        before = world.pos.copy()
        # explore: weave + turn away from walls
        world.step(forward=1.0, dyaw=0.12 * np.sin(t * 0.3) + 0.05)
        if np.allclose(before[:2], world.pos[:2]):
            world.turn(0.9)
        if t % 2 == 0:
            frames.append(world.render())

    print(f"{len(frames)} frames")
    # taichi renders at 256x256 natively; numpy at 64x64 (upscale)
    imgs = []
    for f in frames:
        im = Image.fromarray((f * 255).astype("uint8"))
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.NEAREST)
        imgs.append(im)
    out = os.path.join(HERE, "prompt_world.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                  duration=120, loop=0)
    print("saved", out)


if __name__ == "__main__":
    main()
