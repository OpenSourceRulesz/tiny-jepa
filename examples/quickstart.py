"""Quickstart: load the pretrained tiny-jepa and watch it hunt pellets.

Run:  python examples/quickstart.py
"""

import numpy as np

from tiny_jepa import TinyJEPA, OrbyWorld

model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
env = OrbyWorld(seed=7)

print("action map: 0=stay 1=up 2=down 3=left 4=right\n")
eaten = 0
for t in range(64):
    frame = env.render()
    pellet = model.perceive_pellet(frame)          # learned vision
    action = model.plan(frame, agent_xy=env.agent)  # 1-step model-based plan
    _, reward = env.step(action)
    eaten += int(reward > 0.5)
    if t % 16 == 0:
        print(f"t={t:2d} pellet~{pellet.round(1)} true~{env.pellet.round(1)} "
              f"action={action} eaten={eaten}")
print(f"\npellets eaten: {eaten}/64 steps (random policy: ~0.2)")
