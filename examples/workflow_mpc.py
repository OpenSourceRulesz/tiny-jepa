"""Workflow example: model-based planning patterns with tiny-jepa.

Demonstrates the three integration primitives — imagine(), predict_reward(),
dream() — and shows, honestly, which planning pattern works on a tiny model:

  * perceive-the-present + known dynamics  -> works (~8 pellets/episode)
  * multi-step MPC on imagined futures      -> the open frontier (the tiny
    predictor drifts; this is exactly what hierarchical JEPA is for)

Run:  python examples/workflow_mpc.py
"""

import itertools

import numpy as np

from tiny_jepa import TinyJEPA, OrbyWorld

model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")

# ---- pattern 1: the controller that works ---------------------------------
# Perceive the pellet in the REAL frame (accurate: ~3.6px error), simulate
# each move with known body kinematics, pick the best. This is model.plan().
env = OrbyWorld(seed=123)
eaten = 0
for t in range(64):
    a = model.plan(env.render(), agent_xy=env.agent)
    _, r = env.step(a)
    eaten += int(r > 0.5)
print(f"[pattern 1] perceive + plan: {eaten} pellets / 64 steps")

# ---- pattern 2: MPC over imagined futures (the JEPA dream) ----------------
# Score every 2-action sequence with the learned reward head. This is the
# textbook JEPA planning loop — and on this tiny model it's the weak link:
# the reward head only weakly predicts eating (0.83 vs 0.32), so MPC under-
# performs the simple controller above. Included so you can feel exactly
# where the frontier is — and beat it.
env = OrbyWorld(seed=123)
eaten = 0
for t in range(64):
    frame = env.render()
    z = model.encode(frame)
    best_seq, best_score = (0, 0), -1e9
    for seq in itertools.product(range(5), repeat=2):
        future = model.imagine(z, seq)
        score = (model.predict_reward(z, seq[0])
                 + model.predict_reward(future[0], seq[1]))
        if score > best_score:
            best_score, best_seq = score, seq
    _, r = env.step(best_seq[0])
    eaten += int(r > 0.5)
print(f"[pattern 2] 2-step MPC on imagined futures: {eaten} pellets / 64 steps")
print("            (weak reward head -> weak MPC; hierarchy/CEM is the fix)")

# ---- pattern 3: counterfactual dreaming (debugging / visualization) -------
# "What does the model THINK happens if I always go right vs always left?"
frame = env.render()
for name, seq in [("right x3", [4, 4, 4]), ("left x3", [3, 3, 3])]:
    dreams = model.dream(frame, seq)
    means = [round(float(d.mean()), 3) for d in dreams]
    print(f"[pattern 3] dream '{name}': frame brightness {means}")
