"""Workflow example: model-based planning patterns with tiny-jepa.

Demonstrates the integration primitives — imagine(), predict_reward(),
dream(), cem_plan() — and shows, honestly, which planning pattern works:

  * perceive-the-present + known dynamics  -> works (~8.8 pellets/episode)
  * multi-step MPC on imagined futures      -> the open frontier. CEM is now
    a shipped, tested primitive (pattern 4) — and the measurement says the
    planner is not the problem: the reward head doesn't transfer to imagined
    latents, so imagined-score ranking carries no signal (rank corr +0.05).
    The fix lives in the value head, not the planner. See challenge 06.

Run:  python examples/workflow_mpc.py
"""

import itertools

import numpy as np

from tiny_jepa import TinyJEPA, OrbyWorld, cem_plan

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
print("            (weak reward head -> weak MPC; see pattern 4)")

# ---- pattern 4: CEM-based MPC (the shipped Mode-2 primitive) ---------------
# cem_plan() optimizes a distribution over action sequences against the
# model's own imagined futures — the textbook JEPA planning loop, with a
# correct, tested optimizer (see tests/test_planning.py). Measured on
# orby_v1: 0.0 pellets/episode vs 8.8 for the greedy controller.
# The planner is exonerated; the reward head doesn't transfer to imagined
# latents (rank corr of imagined scores vs true returns: +0.05). Fixing the
# value head is challenge 06 — the most important open problem in the repo.
env = OrbyWorld(seed=123)
eaten = 0
for t in range(64):
    a = cem_plan(model, env.render(), seed=t)
    _, r = env.step(int(a))
    eaten += int(r > 0.5)
print(f"[pattern 4] CEM MPC on imagined futures: {eaten} pellets / 64 steps")
print("            (optimizer correct, value head broken — challenge 06)")

# ---- pattern 3: counterfactual dreaming (debugging / visualization) -------
# "What does the model THINK happens if I always go right vs always left?"
frame = env.render()
for name, seq in [("right x3", [4, 4, 4]), ("left x3", [3, 3, 3])]:
    dreams = model.dream(frame, seq)
    means = [round(float(d.mean()), 3) for d in dreams]
    print(f"[pattern 3] dream '{name}': frame brightness {means}")
