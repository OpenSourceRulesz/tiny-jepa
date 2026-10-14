# Challenge 01 — Beat the greedy

**The quest:** Hunt more pellets than the shipped greedy controller.

**Baseline:** `model.plan()` — perceive the pellet with the trained head,
simulate each move's kinematics, pick the closest. Scores **8.6
pellets/episode** (64 steps, averaged over seeds).

**Your move:** Any *learned* controller. Ideas: CEM-based MPC over imagined
latent futures, a better reward head, a policy distilled from the greedy
teacher, a hierarchical planner. No hand-coded game AI — the perception and
decisions must come from a model.

**How to measure:**
```python
from tiny_jepa import TinyJEPA, OrbyWorld
model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
eaten = []
for seed in range(100):
    env = OrbyWorld(seed=seed); e = 0
    for _ in range(64):
        env.step(YOUR_CONTROLLER(env.render(), env))
        # count env reward > 0.5
    eaten.append(e)
print(sum(eaten) / len(eaten))
```

**Win condition:** > 8.6 pellets/episode over 100 seeds. Submit a PR with
your controller + score.

**Stretch:** Do it with *no* position heads (see challenge 06).
