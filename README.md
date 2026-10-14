# tiny-jepa

**World Labs for your laptop.** A tiny, hackable, action-conditioned JEPA
(Joint Embedding Predictive Architecture) world model — in pure numpy, zero
required dependencies. The whole thing is ~1,500 lines you can read in an
afternoon. Inspired by Yann LeCun's predictive world-model program and the
spatial-intelligence spirit of Fei-Fei Li's World Labs.

![tiny-jepa hunting pellets](hero.gif)

*The shipped checkpoint, hunting pellets with learned vision — no hand-coded
game AI, just a world model that imagines, perceives, and plans.*

## Install

```bash
pip install tiny-jepa
# extras: pip install "tiny-jepa[onnx]"  (ONNX export)
#         pip install "tiny-jepa[gym]"   (Gymnasium wrapper)
```

## 5-line quickstart

```python
from tiny_jepa import TinyJEPA, OrbyWorld
model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")  # pretrained
env = OrbyWorld(seed=0)
for _ in range(64):
    env.step(model.plan(env.render(), env.agent))  # see, imagine, act
```

That's it — a world model hunting pellets from raw pixels. Now go crazy:
`model.imagine(z, actions)` rolls out latent futures, `model.dream(frame,
actions)` decodes them to images, `train_on_env` / `train_on_data` train
your own worlds.

Try it with zero installs: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenSourceRulesz/tiny-jepa/blob/main/examples/quickstart_colab.ipynb)
— or drive it yourself in the [playable web demo](https://github.com/OpenSourceRulesz/tiny-jepa#playable-demo).

## What's inside

| Module | What it does |
|---|---|
| `TinyJEPA` | `encode` / `imagine` / `decode` / `dream` / `perceive_pellet` / `predict_reward` / `plan` / `save` / `load` |
| `train` | `train_on_env` (built-in world), `train_on_data` (your own trajectories) |
| `export_onnx` | Bit-accurate ONNX export for any runtime |
| `envs` | `OrbyWorld` (built-in 2D world) + `GymWrapper` |
| `checkpoints/orby_v1.npz` | Pretrained: ~8.6 pellets/episode vs ~0.2 random |
| `challenges/` | Quests. Start here if you want to make your mark |
| `tiny_jepa/bench` | **tiny-spatial**: 5-probe spatial-intelligence benchmark (SPEC.md) |

## tiny-spatial — does your world model understand space?

```python
from tiny_jepa.bench import run_benchmark
run_benchmark("tiny_jepa/checkpoints/orby_v1.npz").print()
```

Five probes — object permanence under occlusion, hallucination (does it
collapse to the dataset prior when blind?), surprise at expectation
violations, spatial memory across rooms, memory updates. The shipped
checkpoint scores **spatial_iq 0.501**: it notices surprises and limps
along under occlusion, but half-hallucinates when blind and can't do
rooms. Beat it.

## Hierarchical JEPA — two timescales

`tiny_jepa/hjepa.py` adds a slow level on top of the frozen fast model:
every 8 steps it picks a subgoal (pellet when visible, novel territory
or last-seen position when not) using a learned command-conditioned
forward model, then steers to it greedily with event-driven replans.

```python
from tiny_jepa import HierarchicalJEPA
hier = HierarchicalJEPA.load("tiny_jepa/checkpoints/orby_v1.npz",
                             "tiny_jepa/checkpoints/orby_hier_v1_slow.npz")
hier.plan(frame, agent_xy)   # drop-in for TinyJEPA
```

| | flat orby_v1 | hierarchical |
|---|---|---|
| clean pellets/ep | 8.85 | 6.55 |
| tiny-spatial IQ | 0.501 | **0.577** |
| P4 rooms | 0.20 ❌ | **0.30 ✅** |
| P5 memory update | 0.35 | **0.55** |

The tradeoff is principled: hierarchy costs a little reactive hunting
skill and buys spatial reasoning — rooms, exploration, last-seen
memory. Same story as H-JEPA vs flat JEPA in the literature.

## CEM planning — the Mode-2 primitive (and its honest failure)

`tiny_jepa/planning.py` ships `cem_plan()`: cross-entropy-method MPC that
optimizes a distribution over action sequences against the model's own
imagined futures — no ground-truth kinematics, the pure JEPA planning loop.

```python
from tiny_jepa import cem_plan
action = cem_plan(model, frame)   # ~35ms/step on CPU, pure numpy
```

The optimizer is proven correct (it finds the optimum against a known
reward landscape in `tests/test_planning.py`). Against orby_v1's learned
reward head it scores **0.0 pellets/episode** vs 8.8 for the greedy
controller — and that measurement is the point. Diagnosis: the reward head
separates eat/non-eat on real latents (0.86 vs 0.31) but collapses on
imagined latents, and imagined-score ranking carries no signal about true
outcomes (rank correlation +0.05). The planner is exonerated; the value
head is the suspect. Closing that gap is
[challenges/06-no-pellet-head.md](challenges/06-no-pellet-head.md) — the
most important open problem in the repo, and the leaderboard is empty.

## Challenges — make your mark

`challenges/` has concrete quests with success metrics, from "beat the
greedy planner" to "teach it object permanence." The leaderboard is empty.
Your move.

## Honest limits

32-dim latent, ~200k params, toy 2D world — not a general world model.
Dreams blur with horizon. The greedy planner steers with known kinematics;
the shipped CEM planner does pure imagined-future planning but scores 0.0
because the reward head doesn't transfer to imagined latents (see above).
The architecture is the real JEPA idea end to end, and it's yours to extend:
the imagination-gap value head is the obvious next frontier
(see `challenges/`).

## Citation

```bibtex
@software{tiny_jepa_2026,
  title  = {tiny-jepa: a tiny, hackable, action-conditioned JEPA world model},
  author = {OpenSourceRulesz},
  year   = {2026},
  url    = {https://github.com/OpenSourceRulesz/tiny-jepa}
}
```

MIT licensed.
