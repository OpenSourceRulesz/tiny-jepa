# tiny-spatial v1 — SPEC

A benchmark for **spatial intelligence** in tiny world models: object
permanence, occlusion reasoning, spatial memory. Inspired by Fei-Fei Li's
spatial-intelligence thesis and by EpicWorldModel's finding (Feng et al.,
Oct 2026) that deterministic world models don't just get uncertain under
occlusion — they hallucinate memorized positions with false confidence.

## Design principles

- A probe is a **controlled experiment**, not vibes: scripted pellet paths,
  fixed seeds, a no-occlusion control where one is needed.
- Uses only the public `TinyJEPA` API (`encode`, `imagine`,
  `perceive_pellet`) plus two new environment variants. No retraining.
- Every probe returns a score in [0, 1], higher = better.
  `spatial_iq` = mean of the five probe scores.
- Runs in minutes on a laptop CPU. Fully deterministic given the seed list.
- The benchmark **documents gaps honestly**: probes 4–5 are aspirational for
  flat models and are the motivation for hierarchical JEPA.

## Environments (`tiny_jepa/bench/envs.py`)

### OcclusionWorld(OrbyWorld)
- Wall rect, default x∈[13,19], y∈[4,28] — a vertical bar near the center.
- `render()` draws the wall *over* entities as flat 0.3 gray: anything
  behind it is invisible in the frame.
- `pellet_visible()` → bool: pellet center outside the wall rect.
- Probes override `env.pellet` directly (scripted paths). This is a
  controlled experiment; natural pellet physics would never cross the wall
  on its own (pellets don't move until eaten).

### TwoRoomWorld(OrbyWorld)
- Horizontal wall y∈[14,18], full width, with a corridor gap x∈[13,19].
- **Fog of war:** `render()` hides balls/pellet that are not in the agent's
  room (agent always drawn, walls always drawn; an agent inside the
  corridor band sees everything).
- `step()` blocks wall crossing: a move ending inside the wall rect
  outside the corridor gap is reverted.
- `_respawn_pellet(room=None)`: constrain spawns to room 0 (y<14) or
  room 1 (y>18).

## Probes (`tiny_jepa/bench/probes.py`)

### P1 — Object permanence
*Can the agent still hunt when the pellet keeps disappearing?*
- Natural 64-step episodes in `OcclusionWorld` vs plain `OrbyWorld`
  (20 episodes each), greedy `model.plan`.
- When the pellet goes behind the wall, perception fails; the right
  behavior is to fall back on last-seen position (the pellet is static,
  and eating through the wall is allowed by the physics).
- `score = occluded eat-rate / clean eat-rate`. End-to-end, public API
  only, no probe training. **Pass ≥ 0.5.**

### P2 — Hallucination
*When the pellet is fully hidden, where does perception point?*
- Natural `OcclusionWorld` episodes (120 × 64 steps, random actions —
  the larger count stabilizes the estimate; expect ±0.05 seed wobble);
  on steps where `pellet_visible()` is false, record
  `perceive_pellet(frame)`.
- `center_bias` = fraction of hidden-step predictions within 6px of the
  canvas center (the dataset prior). `score = 1 - center_bias`.
  **Pass: score ≥ 0.7.**
- This is EpicWorldModel's failure mode, made measurable: deterministic
  models don't get vague under occlusion, they hallucinate memorized
  positions with false confidence. Caveat: the wall is mildly
  out-of-distribution for the head, so the probe documents the shipped
  system's behavior whatever the cause.

### P3 — Surprise
*Does the model notice when the world violates its expectations?*
- Natural 64-step episodes on `OrbyWorld` (20 episodes).
- `surprise_t = 1 - cos(imagine(z_t, [a_t])[0], z_{t+1})`.
- Ground-truth violation events: pellet respawns (teleports).
- `detection_auc` = AUC of surprise separating respawn steps from normal
  steps. **Pass: AUC ≥ 0.7.**

### P4 — Spatial memory
*Can the planner act on a pellet it cannot currently see?*
- `TwoRoomWorld`: pellet in room 0, agent starts in room 1 **away from
  the corridor column** (no free lunch — reaching room 0 requires
  finding the corridor first). Greedy `model.plan(frame, agent)` for
  128 steps, 20 episodes.
- `success_rate` = fraction of episodes where the pellet is eaten.
  Also reports room-0 coverage (did it even go look?).
- **Pass ≥ 0.3** — aspirational; the flat greedy planner is expected to
  score ≈ 0. This probe is the reason hierarchical JEPA exists.

### P5 — Memory update
*When the world changes behind its back, does the model update or cling?*
- `TwoRoomWorld`. Phase 1: agent in room 0 observes pellet at p1
  (5 steps). Phase 2: agent teleported to room 1; pellet moved to p2
  (same room, new spot). Phase 3: agent at corridor, planner runs 96 steps.
- `update_rate` = fraction of episodes where the agent eats the pellet
  (finds p2) rather than perseverating at stale p1.
- **Pass ≥ 0.3** — aspirational, same rationale as P4.

## Report card

```python
from tiny_jepa.bench import run_benchmark
report = run_benchmark("tiny_jepa/checkpoints/orby_v1.npz")
report.print()   # per-probe scores, pass/fail, spatial_iq
report.to_json("bench_orby_v1.json")
```

## Baselines (measured 2026-10-14)

| Probe | orby_v1 (flat) | orby_hier_v1 | Pass |
|---|---|---|---|
| P1 permanence | 0.531 ✅ | 0.613 ✅ | ≥ 0.5 |
| P2 hallucination | 0.581 ❌ | 0.581 ❌ | ≥ 0.7 |
| P3 surprise | 0.841 ✅ | 0.841 ✅ | ≥ 0.7 |
| P4 spatial_memory | 0.200 ❌ | **0.300 ✅** | ≥ 0.3 |
| P5 memory_update | 0.350 ✅ | **0.550 ✅** | ≥ 0.3 |
| **spatial_iq** | **0.501** | **0.577** | — |
| clean pellets/ep | 8.85 | 6.55 | — |

The story: the model notices surprises and limps along under occlusion,
but it half-hallucinates the prior when blind, can't hold spatial memory
across rooms, and barely updates stale memory. P2/P4 are the gaps that
stochastic prediction (EpicWorldModel) and hierarchy (H-JEPA) should fill.

## Out of scope for v1
- Partial-view (local window) probe; multi-pellet tracking.
- Proper scoring of predictive variance (needs a stochastic model —
  see EpicWorldModel; that's a challenge-06-adjacent research direction).
- Learned confidence outputs (same).
