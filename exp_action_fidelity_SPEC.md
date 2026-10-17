# Experiment spec: faithful action effects for the arm-world predictor

DRAFT spec — not run, not reviewed. Awaits the user's go-ahead (offered 2026-10-15:
"Say the word and it runs overnight, results by morning"). No response recorded as of
2026-10-16 ~11:05 ET.

## Background (all from delivered research notes, sent 2026-10-15)

- Sim arm world (`tiny_jepa/envs_arm.py`): 2-link planar arm, 5 actions
  (0=stay, 1=j1-, 2=j1+, 3=j2-, 4=j2+), base (16,27), links 10/9, 32×32 canvas.
  Ground truth available every frame: `world.tip` (gripper xy), `world.pellet` (target xy);
  `collect_arm_episodes()` returns tips + pellets alongside frames/actions.
- Perception/imagination TRANSFER to the arm: 2.8px target error, 4.0px tip error;
  imagined-vs-true latent cosine 0.84 at 8 steps — but that cosine is misleading
  (static scene content dominates the latent).
- Control does NOT transfer: reward-head MPC scores 0.0 targets/episode.
- Mechanism diagnosis (corrected 2026-10-15 night): the blocker is the PREDICTOR, not the
  heads. Imagined action effects have near-zero directional fidelity — cosine between
  imagined and true tip displacement is 0.0–0.3 for three of four joint actions.
  "Little moves" problem: joint 2's sub-2px visual changes give the JEPA loss no pressure
  to learn fine action effects.
- Sanity numbers: FK-greedy with true kinematics gets 0.56 targets/episode
  (4.7× random), so the workspace is reachable; imagination-based planning is the gap.
- The previous experiment (`exp_imagined_heads.py`) trained tip/target heads on imagined
  latents (readout error 8–11px → ~3–4px at 6 steps, no regression on real frames);
  checkpoints `arm_v2.npz` hold those heads.

## Proposed experiment: `exp_action_fidelity.py`

Goal: make the predictor carry faithful action effects, measured by action-effect
directional fidelity, then re-run the arm MPC scoreboard.

### Design A (primary): auxiliary tip-displacement loss
1. Load `arm_v2.npz` (heads trained on imagined latents — the honest readout).
2. Freeze encoder + heads. Fine-tune ONLY the predictor on arm episodes with:
   loss = JEPA_loss + λ · mean over transitions of
     || (tip_head(z_hat_{t+1}) − tip_head(z_hat_t)) − (tips[t+1] − tips[t]) ||²
   where z_hat comes from `model.imagine(z0, actions)` rollouts
   (`model.py::imagine` returns (H, latent_dim)), and `tip_head` is the frozen
   arm_v2 head readout (`model.nets["agenth"]` forward × CANVAS, per exp_imagined_heads.py).
3. λ schedule: start 1.0, grid {0.3, 1.0, 3.0} on a validation split.

### Design B (alternative, cheaper): action-conditional JEPA emphasis
Same fine-tune but NO auxiliary loss; instead weight the standard JEPA loss ×k
(k grid {3, 10}) on transitions whose action moves joint 2 (actions 3, 4) —
countering the "little moves" gradient starvation directly.

### Controls
- Same fine-tune with λ=0 / k=1 (standard JEPA only): isolates whether the auxiliary
  signal is the cause, vs. more training steps.
- Same arm MPC scoreboard from exp_imagined_heads.py (perception-based CEM) for all arms.

### Metrics to beat
1. **Action-effect fidelity** (primary): per action a∈{1,2,3,4},
   cosine(imagined Δtip, true Δtip) averaged over validation transitions.
   Baseline: 0.0–0.3 on 3 of 4 actions. Target: ≥0.6 on ≥3/4 actions.
2. **Arm MPC scoreboard**: targets/episode. Current 0.0 (MPC), FK-greedy 0.56, random ~0.12–0.25.
3. **No-regression guard**: imagined-vs-real head readout error must not regress
   beyond 4–5px at 6 steps (the arm_v2 level).

### Success criterion
Fidelity ≥0.6 on ≥3/4 actions AND MPC strictly beats the λ=0 control.
If met: imagination-based planning is unblocked — green light to push toward
real-arm work under the standing rule (no imagination-based planning touches
hardware until the planner reads task-aligned latent geometry).
If not: evidence the toy predictor capacity/depth is the limit — feed into the
tiny-jepa v-next architecture call.

### Practical notes
- Output: `results_action_fidelity.json` + `checkpoints/arm_v3.npz`.
- Cost: same order as the imagined-heads run (2500 steps, pure numpy) — overnight-safe.
- Both designs are $0, local-only, no new deps — fits the project's constraints.

## Open decision (the user's)
"Say the word" → build `exp_action_fidelity.py` from this spec, run A+B+controls
overnight, report fidelity curves + scoreboard by morning.
