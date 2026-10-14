"""Cross-entropy-method (CEM) model-predictive control for tiny-jepa.

This is the "Mode-2" planning story from Yann LeCun's JEPA program: instead of
reacting greedily to the current frame (TinyJEPA.plan), the agent optimizes a
distribution over *action sequences* against the model's own imagined futures,
executes the first action of the best sequence, then replans (MPC).

Algorithm (standard CEM over a discrete action space):
  1. Encode the current frame -> latent z0.
  2. Hold a categorical distribution (logits) over actions for each timestep
     of the horizon, initialized uniform.
  3. Repeat n_iters: sample n_samples sequences, roll each through
     model.imagine() from z0, score each with discounted predicted reward
     along its imagined trajectory, refit the logits to the top n_elite
     sequences (MLE with Laplace smoothing).
  4. Return the first action of the best sequence seen.

MEASURED STATUS (orby_v1, 5 seeds x 64 steps, 2026-10-14) — read before use:
  greedy plan():  8.8 pellets/episode
  cem_plan():     0.0 pellets/episode

The optimizer itself is correct (test_cem_finds_known_optimum proves it finds
the optimum against a known reward landscape). The blocker is the objective:
the reward head separates eat/non-eat frames well on REAL latents (0.86 vs
0.31) but collapses to all-negative predictions on IMAGINED latents — even
one step out — and the imagined-score ranking carries no signal about true
outcomes (mean rank correlation +0.05 over 360 sampled sequences). CEM is
rank-based, so it dutifully optimizes noise and converges on standing still.

In other words: the planner works, the value function doesn't transfer to
imagination. That gap is challenge 06 in challenges/ — "No pellet head
(LeCun's challenge)". The fix lives in the value head (e.g. train it on
imagined latents), not in the planner. Ship your fix and take the
leaderboard.
"""

import numpy as np


def _softmax(logits):
    m = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=-1, keepdims=True)


def cem_plan(model, frame, horizon=6, n_samples=48, n_elite=10, n_iters=3,
             gamma=0.97, seed=None, return_info=False):
    """Plan one action with CEM over imagined futures.

    Args:
        model: a TinyJEPA instance.
        frame: (32,32) current observation.
        horizon: planning horizon in steps.
        n_samples: sequences sampled per CEM iteration.
        n_elite: elite sequences used to refit the distribution.
        n_iters: CEM refit iterations.
        gamma: discount on predicted reward along the trajectory.
        seed: int or None for reproducible sampling.
        return_info: if True, also return a dict with diagnostics
            (best_score, best_seq, elite_mean per iteration).

    Returns:
        int action in [0, model.n_actions). With return_info, a tuple
        (action, info).
    """
    rng = np.random.default_rng(seed)
    n_actions = model.n_actions
    z0 = model.encode(frame)
    discounts = gamma ** np.arange(horizon, dtype=np.float32)

    logits = np.zeros((horizon, n_actions), dtype=np.float64)
    best_score = -np.inf
    best_seq = None
    elite_means = []

    for _ in range(n_iters):
        probs = _softmax(logits)
        # sample (n_samples, horizon) action sequences, one categorical per step
        cum = np.cumsum(probs, axis=1)
        u = rng.random((n_samples, horizon, 1))
        seqs = (u < cum[None]).argmax(axis=2).astype(np.int64)

        scores = np.empty(n_samples, dtype=np.float64)
        for s in range(n_samples):
            seq = seqs[s]
            future = model.imagine(z0, seq)          # (horizon, latent_dim)
            total = 0.0
            zt = z0
            for t in range(horizon):
                zt = future[t]
                total += discounts[t] * model.predict_reward(zt, int(seq[t]))
            scores[s] = total
            if total > best_score:
                best_score = total
                best_seq = seq.copy()

        elite_idx = np.argsort(scores)[-n_elite:]
        elite_means.append(float(scores[elite_idx].mean()))
        # MLE refit with Laplace smoothing so no action ever hits zero prob
        counts = np.zeros((horizon, n_actions), dtype=np.float64)
        for idx in elite_idx:
            for t in range(horizon):
                counts[t, seqs[idx, t]] += 1.0
        logits = np.log((counts + 1.0) / (n_elite + n_actions))

    action = int(best_seq[0])
    if return_info:
        return action, {
            "best_score": float(best_score),
            "best_seq": [int(a) for a in best_seq],
            "elite_mean": elite_means,
            "horizon": horizon,
        }
    return action
