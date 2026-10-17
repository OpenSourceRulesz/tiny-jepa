"""GWMR-style latent-goal CEM planner for tiny-jepa.

The existing cem_plan() (planning.py) scores imagined futures with the
reward head, which collapses on imagined latents (all-negative predictions
even one step out) -- so CEM optimizes noise. This module takes the GWMR
approach instead: plan directly in latent space by minimizing COSINE
DISTANCE between the imagined future latent and a GOAL embedding, with no
reward head involved at all.

  z_goal  = encode(render(agent ON pellet))     # "what success looks like"
  cost    = 1 - cos(z_imagined_final, z_goal)   # lower is better
  a*      = CEM argmin over action sequences, execute first, replan (MPC)

Why this should work where reward-head CEM failed: the maze_v1 encoder
already separates across-wall from open-adjacent state pairs (3.2x latent
distance ratio, 97% separability -- see results_temporal.json), so the
geometry CEM needs is present in the latents. The reward head was the
broken component; this planner routes around it entirely.

References: GWMR (dase8601/GWMR) -- SIGReg + Transformer + CEM planner
minimizing cosine distance to goal embeddings, 50% MiniGrid-DoorKey.

Update (results_predictor.json): absolute distance scoring has a degenerate
optimum -- the predictor is most accurate when nothing changes, so the
imagined "stay" latent always beats any "move" latent and CEM converges to
[0,0,0,...] (0.00 pellets). score_type="progress" fixes this by scoring
dist(z_current, z_goal) - dist(z_imagined, z_goal): staying still scores ~0,
moving toward the goal scores positive.
"""

import numpy as np

from .model import one_hot


def _softmax(logits):
    m = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=-1, keepdims=True)


def _l2_dist(A, b):
    """Euclidean distance. A: (K, D), b: (D,) -> (K,). Lower = closer."""
    return np.linalg.norm(A - b[None], axis=1)


def _cosine_dist(A, b):
    """1 - cosine similarity. A: (K, D), b: (D,) -> (K,). Lower = closer."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(A, axis=1) * (np.linalg.norm(b) + 1e-12) + 1e-12
    return 1.0 - (A @ b) / denom


def _score_fn(A, b, metric):
    if metric == "l2":
        return _l2_dist(A, b)
    return _cosine_dist(A, b)


def build_goal_embedding(model, pellet_xy, n_jitter=3, jitter=1.0, seed=0):
    """Encode "agent on the pellet" -> goal latent z_goal.

    Constructs a MazeWorld with the agent placed at pellet_xy (plus small
    Gaussian jitter, averaged), renders it, and encodes. Walls are identical
    across MazeWorld instances (class constant), so any instance works.
    """
    from .envs_newworlds import MazeWorld
    rng = np.random.default_rng(seed)
    pellet_xy = np.asarray(pellet_xy, dtype=np.float64)
    embs = []
    for _ in range(n_jitter):
        w = MazeWorld()
        w.agent = np.clip(pellet_xy + rng.normal(0, jitter, size=2), 2.5, 29.5)
        w.pellet = pellet_xy.copy()
        embs.append(model.encode(w.render()))
    return np.mean(np.stack(embs), axis=0).astype(np.float32)


def plan_cem_latent_goal(model, frame, pellet_xy=None, z_goal=None,
                         horizon=6, n_samples=200, n_elite=20, n_iters=5,
                         seed=None, score_mode="final", metric="l2",
                         score_type="absolute", return_info=False,
                         value_head=None):
    """One CEM planning step toward a latent goal embedding.

    Args:
        model: a TinyJEPA instance (use maze_v1 -- its encoder has the wall
            geometry; NOT maze_temporal).
        frame: (32,32) current observation.
        pellet_xy: (2,) pellet position in pixels. If None, uses the model's
            own perceive_pellet (learned vision, ~6px error on maze_v1).
        z_goal: precomputed goal embedding (saves one encode per step).
        value_head: optional ValueHead instance. When given, candidates are
            scored by -V(z_imagined, z_goal) (CEM minimizes, so this
            maximizes predicted success probability) instead of any
            latent-distance metric. This is the task-aligned alternative to
            the broken distance metric: V is *learned* from outcomes, so it
            can't point the wrong way the way raw latent distance did.
            Overrides score_type/metric for scoring (d0 unused).
        horizon, n_samples, n_elite, n_iters: CEM hyperparameters.
        seed: int or None.
        score_mode: "final" scores the last imagined latent (faithful GWMR);
            "min" scores the closest approach along the trajectory (denser
            signal when the goal is beyond one horizon).
        metric: "l2" (default -- matches this encoder's geometry; cosine
            distance is ~0 for all states because pellet+wall features
            dominate latent direction, see results_gwmr_planner.json) or
            "cosine" (faithful GWMR, requires hyperspherical latents).
        score_type: "absolute" (default, legacy) scores absolute
            distance-to-goal of the imagined future -- LOWER is better.
            "progress" scores PROGRESS toward the goal:
            dist(z_current, z_goal) - dist(z_imagined, z_goal) -- HIGHER is
            better (negated internally so CEM still minimizes). Staying
            still scores exactly ~0; moving toward the goal scores positive.
            This fixes the stay-exploit: the predictor is most accurate when
            nothing changes, so under absolute scoring the imagined "stay"
            latent always beats any "move" latent and CEM converges to
            [0,0,0,...]. Progress scoring removes that degenerate optimum.
        return_info: if True, also return diagnostics dict.

    Returns:
        int action in [0, model.n_actions). With return_info, (action, info).
    """
    rng = np.random.default_rng(seed)
    n_actions = model.n_actions
    D = model.latent_dim
    z0 = model.encode(frame).astype(np.float32)
    if z_goal is None:
        if pellet_xy is None:
            pellet_xy = model.perceive_pellet(frame)
        z_goal = build_goal_embedding(model, pellet_xy)
    zg = np.asarray(z_goal, dtype=np.float32)
    if score_type not in ("absolute", "progress"):
        raise ValueError(f"score_type must be 'absolute' or 'progress', got {score_type!r}")
    # Baseline distance for progress scoring: staying still scores ~0.
    d0 = float(_score_fn(z0[None], zg, metric)[0]) if score_type == "progress" else None

    logits = np.zeros((horizon, n_actions), dtype=np.float64)
    best_score = np.inf
    best_seq = None
    elite_means = []
    first_iter_spread = None

    for it in range(n_iters):
        probs = _softmax(logits)
        cum = np.cumsum(probs, axis=1)
        u = rng.random((n_samples, horizon, 1))
        seqs = (u < cum[None]).argmax(axis=2).astype(np.int64)

        # Batched rollout: (K, D) latents through the predictor MLP.
        Z = np.tile(z0, (n_samples, 1)).astype(np.float32)
        if score_mode == "min":
            traj = np.empty((n_samples, horizon, D), dtype=np.float32)
        for t in range(horizon):
            oh = one_hot(seqs[:, t], n_actions).astype(np.float32)
            Z = model.nets["pred"].forward(
                np.concatenate([Z, oh], axis=1)).astype(np.float32)
            if score_mode == "min":
                traj[:, t] = Z
        if value_head is not None:
            # Task-aligned scoring: maximize V(z_imagined, z_goal).
            # CEM minimizes, so scores = -V.
            zg_b = np.tile(zg, (n_samples, 1)).astype(np.float32)
            if score_mode == "final":
                v = value_head.predict(Z, zg_b)
            else:
                v = value_head.predict(
                    traj.reshape(-1, D),
                    np.tile(zg, (n_samples * horizon, 1)).astype(np.float32),
                ).reshape(n_samples, horizon).max(axis=1)
            scores = -v
        elif score_mode == "final":
            d = _score_fn(Z, zg, metric)
        else:
            d = _score_fn(traj.reshape(-1, D), zg, metric).reshape(n_samples, horizon).min(axis=1)
        if value_head is None:
            if score_type == "progress":
                # Maximize progress = d0 - d  <=>  minimize -(d0 - d).
                # With score_mode="min", d is already min_t d_t, so this is
                # max-over-trajectory progress: d0 - min_t d_t.
                scores = -(d0 - d)
            else:
                scores = d

        if it == 0:
            first_iter_spread = float(scores.std())
        i = int(np.argmin(scores))
        if scores[i] < best_score:
            best_score = float(scores[i])
            best_seq = seqs[i].copy()

        elite_idx = np.argsort(scores)[:n_elite]
        elite_means.append(float(scores[elite_idx].mean()))
        counts = np.zeros((horizon, n_actions), dtype=np.float64)
        for idx in elite_idx:
            for t in range(horizon):
                counts[t, seqs[idx, t]] += 1.0
        logits = np.log((counts + 1.0) / (n_elite + n_actions))

    action = int(best_seq[0])
    if return_info:
        return action, {
            "best_score": best_score,
            "best_seq": [int(a) for a in best_seq],
            "elite_mean": elite_means,
            "first_iter_score_std": first_iter_spread,
            "score_mode": score_mode,
            "score_type": score_type,
            "metric": metric,
            "horizon": horizon,
        }
    return action
