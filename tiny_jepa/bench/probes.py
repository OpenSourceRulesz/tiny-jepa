"""tiny-spatial v1 probes.

Each probe is a controlled experiment against a TinyJEPA checkpoint and
returns a ProbeResult with score in [0, 1] (higher = better). See SPEC.md.
"""

import numpy as np

from .._orby import OrbyWorld, CANVAS
from .envs import OcclusionWorld, TwoRoomWorld

CENTER = np.array([CANVAS / 2, CANVAS / 2], dtype=np.float64)

PASS = {
    "permanence": 0.5,
    "hallucination": 0.7,
    "surprise": 0.7,
    "spatial_memory": 0.3,
    "memory_update": 0.3,
}


class ProbeResult:
    def __init__(self, name, score, details=None):
        self.name = name
        self.score = float(np.clip(score, 0.0, 1.0))
        self.passed = self.score >= PASS[name]
        self.details = details or {}

    def __repr__(self):
        mark = "PASS" if self.passed else "FAIL"
        return f"{self.name}: {self.score:.3f} [{mark}]"


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------- P1
def permanence(model, n_episodes=20, ep_len=64):
    """P1 — Object permanence (behavioral): can the agent still hunt when
    the pellet is intermittently occluded?

    Natural episodes in OcclusionWorld vs plain OrbyWorld, greedy planner.
    When the pellet goes behind the wall, perception fails; the right
    behavior is to fall back on last-seen position (the pellet is static,
    and eating through the wall is allowed). Score = occluded eat-rate /
    clean eat-rate. End-to-end, public API only, no probe training.
    """
    def eat_rate(env_cls, seed_base):
        total = 0
        for e in range(n_episodes):
            env = env_cls(seed=seed_base + e)
            for _ in range(ep_len):
                f = env.render()
                _, rew = env.step(model.plan(f, env.agent))
                total += rew > 0.5
        return total / n_episodes

    occ = eat_rate(OcclusionWorld, 10000)
    clean = eat_rate(OrbyWorld, 11000)
    return ProbeResult("permanence", occ / (clean + 1e-6),
                       {"occ_per_ep": round(occ, 2),
                        "clean_per_ep": round(clean, 2)})


# ---------------------------------------------------------------- P2
def hallucination(model, n_episodes=120, ep_len=64):
    """P2 — Hallucination: when the pellet is fully hidden (natural
    OcclusionWorld episodes), where does the perception head point?

    EpicWorldModel's finding: deterministic world models don't get vague
    under occlusion, they hallucinate memorized positions with false
    confidence. center_bias = fraction of hidden-step predictions within
    6px of the canvas center (the dataset prior). score = 1 - center_bias.
    Caveat: the wall itself is mildly out-of-distribution for the head,
    so this documents the shipped system's behavior, whatever the cause.
    """
    hid = []
    for e in range(n_episodes):
        env = OcclusionWorld(seed=12000 + e)
        rng = np.random.default_rng(12100 + e)
        for _ in range(ep_len):
            env.step(int(rng.integers(5)))
            if not env.pellet_visible():
                hid.append(model.perceive_pellet(env.render()))
    hid = np.array(hid)
    rate = float(np.mean(np.linalg.norm(hid - CENTER, axis=1) < 6.0)) \
        if len(hid) else 0.0
    return ProbeResult("hallucination", 1.0 - rate,
                       {"hidden_steps": len(hid),
                        "center_bias": round(rate, 3)})


# ---------------------------------------------------------------- P3
def _auc(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    n_pos = len(pos)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * len(neg)))


def surprise(model, n_episodes=40, ep_len=64):
    """P3 — Surprise: 1-step prediction error should spike on pellet
    respawns (teleports). AUC over steps, respawn vs normal.
    """
    pos, neg = [], []
    for e in range(n_episodes):
        env = OrbyWorld(seed=3000 + e)
        rng = np.random.default_rng(4000 + e)
        z = model.encode(env.render())
        for _ in range(ep_len):
            a = int(rng.integers(5))
            _, rew = env.step(a)
            z2 = model.encode(env.render())
            pred = model.imagine(z, [a])[0]
            s = 1.0 - _cos(pred, z2)
            (pos if rew > 0.5 else neg).append(s)
            z = z2
    auc = _auc(pos, neg) if pos and neg else 0.5
    return ProbeResult("surprise", auc,
                       {"n_respawns": len(pos), "auc": round(auc, 3)})


# ---------------------------------------------------------------- P4
def spatial_memory(model, n_episodes=20, max_steps=128):
    """P4 — Spatial memory: pellet in room 0, agent starts in room 1
    (fog of war), away from the corridor column. Greedy planner;
    success = pellet eaten. Reaching room 0 requires finding the
    corridor first — no free lunch from the start position.
    """
    wins, coverages = 0, 0.0
    for e in range(n_episodes):
        env = TwoRoomWorld(seed=5000 + e)
        env.agent = np.array([5.0, 24.0])  # room 1, away from corridor
        env._respawn_pellet(room=0)
        visited = set()
        ate = False
        for _ in range(max_steps):
            f = env.render()
            a = model.plan(f, env.agent)
            _, rew = env.step(a)
            visited.add((int(env.agent[0]) // 4, int(env.agent[1]) // 4))
            if rew > 0.5:
                ate = True
                break
        wins += ate
        room0 = {(cx, cy) for cx in range(8) for cy in range(4)}
        coverages += len(visited & room0) / len(room0)
    return ProbeResult("spatial_memory", wins / n_episodes,
                       {"success_rate": round(wins / n_episodes, 3),
                        "room0_coverage": round(coverages / n_episodes, 3)})


# ---------------------------------------------------------------- P5
def memory_update(model, n_episodes=20, phase_steps=5, max_steps=96):
    """P5 — Memory update: agent sees pellet at p1, leaves; pellet moves
    to p2; agent returns. Does it find p2 or perseverate at stale p1?
    """
    updated = 0
    for e in range(n_episodes):
        env = TwoRoomWorld(seed=6000 + e)
        p1 = np.array([8.0, 8.0])
        p2 = np.array([24.0, 8.0])
        env.agent = np.array([8.0, 12.0])
        env.pellet = p1.copy()
        for _ in range(phase_steps):
            f = env.render()
            env.step(model.plan(f, env.agent))
        env.agent = np.array([5.0, 24.0])   # room 1, away from corridor
        env.pellet = p2.copy()
        for _ in range(phase_steps):
            f = env.render()
            env.step(model.plan(f, env.agent))
        env.agent = np.array([5.0, 20.0])   # room 1 edge: go find it
        ate = False
        for _ in range(max_steps):
            f = env.render()
            _, rew = env.step(model.plan(f, env.agent))
            if rew > 0.5:
                ate = True
                break
        updated += ate
    return ProbeResult("memory_update", updated / n_episodes,
                       {"update_rate": round(updated / n_episodes, 3)})


PROBES = [permanence, hallucination, surprise, spatial_memory, memory_update]
