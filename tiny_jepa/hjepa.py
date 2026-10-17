"""Hierarchical JEPA: two timescales, one planner.

Level 0 (fast, every step): the frozen TinyJEPA — encode frame -> z_t,
predictor P0(z_t, a_t) -> z_{t+1}. Already trained (orby_v1).

Level 1 (slow, every K steps): a command-conditioned forward model
F(s_j, g_j) -> xy_{j+1}, where s_j = mean(z_{jK..jK+K}) is the pooled
fast latent (a frozen JEPA representation) and g_j is a SUBGOAL command
in agent-xy space ("be here in K steps"). The target is the agent's
spatial state 8 steps out — an abstract, slow-moving quantity — rather
than pixels. This is the hierarchical move: the slow level reasons
about *where to be*, the fast level about *how to move*.

Two hard-won design lessons are baked in:
1. The slow model MUST train on command-conditioned data (each segment
   pursues a random waypoint g). Training on greedy/random futures
   teaches on-policy dynamics that collapse under counterfactual
   planning queries ("what if I aim for g?").
2. Predicting slow LATENTS fails: pooled latents change so slowly that
   the net takes the identity shortcut P(s,g) ~= s (or ~= 0 for deltas),
   ignoring the command. Predicting the spatial outcome instead gives
   g nowhere to hide — xy_next is dominated by the command.

Planning is top-down: the slow level picks the subgoal whose predicted
outcome scores best — near the pellet when it's visible, novel territory
when it isn't — and the fast level greedily steers to it for K steps,
with event-driven replans (pellet eaten/teleported, trust flips). The
learned wall dynamics are what beat flat greedy in rooms: "aim at the
pellet through a wall" correctly predicts getting stuck, so the
corridor wins instead.

The class is a drop-in for TinyJEPA (delegates everything but plan()),
so tiny-spatial runs it unchanged::

    from tiny_jepa.hjepa import HierarchicalJEPA, train_slow_level
    fast = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
    hier = train_slow_level(fast, env_fns=[OrbyWorld, TwoRoomWorld])
    hier.save("tiny_jepa/checkpoints/orby_hier_v1_slow.npz")
    hier.plan(frame, agent_xy)   # hierarchical action
"""

import time

import numpy as np

from .model import TinyJEPA, ACTION_DELTAS, CANVAS
from .nn import MLP, Adam

K_DEFAULT = 8


# ---------------------------------------------------------------- training
def _collect_segments(fast, env_fns, n_ep, ep_len, K, seed):
    """Roll out episodes (mixed envs) with a WAYPOINT-SEEKING policy and
    encode frames with the frozen fast encoder.

    The slow predictor is queried counterfactually at planning time
    ("what if I aim for g?"), so it must train on command-conditioned
    data: each K-step segment pursues a random waypoint g, giving
    (s_j, g_j, s_{j+1}) where g_j is the actual command. Greedy/random
    data would teach it only on-policy futures, which collapses under
    counterfactual queries.

    Returns dict of segment arrays: s (M,32), g (M,2) in [0,1],
    s_next (M,32), agent (M,2) in [0,1].
    """
    from .model import ACTION_DELTAS
    rng = np.random.default_rng(seed)
    S, G, XY = [], [], []
    for i in range(n_ep):
        env_fn = env_fns[i % len(env_fns)]
        env = env_fn(seed=int(rng.integers(1e9)))
        frames, agents = [env.render()], [env.agent.copy()]
        seg_cmds = []  # command g for each step
        g = None
        for t in range(ep_len):
            if t % K == 0 or g is None:
                g = rng.uniform(4, CANVAS - 4, size=2)  # random waypoint
            # greedy 1-step toward g (known kinematics)
            best, best_d = 0, float("inf")
            for ai, (dx, dy) in enumerate(ACTION_DELTAS):
                nx = min(max(env.agent[0] + float(dx), 2.5), CANVAS - 2.5)
                ny = min(max(env.agent[1] + float(dy), 2.5), CANVAS - 2.5)
                dd = (nx - g[0]) ** 2 + (ny - g[1]) ** 2
                if dd < best_d:
                    best_d, best = dd, ai
            env.step(best)
            seg_cmds.append(g.copy())
            frames.append(env.render())
            agents.append(env.agent.copy())
        z = fast.encode_batch(np.stack(frames))          # (T+1, 32)
        agents = np.stack(agents).astype(np.float32) / CANVAS
        n_seg = (ep_len + 1) // K - 1
        for j in range(n_seg):
            S.append(z[j * K:(j + 1) * K].mean(axis=0))
            G.append(seg_cmds[j * K] / CANVAS)
            XY.append(agents[(j + 1) * K])  # where the command got us
    return {"s": np.stack(S).astype(np.float32),
            "g": np.stack(G).astype(np.float32),
            "xy_next": np.stack(XY).astype(np.float32)}


def train_slow_level(fast, env_fns, K=K_DEFAULT, steps=800, batch=256,
                     ep_len=64, seed=0, log_every=100):
    """Train the slow forward model F(s, g) -> xy_next on frozen fast
    latents. Returns a HierarchicalJEPA. Pure numpy, CPU minutes.
    """
    hier = HierarchicalJEPA(fast, K=K, seed=seed)
    rng = np.random.default_rng(seed)
    opt = Adam(hier.slow_fwd.params_and_grads(), lr=3e-4)
    t0 = time.time()
    for step in range(1, steps + 1):
        d = _collect_segments(fast, env_fns, 12, ep_len, K,
                              int(rng.integers(1e9)))
        M = len(d["s"])
        idx = rng.integers(M, size=batch)
        s, g, xy = d["s"][idx], d["g"][idx], d["xy_next"][idx]
        pin = np.concatenate([s, g], axis=1)
        yhat = hier.slow_fwd.forward(pin)
        err = yhat - xy
        loss = float(np.mean(err ** 2))
        hier.slow_fwd.backward((2.0 * err / err.size).astype(np.float32))
        tot = float(sum(float((gv ** 2).sum())
                        for _, gv in opt.params)) ** 0.5
        if tot > 5.0:
            sc = 5.0 / (tot + 1e-8)
            for _, gv in opt.params:
                gv *= sc
        opt.step()
        if log_every and step % log_every == 0:
            spread = float(yhat.std())
            print(f"slow step {step:5d}/{steps} mse={loss:.4f} "
                  f"pred_std={spread:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
    return hier


# ------------------------------------------------------- hierarchical agent
class HierarchicalJEPA:
    """Wraps a trained TinyJEPA; overrides plan() with two-timescale
    planning. Everything else delegates to the fast model, so benchmarks
    and tools accept it anywhere a TinyJEPA goes.
    """

    WAYPOINTS = {  # named subgoals in xy pixels
        "corridor": (16.0, 16.0),
        "room0": (16.0, 8.0),
        "room1": (16.0, 24.0),
    }

    def __init__(self, fast, K=K_DEFAULT, seed=0):
        object.__setattr__(self, "fast", fast)
        self.K = K
        # (pooled slow latent 32 + command 2) -> agent xy in 8 steps ([0,1])
        self.slow_fwd = MLP([fast.latent_dim + 2, 64, 2], seed=seed + 1)
        self.reset_episode()

    # -- delegation (plan/save/load are ours) ---------------------------
    def __getattr__(self, name):
        if name in ("fast", "slow_fwd"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "fast"), name)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    # -- planner state ---------------------------------------------------
    def reset_episode(self):
        self._zhist = []       # recent fast latents (for slow segment)
        self._subgoal = None   # current xy subgoal
        self._ticks_left = 0
        self._visited = {}     # (cx, cy) cell -> count, for novelty
        self._t = 0            # planner steps this episode
        self._trusted_prev = False
        self._last_pellet = None   # last trusted pellet position (memory)
        self._last_seen_t = -10**9

    @staticmethod
    def _pellet_visible(frame, pellet_xy):
        """Is the perceived pellet actually in the frame? The pellet
        renders at peak brightness 1.0 (distractor balls peak at 0.55),
        so a bright blob near the perceived position means visible.
        Grounds trust in pixels, not in the head's (stable) garbage.
        """
        x = int(round(float(pellet_xy[0])))
        y = int(round(float(pellet_xy[1])))
        patch = frame[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        return patch.size > 0 and float(patch.max()) > 0.7

    # -- slow tick --------------------------------------------------------
    def _slow_replan(self, agent_xy, trusted, pellet):
        s = np.mean(self._zhist[-self.K:], axis=0).astype(np.float32)
        cands = []  # (name, xy, score_fn)
        if trusted:
            cands.append(("pellet", pellet, "goal"))
        else:
            # memory: go check where we last saw it (invalidated on arrival)
            if (self._last_pellet is not None
                    and self._t - self._last_seen_t < 64):
                if float(np.linalg.norm(agent_xy - self._last_pellet)) < 6.0:
                    self._last_pellet = None  # checked: not there. forget.
                else:
                    cands.append(("last_seen", self._last_pellet, "goal"))
            for name, xy in self.WAYPOINTS.items():
                cands.append((name, np.asarray(xy, dtype=np.float64),
                              "novel"))
        best, best_score = None, -1e18
        for name, g, kind in cands:
            pin = np.concatenate(
                [s, (g / CANVAS).astype(np.float32)])[None]
            p_pred = self.slow_fwd.forward(pin)[0] * CANVAS
            if kind == "goal":
                score = -float(np.linalg.norm(p_pred - g))
            else:
                cell = (int(p_pred[0]) // 4, int(p_pred[1]) // 4)
                novelty = 1.0 / (1.0 + self._visited.get(cell, 0))
                score = novelty - 0.05 * float(
                    np.linalg.norm(p_pred - agent_xy))
            if score > best_score:
                best_score, best = score, (name, g)
        self._subgoal = np.asarray(best[1], dtype=np.float64)
        self._ticks_left = self.K

    # -- fast tick ----------------------------------------------------------
    def plan(self, frame, agent_xy, pellet_xy=None):
        """Hierarchical action: slow subgoal every K steps, greedy 1-step
        steering toward it otherwise. Replans early on pellet events
        (eaten/respawned: perceived position jumps) and trust flips.
        Pass pellet_xy to override learned vision (e.g. physics-grounded
        detection in dreamed worlds); otherwise the pellet head is used.
        """
        z = self.fast.encode(frame)
        self._zhist.append(z)
        if pellet_xy is None:
            pellet_xy = self.fast.perceive_pellet(frame)
        pellet = np.asarray(pellet_xy, dtype=np.float64)
        trusted = self._pellet_visible(frame, pellet)
        agent_xy = np.asarray(agent_xy, dtype=np.float64)
        cell = (int(agent_xy[0]) // 4, int(agent_xy[1]) // 4)
        self._visited[cell] = self._visited.get(cell, 0) + 1
        self._t += 1

        replan = self._ticks_left <= 0 or self._subgoal is None
        if trusted != self._trusted_prev:
            replan = True  # pellet appeared / disappeared
        if (trusted and self._trusted_prev and self._last_pellet is not None
                and float(np.linalg.norm(pellet - self._last_pellet)) > 10.0):
            replan = True  # pellet eaten or teleported: stale subgoal
        self._trusted_prev = trusted
        if trusted:
            self._last_pellet = pellet
            self._last_seen_t = self._t
        if replan:
            self._slow_replan(agent_xy, trusted, pellet)
        # greedy 1-step toward subgoal (known kinematics)
        best, best_d = 0, float("inf")
        for a, (dx, dy) in enumerate(ACTION_DELTAS):
            nx = min(max(agent_xy[0] + float(dx), 2.5), CANVAS - 2.5)
            ny = min(max(agent_xy[1] + float(dy), 2.5), CANVAS - 2.5)
            d = (nx - self._subgoal[0]) ** 2 + (ny - self._subgoal[1]) ** 2
            if d < best_d:
                best_d, best = d, a
        self._ticks_left -= 1
        return int(best)

    # -- persistence (slow weights only; fast loads separately) -----------
    def save(self, path):
        arrays = {f"slow_fwd.{k}": v
                  for k, v in self.slow_fwd.parameters().items()}
        arrays["K"] = np.array([self.K])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, fast_checkpoint, slow_checkpoint, K=None):
        fast = TinyJEPA.load(fast_checkpoint)
        d = np.load(slow_checkpoint)
        k = int(d["K"][0]) if K is None else K
        self = cls(fast, K=k, seed=0)
        self.slow_fwd.load_parameters(
            {k2.split(".", 1)[1]: d[k2]
             for k2 in d.files if k2.startswith("slow_fwd.")})
        return self
