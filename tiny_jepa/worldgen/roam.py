"""tiny-inspatio: roamable dreamed worlds with anchored novel-view prediction.

Architecture inspired by InSpatio-WorldFM (arXiv 2603.11911, Apache-2.0)
and INSPATIO-WORLD (arXiv 2604.07209): explicit geometric anchors +
persistent world memory + frame-independent novel-view generation,
miniaturized to 2D tile worlds. Original tiny implementation, not a port.

The translation:
- their 6-DoF camera roaming      -> 2D viewport roaming over a 64px world
- their implicit spatiotemporal cache -> WorldMemory: persistent
  observed-tile map, written as the agent explores
- their explicit spatial constraint (warped reference + validity mask)
  -> anchored_view(): reproject known tiles into a novel camera frame,
  with a mask marking hallucinated regions
- their diffusion DiT             -> ViewCompletion: tiny generative
  completion of unseen tiles conditioned on the anchored partial view
- their frame-based paradigm      -> each novel view predicted
  independently from memory (no video windows)

The 32x32 viewport is drop-in for the existing 32-dim encoder/planner:
the world got bigger, the agent's eyes stayed the same.
"""

import time

import numpy as np

from .._orby import (
    AGENT_R, AGENT_BRIGHT, BALL_R, BALL_BRIGHT, BALL_SPEED,
    PELLET_R, PELLET_BRIGHT, EAT_DIST,
)
from ..nn import MLP, Adam
from .maps import GRID, STYLES, N_STYLES, procedural_map, reachable, \
    spawn_points, largest_empty_component

# big-world geometry: same 16x16 tile maps, bigger pixels
BTILE = 4
BCANVAS = GRID * BTILE          # 64
VIEW = 32                       # viewport: 32x32 camera
VTILES = VIEW // BTILE          # 8x8 tiles per view

_byy, _bxx = np.linspace(0, BCANVAS - 1, BCANVAS), \
    np.linspace(0, BCANVAS - 1, BCANVAS)
_BGX, _BGY = np.meshgrid(_bxx, _byy)

WALL_FILL = 0.3
PRED_FILL = 0.15  # predicted (hallucinated) walls render lighter


def _blob64(x, y, r, bright):
    d2 = (_BGX - x) ** 2 + (_BGY - y) ** 2
    return bright * np.exp(-d2 / (2.0 * r * r))


def _tile_of_xy(xy):
    return (int(np.clip(xy[1] // BTILE, 0, GRID - 1)),
            int(np.clip(xy[0] // BTILE, 0, GRID - 1)))


def _tile_center(t):
    return np.array([(t[1] + 0.5) * BTILE, (t[0] + 0.5) * BTILE])


class BigWorld:
    """64x64 playable world from a 16x16 tile map, with a 32x32 roaming
    viewport camera. Pellet spawns are BFS-reachable; walls block."""

    def __init__(self, map, seed=None, agent_tile=None, pellet_tile=None):
        self.map = np.asarray(map, dtype=np.int64)
        assert self.map.shape == (GRID, GRID)
        self.rng = np.random.default_rng(seed)
        self._wall_mask = np.repeat(np.repeat(self.map, BTILE, axis=0),
                                    BTILE, axis=1).astype(bool)
        if agent_tile is None or pellet_tile is None:
            agent_tile, pellet_tile = spawn_points(
                self.map, seed=int(self.rng.integers(1e9)))
        self.agent = _tile_center(agent_tile).astype(np.float64)
        self.pellet = _tile_center(pellet_tile).astype(np.float64)
        self.balls = np.zeros((2, 2))
        for b in range(2):
            t = (int(self.rng.integers(1, GRID - 1)),
                 int(self.rng.integers(1, GRID - 1)))
            for _ in range(50):
                if self.map[t] == 0:
                    break
                t = (int(self.rng.integers(1, GRID - 1)),
                     int(self.rng.integers(1, GRID - 1)))
            self.balls[b] = _tile_center(t)
        ang = self.rng.uniform(0, 2 * np.pi, size=2)
        spd = self.rng.uniform(0.6, 1.0, size=2) * BALL_SPEED
        self.bvel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * spd[:, None]
        self.eaten = 0

    # -- camera ---------------------------------------------------------
    def cam_origin(self):
        """Top-left of the 32x32 viewport following the agent."""
        x0 = int(np.clip(self.agent[0] - VIEW / 2, 0, BCANVAS - VIEW))
        y0 = int(np.clip(self.agent[1] - VIEW / 2, 0, BCANVAS - VIEW))
        return x0, y0

    def render_full(self):
        img = np.zeros((BCANVAS, BCANVAS), dtype=np.float32)
        img[self._wall_mask] = WALL_FILL
        img = img + _blob64(*self.agent, AGENT_R * 1.6, AGENT_BRIGHT)
        for b in range(2):
            img = img + _blob64(*self.balls[b], BALL_R[b] * 1.6,
                                BALL_BRIGHT[b])
        img = img + _blob64(*self.pellet, PELLET_R * 1.6, PELLET_BRIGHT)
        return np.clip(img, 0, 1).astype(np.float32)

    def viewport(self):
        """(32x32 frame, (x0, y0), agent_xy_in_viewport)."""
        x0, y0 = self.cam_origin()
        frame = self.render_full()[y0:y0 + VIEW, x0:x0 + VIEW]
        return (frame.astype(np.float32), (x0, y0),
                self.agent - np.array([x0, y0]))

    # -- dynamics -------------------------------------------------------
    def _blocked(self, xy):
        return self.map[_tile_of_xy(xy)] == 1

    def step(self, action):
        a = int(action)
        dx, dy = 0.0, 0.0
        if a == 1:
            dy = -2.0
        elif a == 2:
            dy = 2.0
        elif a == 3:
            dx = -2.0
        elif a == 4:
            dx = 2.0
        old = self.agent.copy()
        self.agent[0] = min(max(self.agent[0] + dx, 2.5), BCANVAS - 2.5)
        self.agent[1] = min(max(self.agent[1] + dy, 2.5), BCANVAS - 2.5)
        if self._blocked(self.agent):
            self.agent = old
        for b in range(2):
            for d in (0, 1):
                nb = self.balls[b].copy()
                nb[d] += self.bvel[b, d]
                nb[d] = min(max(nb[d], 2.0), BCANVAS - 2.0)
                if self._blocked(nb):
                    self.bvel[b, d] *= -1
                else:
                    self.balls[b, d] = nb[d]
        reward = 0.0
        if np.linalg.norm(self.agent - self.pellet) < EAT_DIST:
            reward = 1.0
            self.eaten += 1
            self._respawn_pellet()
        return self.viewport()[0], reward

    def _respawn_pellet(self):
        at = _tile_of_xy(self.agent)
        t = None
        for _ in range(60):
            cand = (int(self.rng.integers(1, GRID - 1)),
                    int(self.rng.integers(1, GRID - 1)))
            if self.map[cand] == 0 and reachable(self.map, at, cand):
                if abs(cand[0] - at[0]) + abs(cand[1] - at[1]) >= 3:
                    t = cand
                    break
        if t is None:
            t = at
        self.pellet = _tile_center(t).astype(np.float64)


def dream_big_world(prior=None, style=None, seed=None, max_tries=20):
    """Dream a 64px roamable world. Same validation contract as
    dream_world, but the map renders at 4px/tile."""
    rng = np.random.default_rng(seed)
    if style is None:
        style = int(rng.integers(len(STYLES)))
    m = None
    if prior is not None:
        for _ in range(max_tries):
            cand = prior.sample(style=style,
                                seed=int(rng.integers(1e9)))[0]
            if largest_empty_component(cand) >= 0.5:
                m = cand
                break
    if m is None:
        m = procedural_map(style=style, seed=int(rng.integers(1e9)))
    return BigWorld(m, seed=int(rng.integers(1e9)))


class WorldMemory:
    """Persistent world representation: the tiny implicit spatiotemporal
    cache. Records tiles the agent has actually seen; anchored_view()
    reprojects known geometry into any novel camera frame with a
    validity mask (their explicit spatial constraint, tiny)."""

    def __init__(self):
        self.seen = np.zeros((GRID, GRID), dtype=bool)
        self.tiles = np.full((GRID, GRID), -1, dtype=np.int64)

    def observe(self, map, x0, y0):
        """Record tiles visible in viewport with top-left (x0, y0)."""
        for i in range(VTILES):
            for j in range(VTILES):
                ty, tx = y0 // BTILE + i, x0 // BTILE + j
                if 0 <= ty < GRID and 0 <= tx < GRID:
                    self.seen[ty, tx] = True
                    self.tiles[ty, tx] = map[ty, tx]

    def coverage(self):
        return float(self.seen.mean())

    def anchored_view(self, x0, y0):
        """(partial, mask) for the 8x8-tile viewport at (x0, y0):
        partial has +/-1 for known tiles, 0 for unseen; mask marks known.
        This is the explicit geometric anchor for novel-view prediction."""
        partial = np.zeros((VTILES, VTILES), dtype=np.float32)
        mask = np.zeros((VTILES, VTILES), dtype=np.float32)
        for i in range(VTILES):
            for j in range(VTILES):
                ty, tx = y0 // BTILE + i, x0 // BTILE + j
                if 0 <= ty < GRID and 0 <= tx < GRID \
                        and self.seen[ty, tx]:
                    partial[i, j] = 1.0 if self.tiles[ty, tx] == 1 else -1.0
                    mask[i, j] = 1.0
        return partial, mask

    def render_belief(self, x0, y0, completed=None):
        """32x32 viewport of what the model believes: known walls solid,
        model-completed (hallucinated) walls lighter. No entities --
        this is the geometry belief, v1 predicts static structure."""
        img = np.zeros((VIEW, VIEW), dtype=np.float32)
        for i in range(VTILES):
            for j in range(VTILES):
                ty, tx = y0 // BTILE + i, x0 // BTILE + j
                v = 0.0
                if 0 <= ty < GRID and 0 <= tx < GRID:
                    if self.seen[ty, tx]:
                        v = WALL_FILL if self.tiles[ty, tx] == 1 else 0.0
                    elif completed is not None and completed[i, j] == 1:
                        v = PRED_FILL
                img[i * BTILE:(i + 1) * BTILE,
                    j * BTILE:(j + 1) * BTILE] = v
        return img


class ViewCompletion:
    """Generative completion of unseen tiles, conditioned on the anchored
    partial view + mask (their diffusion DiT, tiny: a direct MLP
    predictor for v1; the interface fits a diffusion later)."""

    def __init__(self, n_styles=N_STYLES, seed=0):
        self.n_styles = n_styles
        # (partial 64 + mask 64 + style 3) -> completed 64 (tanh-ish ±1)
        self.net = MLP([2 * VTILES * VTILES + n_styles, 128, 128,
                        VTILES * VTILES], seed=seed)

    def train_step(self, partial, mask, styles, target, opt):
        B = len(partial)
        onehot = np.zeros((B, self.n_styles), dtype=np.float32)
        onehot[np.arange(B), styles] = 3.0
        inp = np.concatenate([partial.reshape(B, -1), mask.reshape(B, -1),
                              onehot], axis=1).astype(np.float32)
        out = self.net.forward(inp)
        err = out - target.reshape(B, -1).astype(np.float32)
        loss = float(np.mean(err ** 2))
        self.net.backward((2.0 * err / err.size).astype(np.float32))
        opt.step()
        return loss

    def fit(self, partials, masks, styles, targets, steps=2000, batch=128,
            seed=0, log_every=500):
        rng = np.random.default_rng(seed)
        opt = Adam(self.net.params_and_grads(), lr=1e-3)
        N = len(partials)
        t0 = time.time()
        for step in range(1, steps + 1):
            idx = rng.integers(N, size=batch)
            loss = self.train_step(partials[idx], masks[idx], styles[idx],
                                   targets[idx], opt)
            if log_every and step % log_every == 0:
                print(f"completion step {step:5d}/{steps} loss={loss:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        return self

    def predict(self, partial, mask, style=0):
        """(8,8) +/-1 completed tile map for one viewport."""
        onehot = np.zeros((1, self.n_styles), dtype=np.float32)
        onehot[0, int(style) % self.n_styles] = 3.0
        inp = np.concatenate([partial.reshape(1, -1), mask.reshape(1, -1),
                              onehot], axis=1).astype(np.float32)
        out = self.net.forward(inp)[0]
        return (out.reshape(VTILES, VTILES) > 0).astype(np.int64)

    def save(self, path):
        arrays = {f"net.{k}": v
                  for k, v in self.net.parameters().items()}
        arrays["n_styles"] = np.array([self.n_styles])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, seed=0):
        d = np.load(path)
        self = cls(n_styles=int(d["n_styles"][0]), seed=seed)
        self.net.load_parameters(
            {k.split(".", 1)[1]: d[k] for k in d.files
             if k.startswith("net.")})
        return self


__all__ = ["BigWorld", "dream_big_world", "WorldMemory", "ViewCompletion",
           "BTILE", "BCANVAS", "VIEW", "VTILES", "WALL_FILL", "PRED_FILL"]
