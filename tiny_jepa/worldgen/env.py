"""tiny-world: play inside dreamed worlds.

GeneratedWorld is an OrbyWorld-shaped environment (render/step/agent/
pellet, 32x32 frames) whose walls come from a 16x16 tile map — dreamed
by the diffusion prior or drawn procedurally. Wall collision is
tile-based; pellet spawns are BFS-guaranteed reachable, so dreamed
worlds are always playable.

Because rendering is deterministic, every view of a dreamed world is
consistent by construction — the "World Labs for your laptop" cheat.
"""

import numpy as np

from .._orby import (
    CANVAS, AGENT_R, AGENT_BRIGHT, BALL_R, BALL_BRIGHT, BALL_SPEED,
    PELLET_R, PELLET_BRIGHT, EAT_DIST, N_ACTIONS, _blob,
)
from .maps import GRID, STYLES, procedural_map, reachable, spawn_points, \
    largest_empty_component

WALL_FILL = 0.3
TILE = CANVAS // GRID  # 2 px per tile

__all__ = ["GeneratedWorld", "dream_world", "detect_pellet", "WALL_FILL"]


def detect_pellet(frame, agent_xy, r_exclude=4.0):
    """Physics-grounded pellet perception for walled worlds.

    The pellet is the only 1.0-brightness blob besides the agent (walls
    are 0.3, distractor balls <= 0.55). Mask out the agent — whose
    position is known via proprioception — and take the brightest
    remaining pixel. Deterministic, no learning, immune to the domain
    shift that breaks the learned head on wall textures.
    """
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float64)
    keep = (xx - float(agent_xy[0])) ** 2 + \
        (yy - float(agent_xy[1])) ** 2 > r_exclude ** 2
    masked = np.where(keep, np.asarray(frame, dtype=np.float64), 0.0)
    y, x = np.unravel_index(int(masked.argmax()), masked.shape)
    return np.array([float(x), float(y)])


def _tile_of(xy):
    return (int(np.clip(xy[1] // TILE, 0, GRID - 1)),
            int(np.clip(xy[0] // TILE, 0, GRID - 1)))


def _tile_center(t):
    return np.array([(t[1] + 0.5) * TILE, (t[0] + 0.5) * TILE])


class GeneratedWorld:
    """Playable world from a 16x16 tile map (1 = wall, 0 = empty)."""

    def __init__(self, map, seed=None, agent_tile=None, pellet_tile=None):
        self.map = np.asarray(map, dtype=np.int64)
        assert self.map.shape == (GRID, GRID)
        self.rng = np.random.default_rng(seed)
        self._wall_mask = np.repeat(np.repeat(self.map, TILE, axis=0),
                                    TILE, axis=1).astype(bool)
        if agent_tile is None or pellet_tile is None:
            agent_tile, pellet_tile = spawn_points(
                self.map, seed=int(self.rng.integers(1e9)))
        self.agent = _tile_center(agent_tile).astype(np.float64)
        self.pellet = _tile_center(pellet_tile).astype(np.float64)
        # two distractor balls on empty tiles
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

    # -- rendering ------------------------------------------------------
    def render(self):
        img = np.zeros((CANVAS, CANVAS), dtype=np.float32)
        img[self._wall_mask] = WALL_FILL
        img = img + _blob(*self.agent, AGENT_R, AGENT_BRIGHT)
        for b in range(2):
            img = img + _blob(*self.balls[b], BALL_R[b], BALL_BRIGHT[b])
        img = img + _blob(*self.pellet, PELLET_R, PELLET_BRIGHT)
        return np.clip(img, 0, 1).astype(np.float32)

    # -- dynamics -------------------------------------------------------
    def _blocked(self, xy):
        return self.map[_tile_of(xy)] == 1

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
        self.agent[0] = min(max(self.agent[0] + dx, 2.5), CANVAS - 2.5)
        self.agent[1] = min(max(self.agent[1] + dy, 2.5), CANVAS - 2.5)
        if self._blocked(self.agent):  # bonk: walls win
            self.agent = old
        # balls drift, bouncing off walls axis-by-axis
        for b in range(2):
            for d in (0, 1):
                nb = self.balls[b].copy()
                nb[d] += self.bvel[b, d]
                nb[d] = min(max(nb[d], 2.0), CANVAS - 2.0)
                if self._blocked(nb):
                    self.bvel[b, d] *= -1
                else:
                    self.balls[b, d] = nb[d]
        reward = 0.0
        if np.linalg.norm(self.agent - self.pellet) < EAT_DIST:
            reward = 1.0
            self.eaten += 1
            self._respawn_pellet()
        return self.render(), reward

    def _respawn_pellet(self):
        at = _tile_of(self.agent)
        for _ in range(60):
            t = (int(self.rng.integers(1, GRID - 1)),
                 int(self.rng.integers(1, GRID - 1)))
            if self.map[t] == 0 and reachable(self.map, at, t):
                da = abs(t[0] - at[0]) + abs(t[1] - at[1])
                if da >= 3:
                    break
        self.pellet = _tile_center(t).astype(np.float64)


def dream_world(prior=None, style=None, seed=None, max_tries=20):
    """Dream up a brand-new playable world.

    prior: a trained WorldPrior (diffusion). style: 0 rooms / 1 maze /
    2 open (or None for random). Without a prior, draws procedurally.
    Invalid samples are rejected (BFS); falls back to procedural.
    Returns a GeneratedWorld.
    """
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
    if m is None:  # prior missing or all samples degenerate
        m = procedural_map(style=style, seed=int(rng.integers(1e9)))
    return GeneratedWorld(m, seed=int(rng.integers(1e9)))
