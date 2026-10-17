"""Two NEW synthetic pixel-worlds for tiny-jepa generalization stress-tests.

Both follow the OrbyWorld interface so train_on_env / planners work unchanged:
  - __init__(seed=None), reset(seed=None) -> frame
  - step(action) -> (frame, reward);  actions 0=stay,1=up,2=down,3=left,4=right
  - render() -> (32,32) float32;  .agent (2,), .pellet (2,) attributes
    (.pellet = nearest pellet, for the greedy data collector + aux head)

World A -- MazeWorld: two wall bars force detours around obstacles.
World B -- MultiPelletWorld: 3 pellets at once, reward per pellet eaten.
"""

import numpy as np

CANVAS = 32
AGENT_R = 2.0
AGENT_SPEED = 2.0
PELLET_R = 2.8
EAT_DIST = 4.5
N_ACTIONS = 5
WALL_BRIGHT = 0.65

_yy, _xx = np.linspace(0, CANVAS - 1, CANVAS), np.linspace(0, CANVAS - 1, CANVAS)
_GX, _GY = np.meshgrid(_xx, _yy)


def _blob(x, y, r, bright):
    d2 = (_GX - x) ** 2 + (_GY - y) ** 2
    return bright * np.exp(-d2 / (2.0 * r * r))


class MazeWorld:
    """32x32 arena with two wall bars. Agent/pellet spawn in free space.

    Walls (x0, y0, x1, y1), 2px thick:
      - horizontal bar y in [13,15], x in [2,22]  -> gap on the right
      - vertical bar   x in [10,12], y in [18,30] -> gap above
    Free space stays connected, so a perfect planner can always reach the
    pellet; a greedy 1-step controller will sometimes get stuck at walls.
    Moving into a wall = no movement (bump).
    """

    WALLS = [(2, 13, 22, 15), (10, 18, 12, 30)]

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._wall_mask = np.zeros((CANVAS, CANVAS), dtype=bool)
        for (x0, y0, x1, y1) in self.WALLS:
            self._wall_mask[y0:y1, x0:x1] = True
        self.reset()

    def _free(self, x, y, margin=3.0):
        for (x0, y0, x1, y1) in self.WALLS:
            if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
                return False
        return 2.5 <= x <= CANVAS - 2.5 and 2.5 <= y <= CANVAS - 2.5

    def _blocked(self, x, y):
        for (x0, y0, x1, y1) in self.WALLS:
            if (x0 - AGENT_R <= x <= x1 + AGENT_R
                    and y0 - AGENT_R <= y <= y1 + AGENT_R):
                return True
        return False

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        for _ in range(60):
            a = r.uniform(3, CANVAS - 3, size=2)
            if self._free(*a):
                break
        self.agent = a.astype(np.float64)
        self._respawn_pellet()
        return self.render()

    def _respawn_pellet(self):
        r = self.rng
        for _ in range(60):
            p = r.uniform(4, CANVAS - 4, size=2)
            if self._free(*p) and np.linalg.norm(p - self.agent) > 6.0:
                break
        self.pellet = p.astype(np.float64)

    def step(self, action):
        a = int(action)
        dx, dy = 0.0, 0.0
        if a == 1:
            dy = -AGENT_SPEED
        elif a == 2:
            dy = AGENT_SPEED
        elif a == 3:
            dx = -AGENT_SPEED
        elif a == 4:
            dx = AGENT_SPEED
        nx = min(max(self.agent[0] + dx, 2.5), CANVAS - 2.5)
        ny = min(max(self.agent[1] + dy, 2.5), CANVAS - 2.5)
        if not self._blocked(nx, ny):
            self.agent[0], self.agent[1] = nx, ny
        reward = 0.0
        if np.linalg.norm(self.agent - self.pellet) < EAT_DIST:
            reward = 1.0
            self._respawn_pellet()
        return self.render(), reward

    def render(self):
        img = self._wall_mask.astype(np.float32) * WALL_BRIGHT
        img = img + _blob(*self.agent, AGENT_R, 1.0)
        img = img + _blob(*self.pellet, PELLET_R, 1.0)
        return np.clip(img, 0, 1).astype(np.float32)


class MultiPelletWorld:
    """Open arena, no walls, no distractors -- but 3 pellets at once.

    .pellet always tracks the NEAREST pellet so the greedy data collector
    and the aux pellet head keep working. Reward = pellets eaten this step
    (pellets respawn to keep 3 on the field). Tests whether the reward head
    and planner handle competing targets.
    """

    N_PELLETS = 3

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent = self.rng.uniform(6, CANVAS - 6, size=2).astype(np.float64)
        self.pellets = np.stack([self._spawn_one(i) for i in range(self.N_PELLETS)])
        self._update_nearest()
        return self.render()

    def _spawn_one(self, idx):
        r = self.rng
        others = getattr(self, "pellets", None)
        for _ in range(40):
            p = r.uniform(4, CANVAS - 4, size=2)
            if np.linalg.norm(p - self.agent) < 6.0:
                continue
            if others is not None and len(others) > 0:
                keep = [j for j in range(len(others)) if j != idx]
                if keep and np.linalg.norm(others[keep] - p, axis=1).min() < 8.0:
                    continue
            break
        return p.astype(np.float64)

    def _update_nearest(self):
        d = np.linalg.norm(self.pellets - self.agent, axis=1)
        self.pellet = self.pellets[int(np.argmin(d))].copy()

    def step(self, action):
        a = int(action)
        dx, dy = 0.0, 0.0
        if a == 1:
            dy = -AGENT_SPEED
        elif a == 2:
            dy = AGENT_SPEED
        elif a == 3:
            dx = -AGENT_SPEED
        elif a == 4:
            dx = AGENT_SPEED
        self.agent[0] = min(max(self.agent[0] + dx, 2.5), CANVAS - 2.5)
        self.agent[1] = min(max(self.agent[1] + dy, 2.5), CANVAS - 2.5)
        reward = 0.0
        for i in range(self.N_PELLETS):
            if np.linalg.norm(self.agent - self.pellets[i]) < EAT_DIST:
                reward += 1.0
                self.pellets[i] = self._spawn_one(i)
        self._update_nearest()
        return self.render(), reward

    def render(self):
        img = _blob(*self.agent, AGENT_R, 1.0)
        for p in self.pellets:
            img = img + _blob(*p, PELLET_R, 1.0)
        return np.clip(img, 0, 1).astype(np.float32)
