"""
OrbyWorld: a tiny 2D world for action-conditioned JEPA training.

- 32x32 canvas, one agent + two bouncing distractor balls + one pellet.
- Actions: 0=stay, 1=up, 2=down, 3=left, 4=right. Agent moves 2px per step.
- Reward 1 when the agent eats the pellet (then it respawns).
- Rendering: sum of gaussian bumps, clipped to [0,1], float32.

The JS port for the playable demo MUST reproduce this exactly (see SPEC.md
test vectors). Keep this file dependency-free (numpy only).
"""

import numpy as np

CANVAS = 32
AGENT_R = 2.0
AGENT_BRIGHT = 1.0
AGENT_SPEED = 2.0
BALL_R = (2.5, 1.8)
BALL_BRIGHT = (0.55, 0.40)
BALL_SPEED = 2.2
PELLET_R = 2.8
PELLET_BRIGHT = 1.0
EAT_DIST = 4.5
N_ACTIONS = 5

_yy, _xx = np.linspace(0, CANVAS - 1, CANVAS), np.linspace(0, CANVAS - 1, CANVAS)
_GX, _GY = np.meshgrid(_xx, _yy)


def _blob(x, y, r, bright):
    d2 = (_GX - x) ** 2 + (_GY - y) ** 2
    return bright * np.exp(-d2 / (2.0 * r * r))


class OrbyWorld:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        r = self.rng
        self.agent = r.uniform(6, CANVAS - 6, size=2).astype(np.float64)
        self.balls = r.uniform(5, CANVAS - 5, size=(2, 2)).astype(np.float64)
        ang = r.uniform(0, 2 * np.pi, size=2)
        spd = r.uniform(0.6, 1.0, size=2) * BALL_SPEED
        self.bvel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * spd[:, None]
        self._respawn_pellet()
        return self.render()

    def _respawn_pellet(self):
        r = self.rng
        for _ in range(20):
            p = r.uniform(4, CANVAS - 4, size=2)
            if np.linalg.norm(p - self.agent) > 6.0:
                break
        self.pellet = p.astype(np.float64)

    def step(self, action):
        a = int(action)
        dx, dy = 0.0, 0.0
        if a == 1: dy = -AGENT_SPEED
        elif a == 2: dy = AGENT_SPEED
        elif a == 3: dx = -AGENT_SPEED
        elif a == 4: dx = AGENT_SPEED
        self.agent[0] = min(max(self.agent[0] + dx, 2.5), CANVAS - 2.5)
        self.agent[1] = min(max(self.agent[1] + dy, 2.5), CANVAS - 2.5)
        # distractor balls drift + bounce
        self.balls = self.balls + self.bvel
        for b in range(2):
            for d in (0, 1):
                if self.balls[b, d] < BALL_R[b] or self.balls[b, d] > CANVAS - BALL_R[b]:
                    self.bvel[b, d] *= -1
                    self.balls[b, d] = min(max(self.balls[b, d], BALL_R[b]), CANVAS - BALL_R[b])
        reward = 0.0
        if np.linalg.norm(self.agent - self.pellet) < EAT_DIST:
            reward = 1.0
            self._respawn_pellet()
        return self.render(), reward

    def render(self):
        img = _blob(*self.agent, AGENT_R, AGENT_BRIGHT)
        for b in range(2):
            img = img + _blob(*self.balls[b], BALL_R[b], BALL_BRIGHT[b])
        img = img + _blob(*self.pellet, PELLET_R, PELLET_BRIGHT)
        return np.clip(img, 0, 1).astype(np.float32)

    def state_vector(self):
        """For test vectors: rounded snapshot of full simulator state."""
        return [round(float(v), 4) for v in
                list(self.agent) + list(self.balls.flatten()) +
                list(self.bvel.flatten()) + list(self.pellet)]


def greedy_action(world):
    """Heuristic pellet-seeker used for data collection (not the demo autopilot)."""
    d = world.pellet - world.agent
    if abs(d[0]) > abs(d[1]):
        return 4 if d[0] > 0 else 3
    else:
        return 2 if d[1] > 0 else 1


def collect_episodes(n_episodes, ep_len, seed=0, greedy_prob=0.65):
    """Returns dict of arrays: frames (N,T+1,32,32), actions (N,T),
    rewards (N,T) binary pellet-eaten, shaped (N,T) dense reward for planning:
    1.0 on eat, else clipped 0.2 * (dist_before - dist_after).
    pellet (N,T+1,2): pellet (x,y) at each frame, for auxiliary supervision.
    agentxy (N,T+1,2): agent (x,y) at each frame, for auxiliary supervision."""
    frames, actions, rewards, shaped, pellets, agents = [], [], [], [], [], []
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        w = OrbyWorld(seed=int(rng.integers(1e9)))
        ep_f = [w.render()]
        ep_p = [w.pellet.copy()]
        ep_g = [w.agent.copy()]
        ep_a, ep_r, ep_s = [], [], []
        for _ in range(ep_len):
            if rng.random() < greedy_prob:
                a = greedy_action(w)
            else:
                a = int(rng.integers(N_ACTIONS))
            d_before = float(np.linalg.norm(w.agent - w.pellet))
            fr, rew = w.step(a)
            if rew > 0.5:
                sh = 1.0
            else:
                d_after = float(np.linalg.norm(w.agent - w.pellet))
                sh = float(np.clip(0.2 * (d_before - d_after), -0.2, 0.2))
            ep_f.append(fr); ep_a.append(a); ep_r.append(rew); ep_s.append(sh)
            ep_p.append(w.pellet.copy()); ep_g.append(w.agent.copy())
        frames.append(np.stack(ep_f)); actions.append(ep_a)
        rewards.append(ep_r); shaped.append(ep_s); pellets.append(np.stack(ep_p))
        agents.append(np.stack(ep_g))
    return {
        "frames": np.stack(frames).astype(np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "shaped": np.array(shaped, dtype=np.float32),
        "pellet": np.stack(pellets).astype(np.float32),
        "agentxy": np.stack(agents).astype(np.float32),
    }


if __name__ == "__main__":
    # Test vector for the JS port (SPEC.md): fixed seed, fixed actions.
    w = OrbyWorld(seed=42)
    acts = [4, 4, 2, 2, 3, 0, 1, 4]
    print("init_state:", w.state_vector())
    for i, a in enumerate(acts):
        fr, rew = w.step(a)
        print(f"step{i} a={a} reward={rew} state={w.state_vector()} frame_sum={round(float(fr.sum()),4)}")
