"""ArmWorld: a 2-link planar robot arm reaching for targets, as pixels.

Follows the OrbyWorld interface so the tiny-jepa training + planning stack
works unchanged:
  - __init__(seed=None), reset(seed=None) -> frame
  - step(action) -> (frame, reward); actions 0=stay, 1=j1-, 2=j1+, 3=j2-, 4=j2+
  - render() -> (32,32) float32; .agent (2,) = gripper-tip xy; .pellet (2,) =
    target xy; .n_actions = 5

Why this world exists: the user's real robot arms are LeRobot-compatible,
and the observe -> imagine -> plan -> act loop needs a sim arm first. The
agent here is the *tip* of the arm, so the same perception heads and
greedy-style controllers transfer directly.

Kinematics (screen coords, y down): base at (16, 27), link lengths 10 and 9.
theta1 in [-2.35, -0.79] rad (-135..-45 deg, i.e. pointing up into the canvas),
theta2 in [-1.4, +1.4] relative. Joint step 0.12 rad.
"""

import numpy as np

CANVAS = 32
N_ACTIONS = 5
BASE = (16.0, 27.0)
L1, L2 = 10.0, 9.0
DTH = 0.12
T1_MIN, T1_MAX = -2.35, -0.79
T2_MIN, T2_MAX = -1.4, 1.4
TIP_R = 2.0
LINK_R = 1.3
TARGET_R = 2.8
EAT_DIST = 3.5
MAX_REACH = L1 + L2 - 2.0  # keep targets comfortably inside reach

_yy, _xx = np.linspace(0, CANVAS - 1, CANVAS), np.linspace(0, CANVAS - 1, CANVAS)
_GX, _GY = np.meshgrid(_xx, _yy)


def _blob(x, y, r, bright):
    d2 = (_GX - x) ** 2 + (_GY - y) ** 2
    return bright * np.exp(-d2 / (2.0 * r * r))


def _capsule(p0, p1, r, bright):
    """Anti-aliased capsule (segment) rasterization on the 32x32 grid."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    L2d = float(np.dot(d, d)) + 1e-9
    gx = _GX - p0[0]
    gy = _GY - p0[1]
    t = np.clip((gx * d[0] + gy * d[1]) / L2d, 0.0, 1.0)
    cx = p0[0] + t * d[0]
    cy = p0[1] + t * d[1]
    d2 = (gx - (cx - p0[0])) ** 2 + (gy - (cy - p0[1])) ** 2
    return bright * np.exp(-d2 / (2.0 * r * r))


def fk(t1, t2):
    """Forward kinematics -> (elbow_xy, tip_xy)."""
    elbow = (BASE[0] + L1 * np.cos(t1), BASE[1] + L1 * np.sin(t1))
    a = t1 + t2
    tip = (elbow[0] + L2 * np.cos(a), elbow[1] + L2 * np.sin(a))
    return np.array(elbow), np.array(tip)


class ArmWorld:
    n_actions = N_ACTIONS

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.t1 = float(r.uniform(T1_MIN, T1_MAX))
        self.t2 = float(r.uniform(T2_MIN, T2_MAX))
        self._update_pose()
        self._respawn_target()
        self.last_shaped = 0.0
        return self.render()

    def _update_pose(self):
        self.elbow, self.tip = fk(self.t1, self.t2)
        self.agent = self.tip.copy()  # the "agent" is the gripper tip

    def _respawn_target(self):
        r = self.rng
        for _ in range(60):
            ang = r.uniform(-np.pi, 0.0)  # upper half-plane from base
            rad = r.uniform(6.0, MAX_REACH)
            p = np.array([BASE[0] + rad * np.cos(ang),
                          BASE[1] + rad * np.sin(ang)])
            if (3.0 <= p[0] <= CANVAS - 3.0 and 3.0 <= p[1] <= CANVAS - 3.0
                    and np.linalg.norm(p - self.tip) > 6.0):
                break
        self.pellet = p.astype(np.float64)  # the "pellet" is the target

    def step(self, action):
        a = int(action)
        t1, t2 = self.t1, self.t2
        if a == 1:
            t1 -= DTH
        elif a == 2:
            t1 += DTH
        elif a == 3:
            t2 -= DTH
        elif a == 4:
            t2 += DTH
        self.t1 = min(max(t1, T1_MIN), T1_MAX)
        self.t2 = min(max(t2, T2_MIN), T2_MAX)
        d_before = float(np.linalg.norm(self.tip - self.pellet))
        self._update_pose()
        reward = 0.0
        if np.linalg.norm(self.tip - self.pellet) < EAT_DIST:
            reward = 1.0
            self._respawn_target()
        d_after = float(np.linalg.norm(self.tip - self.pellet))
        self.last_shaped = float(reward) - 0.02 * d_after
        return self.render(), reward

    def render(self):
        img = _blob(*BASE, 3.0, 0.9)                    # base
        img = img + _capsule(BASE, self.elbow, LINK_R, 0.85)
        img = img + _capsule(self.elbow, self.tip, LINK_R, 0.85)
        img = img + _blob(*self.tip, TIP_R, 1.0)        # gripper tip
        img = img + _blob(*self.pellet, TARGET_R, 1.0)  # target
        return np.clip(img, 0, 1).astype(np.float32)


def collect_arm_episodes(n_episodes, ep_len, seed=0, greedy_prob=0.65):
    """Joint-aware data collection: greedy = the joint action minimizing
    resulting tip-to-target distance (via FK), else random."""
    frames, actions, shaped, pellets, tips = [], [], [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        w = ArmWorld(seed=int(rng.integers(1e9)))
        ep_f = [w.render()]
        ep_p = [w.pellet.copy()]
        ep_g = [w.tip.copy()]
        ep_a, ep_s = [], []
        for _ in range(ep_len):
            if rng.random() < greedy_prob:
                best, best_d = 0, float(np.linalg.norm(w.tip - w.pellet))
                for cand in (1, 2, 3, 4):
                    t1, t2 = w.t1, w.t2
                    if cand == 1:
                        t1 -= DTH
                    elif cand == 2:
                        t1 += DTH
                    elif cand == 3:
                        t2 -= DTH
                    else:
                        t2 += DTH
                    t1 = min(max(t1, T1_MIN), T1_MAX)
                    t2 = min(max(t2, T2_MIN), T2_MAX)
                    _, tip = fk(t1, t2)
                    d = float(np.linalg.norm(tip - w.pellet))
                    if d < best_d - 1e-9:
                        best, best_d = cand, d
                a = best
            else:
                a = int(rng.integers(N_ACTIONS))
            fr, _ = w.step(a)
            ep_f.append(fr)
            ep_a.append(a)
            ep_s.append(w.last_shaped)
            ep_p.append(w.pellet.copy())
            ep_g.append(w.tip.copy())
        frames.append(np.stack(ep_f))
        actions.append(np.array(ep_a))
        shaped.append(np.array(ep_s, dtype=np.float32))
        pellets.append(np.stack(ep_p))
        tips.append(np.stack(ep_g))
    return {
        "frames": np.stack(frames).astype(np.float32),
        "actions": np.stack(actions),
        "shaped": np.stack(shaped).astype(np.float32),
        "pellet": np.stack(pellets).astype(np.float32),
        "agent": np.stack(tips).astype(np.float32),
    }


def arm_greedy_action(model, frame, env):
    """Controller for the arm: learned vision finds the target, proprioception
    gives the true joint state; pick the joint action minimizing predicted
    tip-to-target distance via FK. Mirrors TinyJEPA.plan() semantics."""
    target = model.perceive_pellet(frame)
    best, best_d = 0, float("inf")
    for a in range(N_ACTIONS):
        t1, t2 = env.t1, env.t2
        if a == 1:
            t1 -= DTH
        elif a == 2:
            t1 += DTH
        elif a == 3:
            t2 -= DTH
        elif a == 4:
            t2 += DTH
        t1 = min(max(t1, T1_MIN), T1_MAX)
        t2 = min(max(t2, T2_MIN), T2_MAX)
        _, tip = fk(t1, t2)
        d = float(np.linalg.norm(tip - target))
        if d < best_d:
            best, best_d = a, d
    return best
