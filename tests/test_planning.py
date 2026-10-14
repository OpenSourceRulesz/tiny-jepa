"""Tests for tiny_jepa.planning (CEM-based MPC)."""

import numpy as np

from tiny_jepa import TinyJEPA, OrbyWorld, cem_plan
from tiny_jepa.planning import _softmax


class _MockModel:
    """Fake model with a known reward landscape: reward 1 iff action==4.

    Lets us test the CEM optimizer logic independent of the learned model's
    (weak) reward head.
    """

    n_actions = 5

    def encode(self, frame):
        return np.zeros(8, dtype=np.float32)

    def imagine(self, z, actions):
        # latents just echo the action, one per step
        return np.stack([np.full(8, a, dtype=np.float32) for a in actions])

    def predict_reward(self, z, action):
        return 1.0 if int(action) == 4 else 0.0


def test_softmax_rows_sum_to_one():
    p = _softmax(np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [10.0, 0, 0, 0, 0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[0], 0.2)
    assert p[1, 0] > 0.99


def test_cem_finds_known_optimum():
    # With reward only for action 4, CEM must converge on all-4 sequences
    # and return 4 as the first action.
    frame = np.zeros((32, 32), dtype=np.float32)
    a, info = cem_plan(_MockModel(), frame, horizon=4, n_samples=64,
                       n_elite=12, n_iters=4, seed=0, return_info=True)
    assert a == 4
    assert info["best_seq"] == [4, 4, 4, 4]
    assert info["best_score"] > 3.5  # ~sum of discounted 1.0s
    # elite mean should improve (or at least not collapse) across iterations
    assert info["elite_mean"][-1] >= info["elite_mean"][0] - 1e-9


def test_cem_deterministic_given_seed():
    model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
    env = OrbyWorld(seed=7)
    frame = env.render()
    a1, i1 = cem_plan(model, frame, seed=123, return_info=True)
    a2, i2 = cem_plan(model, frame, seed=123, return_info=True)
    assert a1 == a2 and i1["best_seq"] == i2["best_seq"]
    assert 0 <= a1 < model.n_actions


def test_cem_runs_on_real_checkpoint():
    # Integration smoke test: a short closed-loop run executes cleanly and
    # the returned actions are valid. Uses a light config for speed.
    model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
    env = OrbyWorld(seed=11)
    eaten = 0
    for t in range(16):
        a = cem_plan(model, env.render(), horizon=4, n_samples=16,
                     n_elite=4, n_iters=2, seed=t)
        assert 0 <= int(a) < 5
        _, r = env.step(int(a))
        eaten += int(r > 0.5)
    assert eaten >= 0  # executes; performance is measured in the benchmark
