"""Tests for the hierarchical JEPA level."""

import sys

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/tinyjepa")

from tiny_jepa import TinyJEPA
from tiny_jepa.hjepa import HierarchicalJEPA
from tiny_jepa._orby import OrbyWorld
from tiny_jepa.bench.envs import TwoRoomWorld

FAST = "/home/hatch/workspace/tinyjepa/tiny_jepa/checkpoints/orby_v1.npz"
SLOW = "/home/hatch/workspace/tinyjepa/tiny_jepa/checkpoints/orby_hier_v1_slow.npz"


def _hier():
    return HierarchicalJEPA.load(FAST, SLOW)


def test_delegation():
    h = _hier()
    f = OrbyWorld(seed=0).render()
    assert h.encode(f).shape == (32,)
    assert h.imagine(h.encode(f), [1]).shape == (1, 32)
    assert h.perceive_pellet(f).shape == (2,)
    assert h.decode(h.encode(f)).shape == (32, 32)


def test_save_load_roundtrip(tmp_path):
    h = _hier()
    p = str(tmp_path / "slow.npz")
    h.save(p)
    h2 = HierarchicalJEPA.load(FAST, p)
    assert h2.K == h.K
    assert np.allclose(h.slow_fwd.layers[0].W, h2.slow_fwd.layers[0].W)


def test_reset_episode_clears_state():
    h = _hier()
    env = OrbyWorld(seed=0)
    for _ in range(10):
        h.plan(env.render(), env.agent)
        env.step(0)
    assert len(h._zhist) == 10 and h._visited
    h.reset_episode()
    assert h._zhist == [] and h._visited == {} and h._subgoal is None


def test_slow_forward_model_respects_commands():
    """Counterfactual core: commanding different subgoals must predict
    different 8-step outcomes. (Regression test: two earlier slow-level
    designs — direct latent prediction, then delta prediction — both
    learned to ignore the command via shortcuts.)
    """
    h = _hier()
    spreads = []
    for seed in (3, 11, 23):
        env = OrbyWorld(seed=seed)
        z = h.encode(env.render())
        s = np.repeat(z[None], 8, axis=0).mean(axis=0).astype(np.float32)
        outs = []
        for g in (np.array([4.0, 4.0]), np.array([28.0, 28.0]),
                  np.array([4.0, 28.0])):
            pin = np.concatenate([s, (g / 32.0).astype(np.float32)])[None]
            outs.append(h.slow_fwd.forward(pin)[0] * 32.0)
        spreads.append(max(float(np.linalg.norm(a - b))
                           for a in outs for b in outs))
    assert min(spreads) > 8.0, f"slow model ignores commands: {spreads}"


def test_hierarchical_plan_runs_in_two_rooms():
    h = _hier()
    env = TwoRoomWorld(seed=0)
    h.reset_episode()
    for _ in range(32):
        a = h.plan(env.render(), env.agent)
        assert 0 <= a <= 4
        env.step(a)
    # planner committed to real subgoals, not NaN
    assert h._subgoal is not None and np.all(np.isfinite(h._subgoal))
