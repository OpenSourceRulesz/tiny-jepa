"""Smoke tests for tiny-jepa. Run: pytest tests/  (or python -m pytest)"""

import os

import numpy as np
import pytest

from tiny_jepa import TinyJEPA, OrbyWorld, to_patches, from_patches


@pytest.fixture(scope="module")
def model():
    ckpt = os.path.join(os.path.dirname(__file__), "..", "tiny_jepa",
                        "checkpoints", "orby_v1.npz")
    return TinyJEPA.load(ckpt)


@pytest.fixture(scope="module")
def frame():
    return OrbyWorld(seed=0).render()


def test_encode_shape(model, frame):
    z = model.encode(frame)
    assert z.shape == (32,)
    assert np.all(np.isfinite(z))


def test_encode_batch(model):
    frames = np.stack([OrbyWorld(seed=i).render() for i in range(4)])
    z = model.encode_batch(frames)
    assert z.shape == (4, 32)


def test_imagine_decode_shapes(model, frame):
    z = model.encode(frame)
    future = model.imagine(z, [4, 4, 1])
    assert future.shape == (3, 32)
    dreams = model.dream(frame, [4, 4])
    assert len(dreams) == 2 and dreams[0].shape == (32, 32)


def test_perceive_pellet_reasonable(model):
    env = OrbyWorld(seed=3)
    pred = model.perceive_pellet(env.render())
    assert np.linalg.norm(pred - env.pellet) < 8.0  # pixels


def test_plan_valid_action(model, frame):
    a = model.plan(frame, agent_xy=np.array([16.0, 16.0]))
    assert 0 <= a < 5


def test_save_load_roundtrip(model, tmp_path):
    p = str(tmp_path / "m.npz")
    model.save(p)
    m2 = TinyJEPA.load(p)
    f = OrbyWorld(seed=1).render()
    assert np.abs(model.encode(f) - m2.encode(f)).max() == 0.0


def test_patches_roundtrip():
    f = OrbyWorld(seed=2).render()
    assert np.abs(from_patches(to_patches(f)) - f).max() < 1e-6


def test_onnx_export(model, tmp_path):
    onnx = pytest.importorskip("onnx")
    from tiny_jepa.export_onnx import export
    paths = export(model, str(tmp_path))
    assert set(paths) == {"enc", "pred", "reward", "dec", "pellet"}
    for p in paths.values():
        onnx.load(p)  # validates the file


def test_train_smoke():
    """A few training steps on random data shouldn't crash or NaN."""
    from tiny_jepa.train import train_on_data
    rng = np.random.default_rng(0)
    N, T = 4, 16
    frames = rng.random((N, T + 1, 32, 32), dtype=np.float32)
    actions = rng.integers(0, 5, size=(N, T))
    shaped = rng.random((N, T), dtype=np.float32) * 0.1
    model = train_on_data(frames, actions, shaped, steps=3, batch=32,
                          log_every=0)
    z = model.encode(frames[0, 0])
    assert np.all(np.isfinite(z))
