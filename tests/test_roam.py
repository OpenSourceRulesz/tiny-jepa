"""Tests for tiny-inspatio (worldgen.roam): big worlds, viewport camera,
world memory, anchored novel-view completion."""

import numpy as np

from tiny_jepa.worldgen import procedural_map
from tiny_jepa.worldgen.roam import (
    BigWorld, dream_big_world, WorldMemory, ViewCompletion,
    BTILE, BCANVAS, VIEW, VTILES, GRID,
)


def test_big_world_viewport():
    w = BigWorld(procedural_map(0, seed=0), seed=0)
    frame, (x0, y0), av = w.viewport()
    assert frame.shape == (VIEW, VIEW) and frame.dtype == np.float32
    # agent centered-ish in its own viewport
    assert abs(av[0] - VIEW / 2) <= 4 and abs(av[1] - VIEW / 2) <= 4
    # full render is 64x64
    assert w.render_full().shape == (BCANVAS, BCANVAS)


def test_big_world_wall_collision():
    w = BigWorld(procedural_map(1, seed=1), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        w.step(int(rng.integers(5)))
        ty, tx = int(w.agent[1] // BTILE), int(w.agent[0] // BTILE)
        assert w.map[ty, tx] == 0, "agent inside a wall!"


def test_memory_observe_and_anchor():
    m = procedural_map(0, seed=2)
    mem = WorldMemory()
    assert mem.coverage() == 0.0
    mem.observe(m, 0, 0)
    assert mem.coverage() > 0.0
    partial, mask = mem.anchored_view(0, 0)
    assert partial.shape == (VTILES, VTILES)
    assert mask.all()  # just observed: everything known
    # anchor fidelity: known tiles match ground truth
    truth = np.where(m[0:VTILES, 0:VTILES] == 1, 1.0, -1.0)
    assert np.allclose(partial, truth)
    # novel camera: partially unknown (16,16 overlaps tiles 4..7)
    p2, m2 = mem.anchored_view(16, 16)
    assert 0 < m2.mean() < 1.0


def _tiny_completion():
    rng = np.random.default_rng(0)
    m = procedural_map(2, seed=5)
    P, M, S, T = [], [], [], []
    for _ in range(200):
        mem = WorldMemory()
        mem.observe(m, int(rng.choice([0, 8, 16, 24, 32])),
                    int(rng.choice([0, 8, 16, 24, 32])))
        x0, y0 = int(rng.choice([0, 8, 16, 24, 32])), \
            int(rng.choice([0, 8, 16, 24, 32]))
        p, mk = mem.anchored_view(x0, y0)
        t = np.where(m[y0 // BTILE:y0 // BTILE + VTILES,
                       x0 // BTILE:x0 // BTILE + VTILES] == 1, 1.0, -1.0)
        P.append(p); M.append(mk); S.append(2); T.append(t.astype(np.float32))
    vc = ViewCompletion(seed=0).fit(np.stack(P), np.stack(M),
                                    np.array(S), np.stack(T),
                                    steps=150, batch=32, seed=0,
                                    log_every=0)
    return vc


def test_completion_trains_and_respects_anchors():
    vc = _tiny_completion()
    m = procedural_map(2, seed=5)
    mem = WorldMemory()
    mem.observe(m, 0, 0)
    p, mk = mem.anchored_view(0, 0)  # fully known view
    out = vc.predict(p, mk, style=2)
    assert out.shape == (VTILES, VTILES)
    # on a fully-known view the completion must reproduce the anchors
    truth = (m[0:VTILES, 0:VTILES] == 1).astype(np.int64)
    assert (out == truth).mean() > 0.9


def test_completion_save_load_roundtrip(tmp_path):
    vc = _tiny_completion()
    pth = str(tmp_path / "vc.npz")
    vc.save(pth)
    vc2 = ViewCompletion.load(pth)
    m = procedural_map(2, seed=5)
    mem = WorldMemory()
    mem.observe(m, 8, 8)
    p, mk = mem.anchored_view(8, 8)
    assert (vc.predict(p, mk) == vc2.predict(p, mk)).all()


def test_dream_big_world():
    w = dream_big_world(prior=None, style=1, seed=0)
    assert isinstance(w, BigWorld)
    f, r = w.step(4)
    assert f.shape == (VIEW, VIEW)
