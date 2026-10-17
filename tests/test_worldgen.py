"""Tests for tiny-world (worldgen): procedural maps, diffusion prior,
dreamed environments."""

import numpy as np

from tiny_jepa.worldgen import (
    GRID, STYLES, procedural_map, reachable, WorldPrior, GeneratedWorld,
    dream_world,
)
from tiny_jepa.worldgen.maps import largest_empty_component, spawn_points

SEEDS = [0, 1, 7, 42]


def test_procedural_maps_are_sane():
    for style in range(3):
        for seed in SEEDS:
            m = procedural_map(style, seed=seed)
            assert m.shape == (GRID, GRID)
            assert set(np.unique(m)) <= {0, 1}
            # walled border
            assert m[0, :].all() and m[-1, :].all()
            assert m[:, 0].all() and m[:, -1].all()
            # empty space mostly connected
            assert largest_empty_component(m) >= 0.5, (style, seed)
            # spawn points can reach each other
            a, b = spawn_points(m, seed=seed)
            assert reachable(m, a, b), (style, seed)


def test_maze_style_varies():
    maps = {procedural_map(1, seed=s).tobytes() for s in range(10)}
    assert len(maps) > 5  # not the same maze every time


def _tiny_prior():
    maps = np.stack([procedural_map(s % 3, seed=1000 + s)
                     for s in range(48)])
    styles = np.array([s % 3 for s in range(48)])
    return WorldPrior(seed=0).fit(maps, styles, steps=40, batch=16,
                                  seed=0, log_every=0)


def test_diffusion_trains_and_samples():
    prior = _tiny_prior()
    out = prior.sample(style=1, seed=0, n=4)
    assert out.shape == (4, GRID, GRID)
    assert set(np.unique(out)) <= {0, 1}
    # borders enforced even if the net forgets
    assert out[:, 0, :].all() and out[:, :, 0].all()


def test_prior_save_load_roundtrip(tmp_path):
    prior = _tiny_prior()
    p = str(tmp_path / "prior.npz")
    prior.save(p)
    p2 = WorldPrior.load(p)
    a = prior.sample(style=0, seed=5)
    b = p2.sample(style=0, seed=5)
    assert (a == b).all()


def test_generated_world_physics():
    w = GeneratedWorld(procedural_map(0, seed=3), seed=0)
    f = w.render()
    assert f.shape == (32, 32) and f.dtype == np.float32
    assert 0 <= f.min() and f.max() <= 1
    before = w.agent.copy()
    for _ in range(200):  # wander; walls must never be entered
        w.step(int(np.random.default_rng(0).integers(5)))
        ty, tx = int(w.agent[1] // 2), int(w.agent[0] // 2)
        assert w.map[ty, tx] == 0, "agent inside a wall!"
    assert not np.allclose(before, w.agent)  # it actually moved


def test_pellet_respawns_reachable():
    w = GeneratedWorld(procedural_map(1, seed=11), seed=1)
    for _ in range(10):
        w._respawn_pellet()
        at = (int(w.agent[1] // 2), int(w.agent[0] // 2))
        pt = (int(w.pellet[1] // 2), int(w.pellet[0] // 2))
        assert w.map[pt] == 0 and reachable(w.map, at, pt)


def test_dream_world_procedural_fallback():
    for style in range(3):
        w = dream_world(prior=None, style=style, seed=style)
        assert isinstance(w, GeneratedWorld)
        w.render()


def test_dream_world_with_prior():
    prior = _tiny_prior()
    w = dream_world(prior=prior, style=2, seed=0)
    assert isinstance(w, GeneratedWorld)
    f, r = w.step(4)
    assert f.shape == (32, 32)


def test_detect_pellet_survives_walls():
    # the learned head collapses on wall textures (2.7px -> 47px);
    # physics-grounded detection must not.
    from tiny_jepa.worldgen import detect_pellet
    prior = _tiny_prior()
    errs = []
    for seed in range(3):
        w = dream_world(prior=prior, style=1, seed=seed)
        for _ in range(5):
            f = w.render()
            p = detect_pellet(f, w.agent)
            errs.append(float(np.linalg.norm(p - w.pellet)))
            w.step(0)
    assert np.mean(errs) < 4.0, np.mean(errs)
