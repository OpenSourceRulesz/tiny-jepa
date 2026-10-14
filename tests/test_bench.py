"""Tests for tiny-spatial: envs behave, probes are deterministic."""

import sys

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/tinyjepa")

from tiny_jepa import TinyJEPA
from tiny_jepa.bench.envs import OcclusionWorld, TwoRoomWorld
from tiny_jepa.bench import run_benchmark
from tiny_jepa.bench.probes import (
    permanence, hallucination, surprise, spatial_memory, memory_update,
)

CKPT = "/home/hatch/workspace/tinyjepa/tiny_jepa/checkpoints/orby_v1.npz"


def test_occlusion_hides_pellet():
    w = OcclusionWorld(seed=0)
    w.pellet = np.array([16.0, 16.0])
    assert not w.pellet_visible()
    f = w.render()
    assert abs(float(f[14:20, 15].mean()) - 0.3) < 1e-6
    w.pellet = np.array([6.0, 6.0])
    assert w.pellet_visible()


def test_two_room_fog_of_war():
    t = TwoRoomWorld(seed=1)
    t.agent = np.array([16.0, 24.0])   # room 1
    t.pellet = np.array([16.0, 8.0])   # room 0
    t.balls = np.array([[16.0, 8.0], [16.0, 24.0]])
    f = t.render()
    assert float(f[22:26, 14:18].max()) > 0.9   # own room visible
    assert float(f[6:10, 14:18].max()) < 0.01   # other room hidden


def test_two_room_wall_blocks_but_corridor_passes():
    t = TwoRoomWorld(seed=1)
    t.agent = np.array([5.0, 12.0])
    t.step(2)  # down into wall
    assert float(t.agent[1]) == 12.0  # blocked
    t.agent = np.array([16.0, 12.0])
    for _ in range(4):
        t.step(2)
    assert TwoRoomWorld.room_of(t.agent) == 1  # through corridor


def test_probes_deterministic():
    model = TinyJEPA.load(CKPT)
    r1 = [p(model).score for p in
          (permanence, hallucination, surprise, spatial_memory, memory_update)]
    r2 = [p(model).score for p in
          (permanence, hallucination, surprise, spatial_memory, memory_update)]
    assert r1 == r2, (r1, r2)


def test_benchmark_report_card():
    rep = run_benchmark(CKPT, verbose=False)
    assert len(rep.results) == 5
    assert all(0.0 <= r.score <= 1.0 for r in rep.results)
    assert 0.0 <= rep.spatial_iq <= 1.0
    d = rep.to_dict()
    assert set(d["probes"]) == {"permanence", "hallucination", "surprise",
                                "spatial_memory", "memory_update"}
