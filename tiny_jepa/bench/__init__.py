"""tiny-spatial: a benchmark for spatial intelligence in tiny world models.

Object permanence, occlusion reasoning, spatial memory — the probes every
tiny world model should pass. See SPEC.md for the design.
"""

import json

from ..model import TinyJEPA
from . import envs  # noqa: F401
from .probes import PROBES, ProbeResult  # noqa: F401

__all__ = ["run_benchmark", "BenchmarkReport", "ProbeResult", "envs"]


class BenchmarkReport:
    def __init__(self, checkpoint, results):
        self.checkpoint = str(checkpoint)
        self.results = results
        self.spatial_iq = float(sum(r.score for r in results) / len(results))

    def print(self):
        print(f"tiny-spatial v1  [{self.checkpoint}]")
        print("-" * 44)
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  {r.name:15s} {r.score:.3f}  [{mark}]  {r.details}")
        print("-" * 44)
        print(f"  spatial_iq: {self.spatial_iq:.3f}")

    def to_dict(self):
        return {
            "checkpoint": self.checkpoint,
            "spatial_iq": round(self.spatial_iq, 4),
            "probes": {r.name: {"score": round(r.score, 4),
                                "passed": r.passed,
                                "details": r.details}
                       for r in self.results},
        }

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def run_benchmark(checkpoint, verbose=True):
    """Run all tiny-spatial v1 probes against a checkpoint path OR a
    model instance (TinyJEPA or HierarchicalJEPA).

    Returns a BenchmarkReport (also printable / JSON-serializable).
    Deterministic: fixed seeds inside each probe.
    """
    if isinstance(checkpoint, str):
        model = TinyJEPA.load(checkpoint)
        name = checkpoint
    else:
        model = checkpoint
        name = type(model).__name__
    results = [probe(model) for probe in PROBES]
    report = BenchmarkReport(name, results)
    if verbose:
        report.print()
    return report
