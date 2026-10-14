"""tiny-jepa: a tiny, hackable, action-conditioned JEPA world model.

Pure numpy. Zero required dependencies. Pip-installable, ONNX-exportable.
"""

from .model import TinyJEPA, to_patches, from_patches, one_hot
from .envs import OrbyWorld, GymWrapper
from .hjepa import HierarchicalJEPA, train_slow_level

__all__ = [
    "TinyJEPA",
    "HierarchicalJEPA",
    "train_slow_level",
    "OrbyWorld",
    "GymWrapper",
    "to_patches",
    "from_patches",
    "one_hot",
]

__version__ = "0.1.0"
