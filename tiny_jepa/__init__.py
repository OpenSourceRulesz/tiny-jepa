"""tiny-jepa: a tiny, hackable, action-conditioned JEPA world model.

Pure numpy. Zero required dependencies. Pip-installable, ONNX-exportable.
"""

from .model import TinyJEPA, to_patches, from_patches, one_hot
from .envs import OrbyWorld, GymWrapper

__all__ = [
    "TinyJEPA",
    "OrbyWorld",
    "GymWrapper",
    "to_patches",
    "from_patches",
    "one_hot",
]

__version__ = "0.1.0"
