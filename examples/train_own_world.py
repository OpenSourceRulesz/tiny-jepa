"""Train your own tiny world model from scratch.

Trains a fresh TinyJEPA on the built-in OrbyWorld (fresh episodes every
step, ~8 min on CPU) and saves the checkpoint.

Run:  python examples/train_own_world.py [steps]
"""

import sys

from tiny_jepa import OrbyWorld
from tiny_jepa.train import train_on_env

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
model = train_on_env(lambda seed: OrbyWorld(seed=seed), steps=steps,
                     log_every=max(1, steps // 10))
model.save("tiny_jepa/checkpoints/orby_mine.npz")
print("saved to tiny_jepa/checkpoints/orby_mine.npz")
