"""Goal-conditioned value head: V(z, z_goal) -> P(pellet eaten soon).

A small MLP that predicts the probability of task success (pellet eaten
within K steps) from the current latent z and a goal embedding z_goal.
This replaces the broken latent-distance metric in planning: instead of
minimizing dist(z_imagined, z_goal) -- which isn't task-aligned -- CEM
maximizes V(z_imagined, z_goal), a *learned* task-aligned score.

Pure numpy. Train with binary cross-entropy on (z, z_goal, outcome) data.
"""

import numpy as np

from .nn import MLP, Adam


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class ValueHead:
    """MLP: concat(z, z_goal) -> logit; sigmoid -> P(success)."""

    def __init__(self, latent_dim=32, hidden=64, seed=0):
        self.latent_dim = latent_dim
        self.net = MLP([2 * latent_dim, hidden, 1], seed=seed)
        self._adam = None

    def predict(self, z, zg):
        """(B,D),(B,D) -> (B,) probabilities. Single pair: pass 1-D arrays."""
        z = np.asarray(z, dtype=np.float32)
        zg = np.asarray(zg, dtype=np.float32)
        single = z.ndim == 1
        if single:
            z, zg = z[None], zg[None]
        x = np.concatenate([z, zg], axis=-1)
        p = _sigmoid(self.net.forward(x)[:, 0])
        return float(p[0]) if single else p

    def train_step(self, z, zg, y, lr=1e-3):
        """One Adam step of binary cross-entropy. Returns (loss, accuracy)."""
        z = np.asarray(z, dtype=np.float32)
        zg = np.asarray(zg, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        B = y.shape[0]
        x = np.concatenate([z.reshape(B, -1), zg.reshape(B, -1)], axis=-1)
        logits = self.net.forward(x)[:, 0]
        p = _sigmoid(logits)
        eps = 1e-7
        loss = float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean())
        acc = float(((p > 0.5) == (y > 0.5)).mean())
        # d BCE / d logit = (p - y) / B
        self.net.backward(((p - y) / B)[:, None])
        if self._adam is None or self._adam.lr != lr:
            self._adam = Adam(self.net.params_and_grads(), lr=lr)
        self._adam.step()
        return loss, acc

    def save(self, path):
        arrays = {f"net.{k}": v for k, v in self.net.parameters().items()}
        np.savez(path, latent_dim=np.array(self.latent_dim), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        vh = cls(latent_dim=int(data["latent_dim"]))
        vh.net.load_parameters({k[4:]: data[k] for k in data.files
                                if k.startswith("net.")})
        return vh
