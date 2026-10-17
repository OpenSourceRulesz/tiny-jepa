"""SIGReg-style geometric regularization for tiny-jepa (pure numpy).

Enforces near-Gaussian marginals on random projections of the encoder's
latent vectors, spreading embeddings across the hypersphere so that cosine
distance becomes metrically meaningful -- the property a cosine-to-goal
planner (GWMR-style) relies on.

Simplified robust version of Yarats et al. (2025) SIGReg. The full version
matches 1-D Wasserstein distance of each projection to N(0,1); this version
matches the first two moments, which is cheaper, fully differentiable in
closed form, and stable at tiny batch sizes:

    L_sig(z) = mean_m( mean(r_m^T z)^2 + (std(r_m^T z) - 1)^2 )

over M fixed random unit-norm projection directions r_m.

Usage (opt-in in train.py)::

    reg = SIGReg(dim=32, m=128, seed=0)
    loss, grad_z = reg.loss_and_grad(z)   # z: (B, D)
"""

import numpy as np


class SIGReg:
    """Fixed random projections + moment-matching loss and its gradient."""

    def __init__(self, dim, m=128, seed=0):
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((dim, m)).astype(np.float32)
        R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-12
        self.R = R
        self.dim = dim
        self.m = m

    def loss_and_grad(self, z):
        """z: (B, D) encoder latents. Returns (loss: float, grad: (B, D))
        with grad = dL/dz, ready to add into the encoder's backward input."""
        z = np.asarray(z, dtype=np.float32)
        B, D = z.shape
        M = self.m
        P = z @ self.R                              # (B, M) projections
        mu = P.mean(axis=0)                         # (M,)
        var = P.var(axis=0)                         # (M,) population variance
        std = np.sqrt(var + 1e-8)                   # (M,)
        loss = float(np.mean(mu ** 2 + (std - 1.0) ** 2))

        # dL/dP[b,m]: 2*mu_m/B  +  2*(std_m-1)*(P[b,m]-mu_m)/(B*std_m)
        # (the (1-1/B) Bessel terms cancel exactly in d var/dz)
        t1 = 2.0 * mu / B                                           # (M,)
        t2 = 2.0 * (std - 1.0) / (B * (std + 1e-12))                # (M,)
        C = P - mu[None, :]                        # (B, M) centered
        G = t1[None, :] + t2[None, :] * C          # (B, M) dL/dP
        grad = (G @ self.R.T) / M                  # (B, D) dL/dz
        return loss, grad.astype(np.float32)

    def projection_stats(self, z):
        """Diagnostics: per-projection mean/std of z. Returns dict."""
        P = np.asarray(z, dtype=np.float32) @ self.R
        return {
            "mean_abs": float(np.abs(P.mean(axis=0)).mean()),
            "std_mean": float(P.std(axis=0).mean()),
            "std_std": float(P.std(axis=0).std()),
        }
