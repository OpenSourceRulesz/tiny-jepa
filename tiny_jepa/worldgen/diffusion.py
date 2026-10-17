"""tiny-world: a tiny diffusion prior over world maps.

WorldPrior is a DDPM (Ho et al. 2020, miniaturized) over flattened 16x16
tile maps in {-1, +1}. The denoiser is a small MLP (~70k params) that
takes (noisy_map, time_embedding, style_onehot) and predicts the CLEAN
map x_0 (not the noise -- predicting noise forces the net to amplify a
1%-scale residual 100x at low t, which a tiny MLP cannot do; x_0 targets
stay in [-1, 1]). Sampling uses the x_0-parameterized DDPM posterior.

This is the "World Labs for your laptop" cheat made explicit: we
generate the *world description*, not pixels, and render it
deterministically — so every dreamed view is consistent by construction.
"""

import time

import numpy as np

from ..nn import MLP, Adam
from .maps import GRID, N_STYLES

T_DEFAULT = 40
CODE_DIM = GRID * GRID  # 256
TIME_DIM = 16  # sinusoidal time embedding (a scalar can't carry the
# highly t-dependent denoising function -- same drowning lesson again)


def _time_emb(t01, dim=TIME_DIM):
    """Sinusoidal embedding of t in [0,1). (B,) -> (B, dim)."""
    t01 = np.asarray(t01, dtype=np.float32)[:, None]
    freqs = np.exp(np.linspace(0, np.log(10000.0), dim // 2,
                               dtype=np.float32))[None, :]
    ang = t01 * freqs
    return np.concatenate([np.sin(ang), np.cos(ang)],
                          axis=1).astype(np.float32)


class WorldPrior:
    def __init__(self, n_styles=N_STYLES, T=T_DEFAULT, seed=0):
        self.n_styles = n_styles
        self.T = T
        # aggressive schedule: with small T the signal must be ~gone by
        # step T, or sampling starts from noise the net never saw.
        betas = np.linspace(1e-4, 0.15, T).astype(np.float64)
        alphas = 1.0 - betas
        self.alpha_bar = np.cumprod(alphas)          # (T,)
        self.alphas = alphas
        self.betas = betas
        # (noisy code 256 + time emb 16 + style onehot) -> noise 256
        self.net = MLP([CODE_DIM + TIME_DIM + n_styles, 128, 128,
                        CODE_DIM], seed=seed)

    # -- training ------------------------------------------------------
    def train_step(self, x0, styles, opt):
        """One denoising step: predict the CLEAN map x_0 from its noisy
        version. x0: (B,256) in {-1,+1}, styles: (B,) int. Returns loss."""
        B = len(x0)
        rng = np.random.default_rng()
        t = rng.integers(0, self.T, size=B)
        ab = self.alpha_bar[t][:, None]
        eps = rng.standard_normal((B, CODE_DIM)).astype(np.float32)
        xt = (np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps).astype(np.float32)
        onehot = np.zeros((B, self.n_styles), dtype=np.float32)
        onehot[np.arange(B), styles] = 3.0  # scaled: 3 dims must not
        # drown among 275 input dims (same lesson as hjepa's command)
        inp = np.concatenate([xt, _time_emb(t / self.T), onehot], axis=1)
        x0_hat = self.net.forward(inp)
        err = x0_hat - x0
        loss = float(np.mean(err ** 2))
        self.net.backward((2.0 * err / err.size).astype(np.float32))
        opt.step()
        return loss

    def fit(self, maps, styles, steps=1500, batch=128, seed=0,
            log_every=300):
        """maps: (N,16,16) int {0,1}; styles: (N,) int labels.
        Trains the denoiser with Adam."""
        rng = np.random.default_rng(seed)
        x0 = np.where(maps.reshape(len(maps), -1) == 1, 1.0, -1.0)
        x0 = x0.astype(np.float32)
        styles = np.asarray(styles, dtype=np.int64)
        opt = Adam(self.net.params_and_grads(), lr=1e-3)
        t0 = time.time()
        N = len(x0)
        for step in range(1, steps + 1):
            idx = rng.integers(N, size=batch)
            loss = self.train_step(x0[idx], styles[idx], opt)
            if log_every and step % log_every == 0:
                print(f"prior step {step:5d}/{steps} loss={loss:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        return self

    # -- sampling ------------------------------------------------------
    def sample(self, style=0, seed=None, n=1):
        """DDPM sampling with x_0 parameterization. Returns (n,16,16)
        int {0,1} maps (unvalidated — run env validation before use)."""
        rng = np.random.default_rng(seed)
        style = int(style) % self.n_styles
        onehot = np.zeros((n, self.n_styles), dtype=np.float32)
        onehot[:, style] = 3.0
        x = rng.standard_normal((n, CODE_DIM)).astype(np.float32)
        for ti in reversed(range(self.T)):
            temb = _time_emb(np.full(n, ti / self.T, dtype=np.float32))
            inp = np.concatenate([x, temb, onehot], axis=1)
            x0_hat = self.net.forward(inp)
            # DDPM posterior q(x_{t-1} | x_t, x_0), with x_0 = x0_hat:
            # coefficients are bounded (no epsilon amplification).
            ab_t = self.alpha_bar[ti]
            ab_prev = self.alpha_bar[ti - 1] if ti > 0 else 1.0
            a_t = self.alphas[ti]
            b_t = self.betas[ti]
            c0 = np.sqrt(ab_prev) * b_t / (1.0 - ab_t)
            ct = np.sqrt(a_t) * (1.0 - ab_prev) / (1.0 - ab_t)
            mean = (c0 * x0_hat + ct * x).astype(np.float32)
            if ti > 0:
                post_var = (1.0 - ab_prev) / (1.0 - ab_t) * b_t
                x = (mean + np.sqrt(post_var) *
                     rng.standard_normal((n, CODE_DIM)).astype(np.float32)
                     ).astype(np.float32)
            else:
                x = mean
        maps = (x.reshape(n, GRID, GRID) > 0).astype(np.int64)
        # hard border walls, always
        maps[:, 0, :] = maps[:, -1, :] = 1
        maps[:, :, 0] = maps[:, :, -1] = 1
        return maps

    # -- persistence ---------------------------------------------------
    def save(self, path):
        arrays = {f"net.{k}": v
                  for k, v in self.net.parameters().items()}
        arrays["T"] = np.array([self.T])
        arrays["n_styles"] = np.array([self.n_styles])
        arrays["alpha_bar"] = self.alpha_bar
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, seed=0):
        d = np.load(path)
        self = cls(n_styles=int(d["n_styles"][0]),
                   T=int(d["T"][0]), seed=seed)
        self.alpha_bar = d["alpha_bar"]
        self.betas = 1.0 - self.alpha_bar / np.concatenate(
            [np.ones(1), self.alpha_bar[:-1]])
        self.alphas = 1.0 - self.betas
        self.net.load_parameters(
            {k.split(".", 1)[1]: d[k] for k in d.files
             if k.startswith("net.")})
        return self
