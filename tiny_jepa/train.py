"""Training for tiny-jepa. Pure numpy, no dependencies.

Two entry points::

    from tiny_jepa.train import train_on_env, train_on_data
    from tiny_jepa.envs import OrbyWorld

    # 1) train on the built-in world (fresh episodes every step)
    model = train_on_env(lambda seed: OrbyWorld(seed=seed), steps=1500)

    # 2) bring your own world: supply recorded trajectories
    model = train_on_data(frames, actions, shaped_rewards,
                          pellet_xy, agent_xy, steps=1500)

Data shapes for train_on_data:
    frames        (N, T+1, 32, 32) float32 in [0, 1]
    actions       (N, T) int in [0, n_actions)
    shaped        (N, T) float32 (dense learning signal)
    pellet_xy    (N, T+1, 2) float32 pixels (for the perception heads +
                  the reconstruction spotlight; use zeros if unknown and
                  set aux_weight=0)
    agent_xy      (N, T+1, 2) float32 pixels (same note)

For train_on_env, the env factory must return an object with:
    .reset(seed), .step(a) -> (frame, reward), .render() -> (32,32),
    .agent (2,), .pellet (2,), .n_actions
"""

import time

import numpy as np

from .model import TinyJEPA, to_patches, one_hot, CANVAS, N_ACTIONS
from .nn import MLP, Adam
from .sigreg import SIGReg


def _pellet_weight_mask(pellet_xy, sigma=4.0, boost=6.0):
    """(B,2) pixel coords -> (B,1024) per-patch reconstruction weights that
    spotlight the pellet region so the encoder can't ignore it."""
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
    d2 = ((xx[None] - pellet_xy[:, 0, None, None]) ** 2 +
          (yy[None] - pellet_xy[:, 1, None, None]) ** 2)
    w = 1.0 + boost * np.exp(-d2 / (2 * sigma * sigma))
    p = w.reshape(w.shape[0], CANVAS // 8, 8, CANVAS // 8, 8)
    p = p.transpose(0, 1, 3, 2, 4).reshape(w.shape[0], -1, 64)
    return np.repeat(p.mean(axis=2), 64, axis=1).astype(np.float32)


class _Trainer:
    """Owns the EMA target encoder, optimizer, and one training step."""

    def __init__(self, model, lr=3e-4, ema=0.99,
                 sigreg_weight=0.0, sigreg_m=128, sigreg_seed=0,
                 rollout_steps=0, rollout_weight=1.0):
        self.model = model
        self.ema = ema
        self.sigreg_weight = sigreg_weight
        self.sigreg = (SIGReg(model.latent_dim, m=sigreg_m, seed=sigreg_seed)
                       if sigreg_weight > 0 else None)
        self.rollout_steps = int(rollout_steps)
        self.rollout_weight = float(rollout_weight)
        self.enc_tgt = MLP(model.arch["enc"], seed=999)
        self._sync_target(hard=True)
        params = []
        for name in ("enc", "pred", "reward", "dec", "pellet", "agenth"):
            params += model.nets[name].params_and_grads()
        self.opt = Adam(params, lr=lr)

    def _sync_target(self, hard=False):
        for l, lt in zip(self.model.nets["enc"].layers, self.enc_tgt.layers):
            if hard:
                lt.W[:] = l.W
                lt.b[:] = l.b
            else:
                lt.W[:] = self.ema * lt.W + (1 - self.ema) * l.W
                lt.b[:] = self.ema * lt.b + (1 - self.ema) * l.b

    def _rollout_branch(self, z0, fa_seq, fn_seq):
        """Multi-step rollout loss: unroll the predictor for H steps from z0
        using the true action sequence, MSE vs EMA-target encodings of the
        true future frames. Backprop through time; predictor grads are
        accumulated across unroll steps and ADDED to the single-step grads.

        z0: (B,D) online encoder outputs (start states).
        fa_seq: (B,H) int actions. fn_seq: (B,H,32,32) true future frames.
        Returns (loss, gz0) with gz0 the grad w.r.t. z0 (flows to encoder).
        """
        m = self.model
        pred = m.nets["pred"]
        B, D = z0.shape
        H = fa_seq.shape[1]
        nA = m.n_actions
        # stop-grad targets: EMA encoder on true future frames
        zt = self.enc_tgt.forward(
            to_patches(fn_seq.reshape(-1, CANVAS, CANVAS))).reshape(B, H, D)
        # save the single-step predictor grads (Linear.backward overwrites)
        saved = [(l.gW.copy(), l.gb.copy()) for l in pred.layers]
        # forward unroll, caching activations per step
        caches, z_hats, xs = [], [], []
        z = z0
        for k in range(H):
            pa = np.concatenate([z, one_hot(fa_seq[:, k], nA)], axis=1)
            z = pred.forward(pa)
            z_hats.append(z)
            caches.append([c.copy() for c in pred._cache])
            # Linear.backward reads layer.x for gW: must save/restore too
            xs.append([l.x.copy() for l in pred.layers])
        # backward through time
        acc = [(np.zeros_like(l.gW), np.zeros_like(l.gb)) for l in pred.layers]
        loss, gz_next, decay = 0.0, None, 1.0
        for k in reversed(range(H)):
            pred._cache = caches[k]
            for l, x in zip(pred.layers, xs[k]):
                l.x = x
            diff = z_hats[k] - zt[:, k]
            loss += decay * float(np.mean(diff ** 2))
            g = (2.0 * diff / diff.size) * decay
            if gz_next is not None:
                g = g + gz_next
            g_pa = pred.backward(g)
            for (aW, ab), l in zip(acc, pred.layers):
                aW += l.gW
                ab += l.gb
            gz_next = g_pa[:, :D]
            decay *= 0.9
        # merge: single-step grads + accumulated rollout grads
        for (sW, sb), (aW, ab), l in zip(saved, acc, pred.layers):
            l.gW[:] = sW + aW
            l.gb[:] = sb + ab
        return loss / H, gz_next

    def step(self, ft, fa, fn, shaped, rft, rfa, rshaped, pellet_xy,
             agent_xy, aux_weight=1.0, seq=None):
        m = self.model
        B = ft.shape[0]
        D = m.latent_dim

        # ---- predictor branch (JEPA loss vs EMA target, stop-grad) ----
        z = m.nets["enc"].forward(to_patches(ft))
        pa = np.concatenate([z, one_hot(fa, m.n_actions)], axis=1)
        pred_z = m.nets["pred"].forward(pa)
        zt = self.enc_tgt.forward(to_patches(fn))  # stop-grad target
        diff = pred_z - zt
        loss_pred = float(np.mean(diff ** 2))
        g = (2.0 * diff / diff.size)
        g_pa = m.nets["pred"].backward(g)
        # ---- multi-step rollout branch (opt-in): train the predictor on
        # its own unrolled futures so imagined rollouts stay goal-directed.
        # seq = (fa_seq (B,H), fn_seq (B,H,32,32)); starts from z above.
        loss_roll, g_roll_z0 = 0.0, None
        if self.rollout_steps > 0 and seq is not None:
            fa_seq, fn_seq = seq
            loss_roll, g_roll_z0 = self._rollout_branch(z, fa_seq, fn_seq)
        # variance floor: keep latent dims alive (std >= 1)
        std = z.std(axis=0)
        loss_var = float(np.mean(np.maximum(0.0, 1.0 - std) ** 2))
        gv = np.zeros_like(z)
        active = std < 1.0
        if active.any():
            coef = -2.0 * (1.0 - std) / (B * (std + 1e-8)) / D
            gv[:, active] = coef[None, active] * (z[:, active] - z[:, active].mean(axis=0, keepdims=True))

        # ---- reward branch (balanced: half pellet-eating transitions) --
        zr = m.nets["enc"].forward(to_patches(rft))
        pra = np.concatenate([zr, one_hot(rfa, m.n_actions)], axis=1)
        rhat = m.nets["reward"].forward(pra).ravel()
        rerr = rhat - rshaped
        loss_rew = float(np.mean(rerr ** 2))
        g_r = ((2.0 * rerr / rerr.size) * 2.0)[:, None]  # x2 weight
        g_pra_full = m.nets["reward"].backward(g_r)  # grad w.r.t. [z|a]

        # ---- auxiliary perception heads --------------------------------
        pxy = m.nets["pellet"].forward(z)
        perr = pxy - pellet_xy / CANVAS
        loss_pel = float(np.mean(perr ** 2)) * aux_weight
        g_pxy = (2.0 * perr / perr.size) * aux_weight
        g_pz = m.nets["pellet"].backward(g_pxy)
        axy = m.nets["agenth"].forward(z)
        aerr = axy - agent_xy / CANVAS
        loss_agt = float(np.mean(aerr ** 2)) * aux_weight
        g_axy = (2.0 * aerr / aerr.size) * aux_weight
        g_az = m.nets["agenth"].backward(g_axy)

        # ---- decoder branch (reconstruction -> encoder, pellet spotlit)-
        dec_out = m.nets["dec"].forward(z)
        derr = dec_out - to_patches(ft)
        wmask = _pellet_weight_mask(pellet_xy)
        loss_dec = float(np.mean(wmask * derr ** 2))
        g_dec = (2.0 * wmask * derr / derr.size) * 0.25
        g_dec_z = m.nets["dec"].backward(g_dec)

        # ---- SIGReg geometric regularization (opt-in) --------------------
        # Applied to the predictor branch's encoder outputs: spreads latents
        # so cosine distance becomes metrically meaningful (GWMR-style).
        loss_sig, g_sig = 0.0, None
        if self.sigreg is not None:
            loss_sig, g_sig = self.sigreg.loss_and_grad(z)

        # ---- encoder backward (replay activations per branch) -----------
        # Linear.backward overwrites grads (not accumulates): save the reward
        # branch's grads, run the main branch, then add them.
        m.nets["enc"].forward(to_patches(rft))
        m.nets["enc"].backward(g_pra_full[:, :D])
        saved = [(l.gW.copy(), l.gb.copy()) for l in m.nets["enc"].layers]
        m.nets["enc"].forward(to_patches(ft))
        g_enc = g_pa[:, :D] + gv + g_pz + g_az + g_dec_z
        if g_sig is not None:
            g_enc = g_enc + self.sigreg_weight * g_sig
        if g_roll_z0 is not None:
            g_enc = g_enc + self.rollout_weight * g_roll_z0
        m.nets["enc"].backward(g_enc)
        for l, (gw, gb) in zip(m.nets["enc"].layers, saved):
            l.gW += gw
            l.gb += gb

        # ---- clip + step -------------------------------------------------
        _clip_global_norm(
            [g for _, g in self.opt.params], 5.0)
        self.opt.step()
        self._sync_target()
        loss = (loss_pred + 2.0 * loss_rew + 0.25 * loss_dec + loss_var
                + loss_pel + loss_agt + self.sigreg_weight * loss_sig
                + self.rollout_weight * loss_roll)
        return {"loss": loss, "pred": loss_pred, "rew": loss_rew,
                "dec": loss_dec, "var": loss_var,
                "pel": loss_pel, "agt": loss_agt, "sig": loss_sig,
                "roll": loss_roll}


def _clip_global_norm(grads, max_norm):
    total = float(sum(float((g ** 2).sum()) for g in grads)) ** 0.5
    if total > max_norm:
        s = max_norm / (total + 1e-8)
        for g in grads:
            g *= s


def _sample(d, B, rng, eat_frac=0.0):
    """d: dict of (N,T+1,32,32) frames, (N,T) actions, (N,T) shaped,
    (N,T) binary, (N,T+1,2) pellet_xy, (N,T+1,2) agent_xy."""
    N, T = d["frames"].shape[0], d["frames"].shape[1] - 1
    eat_ep, eat_t = np.where(d["binary"] > 0.5)
    n_eat = int(B * eat_frac)
    eps = np.empty(B, dtype=int)
    steps = np.empty(B, dtype=int)
    if len(eat_ep) > 0 and n_eat > 0:
        pick = rng.integers(len(eat_ep), size=n_eat)
        eps[:n_eat] = eat_ep[pick]
        steps[:n_eat] = eat_t[pick]
    eps[n_eat:] = rng.integers(N, size=B - n_eat)
    steps[n_eat:] = rng.integers(T, size=B - n_eat)
    F, A, S, P, G = d["frames"], d["actions"], d["shaped"], d["pellet_xy"], d["agent_xy"]
    idx = (eps, steps)
    return (np.stack([F[e, t] for e, t in zip(eps, steps)]),
            np.array([A[e, t] for e, t in zip(eps, steps)]),
            np.stack([F[e, t + 1] for e, t in zip(eps, steps)]),
            np.array([S[e, t] for e, t in zip(eps, steps)], dtype=np.float32),
            np.stack([P[e, t] for e, t in zip(eps, steps)]).astype(np.float32),
            np.stack([G[e, t] for e, t in zip(eps, steps)]).astype(np.float32))


def _sample_seq(d, B, H, rng):
    """Sample H-step sequences for the rollout loss.

    Returns (ft, fa_seq, fn_seq, sh, px, ax):
      ft     (B,32,32) start frames
      fa_seq (B,H) int actions t..t+H-1
      fn_seq (B,H,32,32) future frames t+1..t+H
      sh/px/ax for the start timestep t (same semantics as _sample).
    """
    N, T = d["frames"].shape[0], d["frames"].shape[1] - 1
    eps = rng.integers(N, size=B)
    steps = rng.integers(T - H + 1, size=B)
    F, A, S, P, G = d["frames"], d["actions"], d["shaped"], d["pellet_xy"], d["agent_xy"]
    return (np.stack([F[e, t] for e, t in zip(eps, steps)]),
            np.stack([A[e, t:t + H] for e, t in zip(eps, steps)]),
            np.stack([F[e, t + 1:t + H + 1] for e, t in zip(eps, steps)]),
            np.array([S[e, t] for e, t in zip(eps, steps)], dtype=np.float32),
            np.stack([P[e, t] for e, t in zip(eps, steps)]).astype(np.float32),
            np.stack([G[e, t] for e, t in zip(eps, steps)]).astype(np.float32))


def _collect_env_episodes(env_fn, n_ep, ep_len, seed):
    """Roll out n_ep fresh episodes with a 65%-greedy / 35%-random policy."""
    rng = np.random.default_rng(seed)
    frames, actions, shaped, binary, pellet_xy, agent_xy = [], [], [], [], [], []
    for _ in range(n_ep):
        env = env_fn(seed=int(rng.integers(1e9)))
        fr, ac, sh, bi, px, ax = [env.render()], [], [], [], [env.pellet.copy()], [env.agent.copy()]
        for _ in range(ep_len):
            if rng.random() < 0.65:  # greedy toward pellet
                d = env.pellet - env.agent
                a = int(np.argmax([0.0, -d[1], d[1], -d[0], d[0]]))
            else:
                a = int(rng.integers(5))
            _, r = env.step(a)
            dist = float(np.linalg.norm(env.pellet - env.agent))
            s = getattr(env, "last_shaped", None)
            s = float(s) if s else float(r) - 0.02 * dist
            fr.append(env.render())
            ac.append(a)
            sh.append(s)
            bi.append(float(r))
            px.append(env.pellet.copy())
            ax.append(env.agent.copy())
        frames.append(np.stack(fr))
        actions.append(np.array(ac))
        shaped.append(np.array(sh, dtype=np.float32))
        binary.append(np.array(bi, dtype=np.float32))
        pellet_xy.append(np.stack(px))
        agent_xy.append(np.stack(ax))
    return {"frames": np.stack(frames), "actions": np.stack(actions),
            "shaped": np.stack(shaped), "binary": np.stack(binary),
            "pellet_xy": np.stack(pellet_xy).astype(np.float32),
            "agent_xy": np.stack(agent_xy).astype(np.float32)}


def _run_training(model, data_fn, steps, batch, seed, log_every, aux_weight,
                  sigreg_weight=0.0, sigreg_m=128, sigreg_seed=0,
                  rollout_steps=0, rollout_weight=1.0):
    rng = np.random.default_rng(seed)
    trainer = _Trainer(model, sigreg_weight=sigreg_weight,
                       sigreg_m=sigreg_m, sigreg_seed=sigreg_seed,
                       rollout_steps=rollout_steps,
                       rollout_weight=rollout_weight)
    t0 = time.time()
    for step in range(1, steps + 1):
        d = data_fn(rng)  # fresh data every step: no memorization possible
        seq = None
        if rollout_steps > 0:
            sft, sfa_seq, sfn_seq, sh, px, ax = _sample_seq(
                d, batch, rollout_steps, rng)
            ft, fa, fn = sft, sfa_seq[:, 0], sfn_seq[:, 0]
            seq = (sfa_seq, sfn_seq)
        else:
            ft, fa, fn, sh, px, ax = _sample(d, batch, rng, eat_frac=0.0)
        rft, rfa, _, rsh, _, _ = _sample(d, batch, rng, eat_frac=0.5)
        stats = trainer.step(ft, fa, fn, sh, rft, rfa, rsh, px, ax,
                             aux_weight=aux_weight, seq=seq)
        if log_every and step % log_every == 0:
            print(f"step {step:5d}/{steps} " +
                  " ".join(f"{k}={v:.4f}" for k, v in stats.items()) +
                  f" ({time.time()-t0:.0f}s)", flush=True)
    return model


def train_on_env(env_fn, steps=1500, batch=256, ep_len=64, seed=0,
                 log_every=150, aux_weight=1.0, latent_dim=32,
                 n_actions=N_ACTIONS, sigreg_weight=0.0, sigreg_m=128,
                 sigreg_seed=0, rollout_steps=0, rollout_weight=1.0):
    """Train a TinyJEPA on fresh episodes from env_fn every step.

    env_fn(seed) -> env with .reset(seed), .step(a)->(frame, reward),
    .render()->(32,32), .agent, .pellet. See module docstring.

    sigreg_weight > 0 enables SIGReg geometric regularization on the
    encoder (see sigreg.py); 0.0 (default) leaves training unchanged.

    rollout_steps > 0 enables a multi-step rollout loss on the predictor:
    unroll H=rollout_steps steps from the encoded state with the true
    action sequence, MSE vs EMA-target encodings of true future frames
    (decay 0.9 per step). Trains the predictor on its own imagined futures
    so rollouts stay goal-directed for planning.
    """
    model = TinyJEPA(latent_dim=latent_dim, n_actions=n_actions, seed=seed)

    def data_fn(rng):
        return _collect_env_episodes(env_fn, 12, ep_len, int(rng.integers(1e9)))

    return _run_training(model, data_fn, steps, batch, seed,
                         log_every, aux_weight, sigreg_weight=sigreg_weight,
                         sigreg_m=sigreg_m, sigreg_seed=sigreg_seed,
                         rollout_steps=rollout_steps,
                         rollout_weight=rollout_weight)


def train_on_data(frames, actions, shaped, pellet_xy=None, agent_xy=None,
                  steps=1500, batch=256, seed=0, log_every=150,
                  latent_dim=32, n_actions=N_ACTIONS, sigreg_weight=0.0,
                  sigreg_m=128, sigreg_seed=0, rollout_steps=0,
                  rollout_weight=1.0):
    """Train a TinyJEPA on your own recorded trajectories. Shapes documented
    in the module docstring. binary reward is derived as (shaped > 0.5)."""
    frames = np.asarray(frames, dtype=np.float32)
    actions = np.asarray(actions, dtype=int)
    shaped = np.asarray(shaped, dtype=np.float32)
    N, T1 = frames.shape[0], frames.shape[1]
    if pellet_xy is None:
        pellet_xy = np.zeros((N, T1, 2), dtype=np.float32)
    if agent_xy is None:
        agent_xy = np.zeros((N, T1, 2), dtype=np.float32)
    aux_weight = 0.0 if (np.abs(pellet_xy).max() == 0 and np.abs(agent_xy).max() == 0) else 1.0
    d = {"frames": frames, "actions": actions, "shaped": shaped,
         "binary": (shaped > 0.5).astype(np.float32),
         "pellet_xy": np.asarray(pellet_xy, dtype=np.float32),
         "agent_xy": np.asarray(agent_xy, dtype=np.float32)}
    model = TinyJEPA(latent_dim=latent_dim, n_actions=n_actions, seed=seed)
    return _run_training(model, lambda rng: d, steps, batch, seed,
                         log_every, aux_weight, sigreg_weight=sigreg_weight,
                         sigreg_m=sigreg_m, sigreg_seed=sigreg_seed,
                         rollout_steps=rollout_steps,
                         rollout_weight=rollout_weight)
