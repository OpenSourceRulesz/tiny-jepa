"""Experiment: train tiny-jepa's perception heads on IMAGINED latents (arm world).

Background (2026-10-15 sim-arm eval): on the 2-link arm world, perception and
imagination transfer (2.8px target err, 4.0px tip err, imagined-vs-true latent
cosine 0.84 at 8 steps) but control does not (greedy 0.12 targets/ep vs 0.25
random, reward-head MPC 0.0). Diagnosis: the tip/target perception heads were
trained only on latents encoded from REAL frames; on imagined latents they
disagree with themselves by 33px after 1 step, 48px after 6 (32px canvas).

This experiment:
  1. Trains the `pellet` + `agenth` heads on imagined latents (mixed 50/50
     with real-encoded latents so real-frame vision does not regress),
     encoder/predictor/decoder/reward frozen.
  2. Re-measures the self-disagreement curve (the 33/48px metric).
  3. Re-tests MPC on the arm scoreboard with a perception-based CEM:
     score(action seq) = -min_k || tip_head(z_hat_k) - tgt_head(z_hat_k) ||,
     i.e. plan in imagination toward the imagined tip reaching the imagined
     target. Controls: same MPC with the ORIGINAL heads (should score ~0,
     isolating the heads as the blocker), plus greedy and random baselines
     in the identical harness.

Usage:  cd ~/workspace/tinyjepa && .venv/bin/python exp_imagined_heads.py
Output: results_imagined_heads.json + checkpoints/arm_v2.npz
"""

import json
import time

import numpy as np

from tiny_jepa.model import TinyJEPA, CANVAS
from tiny_jepa.envs_arm import ArmWorld, collect_arm_episodes, arm_greedy_action
from tiny_jepa.nn import Adam

H = 8              # imagination horizon for head training
SEED = 7
N_TRAIN_EPS = 224
N_VAL_EPS = 32
EP_LEN = 64
BATCH = 256
TRAIN_STEPS = 2500
LR = 3e-4


def head_xy(model, head_name, z):
    """Head prediction in pixels for latent batch z (B, latent_dim)."""
    return model.nets[head_name].forward(np.asarray(z, np.float32)) * CANVAS


def imagine_rollout(model, z0, actions):
    """(H, latent_dim) imagined latents; actions = list of python ints."""
    return model.imagine(np.asarray(z0, dtype=np.float32).ravel(),
                         [int(a) for a in actions])


def imagined_abs_error(model, data, ks=(1, 2, 4, 6, 8), n_samples=400, seed=0):
    """Mean || head(z_hat_k) - TRUE tip/target at t+k || in px.
    The task-relevant accuracy of reading imagined latents."""
    rng = np.random.default_rng(seed)
    F, A = data["frames"], data["actions"]
    n_eps = F.shape[0]
    out = {}
    for head, key in (("agenth", "agent"), ("pellet", "pellet")):
        errs = {k: [] for k in ks}
        for _ in range(n_samples):
            ep = int(rng.integers(n_eps))
            t = int(rng.integers(0, EP_LEN - max(ks)))
            k = int(rng.choice(ks))
            z0 = model.encode(F[ep, t])
            zhat = imagine_rollout(model, z0, A[ep, t:t + k])[-1]
            d = float(np.linalg.norm(head_xy(model, head, zhat[None])[0]
                                     - data[key][ep, t + k]))
            errs[k].append(d)
        out[head] = {str(k): float(np.mean(errs[k])) for k in ks}
    return out


def disagreement_curve(model, data, ks=(1, 2, 4, 6, 8), n_samples=400, seed=0):
    """Mean || head(z_hat_k) - head(encode(true frame_k)) || in px,
    for the tip (agenth) and target (pellet) heads."""
    rng = np.random.default_rng(seed)
    F, A = data["frames"], data["actions"]
    n_eps = F.shape[0]
    out = {}
    for head in ("agenth", "pellet"):
        errs = {k: [] for k in ks}
        for _ in range(n_samples):
            ep = int(rng.integers(n_eps))
            t = int(rng.integers(0, EP_LEN - max(ks)))
            k = int(rng.choice(ks))
            z0 = model.encode(F[ep, t])
            zhat = imagine_rollout(model, z0, A[ep, t:t + k])[-1]
            ztrue = model.encode(F[ep, t + k])
            d = float(np.linalg.norm(head_xy(model, head, zhat[None])[0]
                                     - head_xy(model, head, ztrue[None])[0]))
            errs[k].append(d)
        out[head] = {str(k): float(np.mean(errs[k])) for k in ks}
    return out


def real_frame_error(model, data, n_samples=800, seed=1):
    """Perception error in px on real frames (must not regress)."""
    rng = np.random.default_rng(seed)
    F = data["frames"]
    n_eps = F.shape[0]
    errs = {"agenth": [], "pellet": []}
    keys = {"agenth": "agent", "pellet": "pellet"}
    for _ in range(n_samples):
        ep = int(rng.integers(n_eps))
        t = int(rng.integers(0, EP_LEN + 1))
        z = model.encode(F[ep, t])
        for head, dk in (("agenth", "agent"), ("pellet", "pellet")):
            pred = head_xy(model, head, z[None])[0]
            errs[head].append(float(np.linalg.norm(pred - data[dk][ep, t])))
    return {h: float(np.mean(v)) for h, v in errs.items()}


def build_head_dataset(model, data, seed=0):
    """(Z, tip_xy, tgt_xy): imagined-latent pairs + real-encoded pairs, 50/50.
    Coords normalized to [0,1] (heads output * CANVAS)."""
    rng = np.random.default_rng(seed)
    F, A = data["frames"], data["actions"]
    tip, tgt = data["agent"] / CANVAS, data["pellet"] / CANVAS
    n_eps = F.shape[0]
    Z, Y_tip, Y_tgt = [], [], []
    per_ep = 8
    for ep in range(n_eps):
        z_all = model.encode_batch(F[ep])          # (T+1, latent)
        for _ in range(per_ep):
            t = int(rng.integers(0, EP_LEN - H))
            k = int(rng.integers(1, H + 1))
            zhat = imagine_rollout(model, z_all[t], A[ep, t:t + k])[-1]
            Z.append(zhat)
            Y_tip.append(tip[ep, t + k])
            Y_tgt.append(tgt[ep, t + k])
            # real-encoded counterpart keeps real-frame vision intact
            Z.append(z_all[t + k])
            Y_tip.append(tip[ep, t + k])
            Y_tgt.append(tgt[ep, t + k])
    return (np.array(Z, dtype=np.float32),
            np.array(Y_tip, dtype=np.float32),
            np.array(Y_tgt, dtype=np.float32))


def train_heads(model, Z, Y_tip, Y_tgt, val_data, steps=TRAIN_STEPS,
                batch=BATCH, lr=LR, seed=0):
    rng = np.random.default_rng(seed)
    heads = [model.nets["agenth"], model.nets["pellet"]]
    opt = Adam(heads[0].params_and_grads() + heads[1].params_and_grads(), lr=lr)
    n = Z.shape[0]
    val_hist = []
    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        zb, tb, gb = Z[idx], Y_tip[idx], Y_tgt[idx]
        # tip head
        pt = heads[0].forward(zb)
        g = 2.0 * (pt - tb) / batch
        heads[0].backward(g)
        # target head
        pg = heads[1].forward(zb)
        g = 2.0 * (pg - gb) / batch
        heads[1].backward(g)
        opt.step()
        if (step + 1) % 500 == 0:
            d = disagreement_curve(model, val_data, ks=(1, 6), n_samples=150,
                                   seed=step)["agenth"]
            val_hist.append({"step": step + 1, "tip_k1": d["1"],
                             "tip_k6": d["6"]})
            print(f"  step {step + 1}: val tip disagreement "
                  f"k=1 {d['1']:.1f}px k=6 {d['6']:.1f}px", flush=True)
    return val_hist


def perception_cem_action(model, frame, horizon=6, n_samples=32, n_elite=8,
                          n_iters=2, seed=None):
    """One CEM step scored by imagined tip->target distance (perception heads
    read the imagined latents)."""
    rng = np.random.default_rng(seed)
    n_actions = model.n_actions
    z0 = model.encode(frame)
    logits = np.zeros((horizon, n_actions))
    best_score, best_seq = -np.inf, None

    def seq_score(seq):
        fut = imagine_rollout(model, z0, seq)              # (h, latent)
        tip = head_xy(model, "agenth", fut)
        tgt = head_xy(model, "pellet", fut)
        return -float(np.min(np.linalg.norm(tip - tgt, axis=1)))

    for _ in range(n_iters):
        m = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(m) / np.exp(m).sum(axis=1, keepdims=True)
        cum = np.cumsum(probs, axis=1)
        seqs = (rng.random((n_samples, horizon, 1)) < cum[None]).argmax(2)
        scores = np.array([seq_score([int(a) for a in s]) for s in seqs])
        b = int(np.argmax(scores))
        if scores[b] > best_score:
            best_score, best_seq = float(scores[b]), [int(a) for a in seqs[b]]
        elite = seqs[np.argsort(scores)[-n_elite:]]
        counts = np.zeros((horizon, n_actions))
        for idx in elite:
            for t in range(horizon):
                counts[t, seqs[idx, t]] += 1
        logits = np.log((counts + 1.0) / (n_elite + n_actions))
    return int(best_seq[0])


def evaluate_controller(make_action, n_eps=16, ep_len=200, seed=0, tag=""):
    """Targets per episode on fresh ArmWorlds."""
    rng = np.random.default_rng(seed)
    scores = []
    for e in range(n_eps):
        w = ArmWorld(seed=int(rng.integers(1e9)))
        frame = w.render()
        got = 0
        for _ in range(ep_len):
            a = make_action(frame, w, e)
            frame, r = w.step(a)
            got += int(r > 0.5)
        scores.append(got)
    mean = float(np.mean(scores))
    print(f"  [{tag}] {mean:.2f} targets/ep over {n_eps} eps", flush=True)
    return mean


def main():
    t0 = time.time()
    print("loading arm_v1...", flush=True)
    model = TinyJEPA.load("tiny_jepa/checkpoints/arm_v1.npz")

    print("collecting episodes...", flush=True)
    train_data = collect_arm_episodes(N_TRAIN_EPS, EP_LEN, seed=11)
    val_data = collect_arm_episodes(N_VAL_EPS, EP_LEN, seed=999)

    print("baseline metrics (original heads):", flush=True)
    base_dis = disagreement_curve(model, val_data)
    base_abs = imagined_abs_error(model, val_data)
    base_real = real_frame_error(model, val_data)
    print("  disagreement:", json.dumps(base_dis), flush=True)
    print("  imagined-abs-err:", json.dumps(base_abs), flush=True)
    print("  real-frame err:", json.dumps(base_real), flush=True)

    print("baselines in eval harness:", flush=True)
    rng = np.random.default_rng(0)
    base_greedy = evaluate_controller(
        lambda f, w, e: arm_greedy_action(model, f, w), tag="greedy(orig heads)")
    base_random = evaluate_controller(
        lambda f, w, e: int(rng.integers(5)), tag="random")
    base_cem = evaluate_controller(
        lambda f, w, e: perception_cem_action(model, f, seed=e * 1000 + 7),
        n_eps=10, tag="perception-CEM(orig heads)")

    print("building imagined-latent head dataset...", flush=True)
    Z, Y_tip, Y_tgt = build_head_dataset(model, train_data)
    print(f"  {Z.shape[0]} pairs (50% imagined / 50% real)", flush=True)

    print("training heads on imagined latents...", flush=True)
    val_hist = train_heads(model, Z, Y_tip, Y_tgt, val_data)

    print("post-training metrics:", flush=True)
    post_dis = disagreement_curve(model, val_data, seed=123)
    post_abs = imagined_abs_error(model, val_data, seed=125)
    post_real = real_frame_error(model, val_data, seed=124)
    print("  disagreement:", json.dumps(post_dis), flush=True)
    print("  imagined-abs-err:", json.dumps(post_abs), flush=True)
    print("  real-frame err:", json.dumps(post_real), flush=True)

    print("re-testing MPC with trained heads:", flush=True)
    post_cem = evaluate_controller(
        lambda f, w, e: perception_cem_action(model, f, seed=e * 1000 + 7),
        n_eps=16, tag="perception-CEM(trained heads)")

    model.save("tiny_jepa/checkpoints/arm_v2.npz")
    results = {
        "baseline_disagreement_px": base_dis,
        "baseline_imagined_abs_err_px": base_abs,
        "baseline_real_frame_err_px": base_real,
        "post_disagreement_px": post_dis,
        "post_imagined_abs_err_px": post_abs,
        "post_real_frame_err_px": post_real,
        "val_history": val_hist,
        "targets_per_episode": {
            "greedy_orig_heads": base_greedy,
            "random": base_random,
            "perception_cem_orig_heads": base_cem,
            "perception_cem_trained_heads": post_cem,
        },
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open("results_imagined_heads.json", "w") as f:
        json.dump(results, f, indent=2)
    print("VERDICT:", json.dumps(results["targets_per_episode"]), flush=True)
    print(f"done in {results['elapsed_min']} min; saved checkpoints/arm_v2.npz",
          flush=True)


if __name__ == "__main__":
    main()
