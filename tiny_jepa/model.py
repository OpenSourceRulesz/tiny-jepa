"""TinyJEPA: a tiny action-conditioned Joint Embedding Predictive Architecture.

The whole model: an encoder, an action-conditioned latent predictor (the
"world model"), a reward head (the "cost"), a decoder (visualizes imagined
latents), and auxiliary perception heads. Pure numpy, no dependencies.

Typical workflow::

    from tiny_jepa import TinyJEPA, OrbyWorld

    model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
    env = OrbyWorld(seed=0)
    frame = env.render()            # (32, 32) float32 in [0, 1]

    # Imagine: what happens if I go right, right, up?
    z = model.encode(frame)
    future = model.imagine(z, [4, 4, 1])       # 3 latent states
    dream_frames = [model.decode(fz) for fz in future]

    # Perceive + act: where is the pellet, and what should I do?
    pellet_xy = model.perceive_pellet(frame)
    action = model.plan(frame, agent_xy=env.agent)
"""

import json
import numpy as np

from .nn import MLP

CANVAS = 32
PATCH = 8
N_ACTIONS = 5
LATENT_DIM = 32

# (dx, dy) per action index: 0=stay, 1=up, 2=down, 3=left, 4=right
ACTION_DELTAS = np.array([
    [0.0, 0.0],
    [0.0, -2.0],
    [0.0, 2.0],
    [-2.0, 0.0],
    [2.0, 0.0],
], dtype=np.float32)

ARCH = {
    "enc": [1024, 128, 64, 32],
    "pred": [37, 64, 32],
    "reward": [37, 64, 1],
    "dec": [32, 64, 1024],
    "pellet": [32, 32, 2],
    "agenth": [32, 32, 2],
}

SUBJECTS = ("enc", "pred", "reward", "dec", "pellet", "agenth")


def to_patches(frames):
    """(B,32,32) -> (B,1024): 8x8 patches, row-major, flattened."""
    frames = np.asarray(frames, dtype=np.float32)
    single = frames.ndim == 2
    if single:
        frames = frames[None]
    B = frames.shape[0]
    p = frames.reshape(B, CANVAS // PATCH, PATCH, CANVAS // PATCH, PATCH)
    out = p.transpose(0, 1, 3, 2, 4).reshape(B, -1)
    return out[0] if single else out


def from_patches(patches):
    """(B,1024) -> (B,32,32): inverse of to_patches."""
    patches = np.asarray(patches, dtype=np.float32)
    single = patches.ndim == 1
    if single:
        patches = patches[None]
    B = patches.shape[0]
    p = patches.reshape(B, CANVAS // PATCH, CANVAS // PATCH, PATCH, PATCH)
    out = p.transpose(0, 1, 3, 2, 4).reshape(B, CANVAS, CANVAS)
    return out[0] if single else out


def one_hot(actions, n=N_ACTIONS):
    a = np.asarray(actions, dtype=int).ravel()
    oh = np.zeros((a.shape[0], n), dtype=np.float32)
    oh[np.arange(a.shape[0]), np.clip(a, 0, n - 1)] = 1.0
    return oh


class TinyJEPA:
    """Action-conditioned JEPA world model."""

    def __init__(self, latent_dim=LATENT_DIM, n_actions=N_ACTIONS, seed=0):
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        arch = {k: ([1024, 128, 64, latent_dim] if k == "enc"
                    else [latent_dim + n_actions, 64, latent_dim] if k == "pred"
                    else [latent_dim + n_actions, 64, 1] if k == "reward"
                    else [latent_dim, 64, 1024] if k == "dec"
                    else [latent_dim, latent_dim, 2])
                for k in SUBJECTS}
        self.arch = arch
        self.nets = {k: MLP(v, seed=seed + i)
                     for i, (k, v) in enumerate(arch.items())}

    # ---- core JEPA ops -------------------------------------------------
    def encode(self, frame):
        """(32,32) grayscale frame -> (latent_dim,) latent vector."""
        return self.nets["enc"].forward(to_patches(frame)[None])[0]

    def encode_batch(self, frames):
        """(B,32,32) -> (B,latent_dim)."""
        return self.nets["enc"].forward(to_patches(frames))

    def imagine(self, z, actions):
        """Roll the predictor forward: z (latent_dim,), actions (list of int)
        -> array (len(actions), latent_dim) of imagined future latents."""
        z = np.asarray(z, dtype=np.float32).ravel()
        out = []
        for a in actions:
            z = self.nets["pred"].forward(
                np.concatenate(
                    [z, one_hot([a], self.n_actions)[0]])[None])[0]
            out.append(z.copy())
        return np.array(out)

    def decode(self, z):
        """latent -> (32,32) reconstructed/predicted frame in [0,1]-ish."""
        img = from_patches(self.nets["dec"].forward(np.asarray(z, np.float32)[None])[0])
        return np.clip(img, 0.0, 1.0)

    def dream(self, frame, actions):
        """frame + action sequence -> decoded frames of the imagined future."""
        return [self.decode(fz) for fz in self.imagine(self.encode(frame), actions)]

    # ---- perception / cost ---------------------------------------------
    def perceive_pellet(self, frame):
        """Where the model thinks the pellet is, in pixels. Learned vision."""
        return self.nets["pellet"].forward(self.encode(frame)[None])[0] * CANVAS

    def perceive_agent(self, frame):
        """Where the model thinks the agent is, in pixels (auxiliary head)."""
        return self.nets["agenth"].forward(self.encode(frame)[None])[0] * CANVAS

    def predict_reward(self, z, action):
        """Predicted shaped reward for (latent, action)."""
        z = np.asarray(z, dtype=np.float32).ravel()
        return float(self.nets["reward"].forward(
            np.concatenate([z, one_hot([action], self.n_actions)[0]])[None])[0, 0])

    # ---- planning --------------------------------------------------------
    def plan(self, frame, agent_xy, pellet_xy=None):
        """Greedy 1-step planner: simulate each move's kinematics, pick the
        action minimizing predicted distance to the (perceived) pellet.
        agent_xy is proprioception (the controller knows its own body);
        the pellet position comes from learned vision unless given."""
        if pellet_xy is None:
            pellet_xy = self.perceive_pellet(frame)
        pellet_xy = np.asarray(pellet_xy, dtype=np.float32)
        agent_xy = np.asarray(agent_xy, dtype=np.float32)
        best, best_d = 0, float("inf")
        for a, (dx, dy) in enumerate(ACTION_DELTAS):
            nx = min(max(agent_xy[0] + dx, 2.5), CANVAS - 2.5)
            ny = min(max(agent_xy[1] + dy, 2.5), CANVAS - 2.5)
            d = (nx - pellet_xy[0]) ** 2 + (ny - pellet_xy[1]) ** 2
            if d < best_d:
                best_d, best = d, a
        return int(best)

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        """Save checkpoint as .npz (arrays + json config)."""
        arrays = {}
        for name in SUBJECTS:
            for k, v in self.nets[name].parameters().items():
                arrays[f"{name}.{k}"] = v
        config = json.dumps({
            "latent_dim": self.latent_dim, "n_actions": self.n_actions,
            "arch": self.arch, "subjects": list(SUBJECTS),
        })
        np.savez(path, config=np.array(config),
                 **{k: np.asarray(v) for k, v in arrays.items()})

    @classmethod
    def load(cls, path):
        """Load a checkpoint saved with .save()."""
        data = np.load(path, allow_pickle=True)
        config = json.loads(str(data["config"]))
        model = cls(latent_dim=config["latent_dim"],
                    n_actions=config["n_actions"], seed=0)
        for name in config["subjects"]:
            params = {}
            for key in data.files:
                if key.startswith(name + "."):
                    params[key[len(name) + 1:]] = data[key]
            model.nets[name].load_parameters(params)
        return model
