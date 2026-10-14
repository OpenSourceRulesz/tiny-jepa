"""Environments for tiny-jepa.

Ships the built-in OrbyWorld (dependency-free, numpy only) and a
Gymnasium-compatible wrapper so the model plugs into standard RL workflows::

    from tiny_jepa.envs import OrbyWorld, GymWrapper

    env = OrbyWorld(seed=0)          # raw: env.render() -> (32,32), env.step(a)
    gym_env = GymWrapper(seed=0)     # gymnasium API, if gymnasium installed
"""

import numpy as np

from ._orby import OrbyWorld  # noqa: F401  (re-exported)

__all__ = ["OrbyWorld", "GymWrapper"]


class GymWrapper:
    """Gymnasium wrapper around OrbyWorld.

    Observation: {"frame": (32,32) float32, "agent_xy": (2,) float32}.
    Action: Discrete(5) with the same 0=stay,1=up,2=down,3=left,4=right map.
    Requires the optional ``gymnasium`` package.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed=0, max_steps=256):
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as e:
            raise ImportError(
                "GymWrapper needs the optional 'gymnasium' package: "
                "pip install 'tiny-jepa[gym]'") from e
        self._gym = gym
        self._spaces = spaces
        self.env = OrbyWorld(seed=seed)
        self.max_steps = max_steps
        self._t = 0
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(0.0, 1.0, shape=(32, 32), dtype=np.float32),
            "agent_xy": spaces.Box(0.0, 32.0, shape=(2,), dtype=np.float32),
        })

    def _obs(self):
        return {"frame": self.env.render(),
                "agent_xy": self.env.agent.astype(np.float32)}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env = OrbyWorld(seed=seed)
        else:
            self.env.reset()
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        _, reward = self.env.step(int(action))
        self._t += 1
        terminated = False
        truncated = self._t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {}

    def render(self):
        frame = self.env.render()
        rgb = (frame * 255).astype(np.uint8)
        return np.stack([rgb, rgb, rgb], axis=-1)

    def close(self):
        pass
