"""tiny-spatial benchmark environments.

Two controlled variants of OrbyWorld (numpy only, 32x32):

- OcclusionWorld: a vertical wall occludes part of the canvas. Entities
  behind it are invisible in the frame. Probes script the pellet path
  directly (pellets don't move on their own).
- TwoRoomWorld: a horizontal wall with a corridor gap splits the canvas
  into two rooms + fog of war (you only see your own room).
"""

import numpy as np

from .._orby import (
    OrbyWorld, CANVAS, AGENT_R, AGENT_BRIGHT, BALL_R, BALL_BRIGHT,
    PELLET_R, PELLET_BRIGHT, _blob, _GX, _GY,
)

WALL_FILL = 0.3

__all__ = ["OcclusionWorld", "TwoRoomWorld"]


class OcclusionWorld(OrbyWorld):
    """OrbyWorld plus a visual occluder: wall rect drawn over entities."""

    WALL = (13.0, 4.0, 19.0, 28.0)  # x0, y0, x1, y1

    def render(self):
        img = super().render()
        x0, y0, x1, y1 = self.WALL
        mask = (_GX >= x0) & (_GX < x1) & (_GY >= y0) & (_GY < y1)
        img = img.copy()
        img[mask] = WALL_FILL
        return img

    def pellet_visible(self):
        x0, y0, x1, y1 = self.WALL
        x, y = self.pellet
        return not (x0 <= x <= x1 and y0 <= y <= y1)


class TwoRoomWorld(OrbyWorld):
    """Two rooms separated by a wall with a corridor gap; fog of war.

    Room 0: y < 14. Room 1: y > 18. Corridor band: 14 <= y <= 18.
    The agent sees entities in its own room (or everything from the
    corridor); the wall blocks movement except through the gap
    x in [13, 19].
    """

    WALL_Y = (14.0, 18.0)
    GAP_X = (13.0, 19.0)

    # -- room helpers -------------------------------------------------
    @classmethod
    def room_of(cls, xy):
        y = float(xy[1])
        if y < cls.WALL_Y[0]:
            return 0
        if y > cls.WALL_Y[1]:
            return 1
        return -1  # corridor / inside wall band

    def _sees(self, entity_xy):
        ar = self.room_of(self.agent)
        if ar == -1:
            return True
        er = self.room_of(entity_xy)
        return er == ar or er == -1

    # -- rendering with fog of war ------------------------------------
    def render(self):
        img = _blob(*self.agent, AGENT_R, AGENT_BRIGHT)
        for b in range(2):
            if self._sees(self.balls[b]):
                img = img + _blob(*self.balls[b], BALL_R[b], BALL_BRIGHT[b])
        if self._sees(self.pellet):
            img = img + _blob(*self.pellet, PELLET_R, PELLET_BRIGHT)
        img = np.clip(img, 0, 1)
        y0, y1 = self.WALL_Y
        gx0, gx1 = self.GAP_X
        mask = (_GY >= y0) & (_GY < y1) & ((_GX < gx0) | (_GX >= gx1))
        img = img.copy()
        img[mask] = WALL_FILL
        return img.astype(np.float32)

    # -- wall collision for the agent ---------------------------------
    def step(self, action):
        old = self.agent.copy()
        frame, reward = super().step(action)
        y0, y1 = self.WALL_Y
        gx0, gx1 = self.GAP_X
        x, y = self.agent
        if y0 <= y <= y1 and not (gx0 <= x <= gx1):
            self.agent = old
            frame = self.render()
        return frame, reward

    # -- room-constrained pellet spawns -------------------------------
    def _respawn_pellet(self, room=None):
        r = self.rng
        for _ in range(40):
            x = r.uniform(4, CANVAS - 4)
            if room == 0:
                y = r.uniform(4, 12)
            elif room == 1:
                y = r.uniform(20, 28)
            else:
                y = r.uniform(4, CANVAS - 4)
                if self.WALL_Y[0] - 1 <= y <= self.WALL_Y[1] + 1:
                    continue
            p = np.array([x, y])
            if np.linalg.norm(p - self.agent) > 6.0:
                break
        self.pellet = p.astype(np.float64)
