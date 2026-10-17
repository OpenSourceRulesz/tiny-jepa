"""tiny-world: procedural 16x16 tile-map generator.

A "world" is a 16x16 grid (1 = wall, 0 = empty); each tile renders as a
2x2 pixel block on the 32x32 canvas. Three styles, all with walled
borders and guaranteed connectivity of empty space:

- style 0 "rooms": horizontal wall with a corridor gap (+ random blocks)
- style 1 "maze": randomized-DFS perfect maze (connected by construction)
- style 2 "open": sparse random wall blocks, connectivity-checked

These maps are the training data for the diffusion prior
(`diffusion.WorldPrior`) and the fallback when it misbehaves.
"""

import numpy as np

GRID = 16
STYLES = ("rooms", "maze", "open")
N_STYLES = len(STYLES)

__all__ = ["GRID", "STYLES", "N_STYLES", "procedural_map", "reachable",
           "largest_empty_component", "spawn_points"]


def _empty_grid():
    m = np.zeros((GRID, GRID), dtype=np.int64)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 1
    return m


def _neighbors4(y, x):
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < GRID and 0 <= xx < GRID:
            yield yy, xx


def reachable(m, a, b):
    """BFS: can you walk from tile a to tile b through empty tiles?"""
    a, b = tuple(a), tuple(b)
    if m[a] or m[b]:
        return False
    seen = {a}
    stack = [a]
    while stack:
        y, x = stack.pop()
        if (y, x) == b:
            return True
        for nb in _neighbors4(y, x):
            if m[nb] == 0 and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return False


def largest_empty_component(m):
    """Fraction of empty tiles in the biggest connected component."""
    seen = set()
    best = 0
    total = 0
    for y in range(GRID):
        for x in range(GRID):
            if m[y, x] == 0:
                total += 1
                if (y, x) not in seen:
                    comp, stack = 0, [(y, x)]
                    seen.add((y, x))
                    while stack:
                        cy, cx = stack.pop()
                        comp += 1
                        for nb in _neighbors4(cy, cx):
                            if m[nb] == 0 and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
                    best = max(best, comp)
    return best / max(total, 1)


def _rooms(rng):
    m = _empty_grid()
    wy = int(rng.integers(6, 11))          # wall row
    gx = int(rng.integers(4, 11))          # gap start
    gw = int(rng.integers(2, 4))           # gap width
    m[wy, 1:GRID - 1] = 1
    m[wy, gx:gx + gw] = 0
    # a few random blocks, never sealing the gap
    for _ in range(int(rng.integers(0, 4))):
        y, x = int(rng.integers(2, GRID - 2)), int(rng.integers(2, GRID - 2))
        if abs(y - wy) <= 1 and gx - 1 <= x <= gx + gw:
            continue
        h, w = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        m[y:y + h, x:x + w] = 1
    m[wy, gx:gx + gw] = 0  # guarantee the corridor survived
    return m


def _maze(rng):
    # randomized DFS on odd-coordinate cells; interior 14x14
    m = np.ones((GRID, GRID), dtype=np.int64)
    start = (1, 1)
    m[start] = 0
    stack = [start]
    seen = {start}
    while stack:
        y, x = stack[-1]
        opts = []
        for dy, dx in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            yy, xx = y + dy, x + dx
            if 1 <= yy < GRID - 1 and 1 <= xx < GRID - 1 \
                    and (yy, xx) not in seen:
                opts.append((dy, dx))
        if not opts:
            stack.pop()
            continue
        dy, dx = opts[int(rng.integers(len(opts)))]
        m[y + dy // 2, x + dx // 2] = 0
        m[y + dy, x + dx] = 0
        seen.add((y + dy, x + dx))
        stack.append((y + dy, x + dx))
    # knock a few extra loops so mazes aren't all perfect trees
    for _ in range(int(rng.integers(4, 10))):
        y = int(rng.integers(1, GRID - 1, ))
        x = int(rng.integers(1, GRID - 1))
        if m[y, x] == 1:
            m[y, x] = 0
    return m


def _open(rng):
    for _ in range(30):  # retries for connectivity
        m = _empty_grid()
        for _ in range(int(rng.integers(3, 8))):
            y = int(rng.integers(1, GRID - 3))
            x = int(rng.integers(1, GRID - 3))
            h, w = int(rng.integers(1, 3)), int(rng.integers(1, 4))
            if rng.random() < 0.5:
                h, w = w, h
            m[y:y + h, x:x + w] = 1
        if largest_empty_component(m) >= 0.9:
            return m
    m = _empty_grid()  # fallback: plain room
    return m


def procedural_map(style=0, seed=None):
    """Draw a guaranteed-sane 16x16 world map. style: 0 rooms, 1 maze,
    2 open. Returns int array (1 = wall, 0 = empty)."""
    rng = np.random.default_rng(seed)
    style = int(style) % N_STYLES
    if style == 0:
        return _rooms(rng)
    if style == 1:
        return _maze(rng)
    return _open(rng)


def spawn_points(m, seed=None, min_tiles=4):
    """Two empty tiles with a walkable path between them, at least
    min_tiles apart (BFS distance). Returns (agent_tile, pellet_tile)."""
    rng = np.random.default_rng(seed)
    empty = [(y, x) for y in range(GRID) for x in range(GRID)
             if m[y, x] == 0]
    # BFS distances from a random root
    root = empty[int(rng.integers(len(empty)))]
    dist = {root: 0}
    stack = [root]
    while stack:
        y, x = stack.pop()
        for nb in _neighbors4(y, x):
            if m[nb] == 0 and nb not in dist:
                dist[nb] = dist[(y, x)] + 1
                stack.append(nb)
    far = [t for t, d in dist.items() if d >= min_tiles]
    if not far:  # degenerate map; fall back to any two distinct tiles
        far = list(dist)
    a = root
    b = far[int(rng.integers(len(far)))]
    return a, b
