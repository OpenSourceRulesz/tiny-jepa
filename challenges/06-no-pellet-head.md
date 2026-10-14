# Challenge 06 — No pellet head (LeCun's challenge)

**The quest:** Hunt using *only* the JEPA core — predictor + reward head.
No position heads allowed.

**Context:** The shipped autopilot cheats a little: it reads the pellet's
position from a supervised auxiliary head, then steers with known
kinematics. That's perception + control, not planning. LeCun's "Mode-2"
vision is: imagine futures in latent space, score them with a learned
cost, act. Nobody has made that loop work well on this toy — the reward
head is the weak link (see `examples/workflow_mpc.py`).

**Your move:** Close the loop. Better reward shaping, CEM planning over
imagined trajectories, a value head, whatever it takes — but the pellet
position head stays off.

**Win condition:** > 4 pellets/episode (100 seeds) with the position heads
disabled. This is the hardest challenge in the set and the most important
one: it's the difference between a world model and a detector with a
steering wheel.

**Stretch:** Beat the greedy 8.6. Do that and you've earned the top row of
the leaderboard permanently.
