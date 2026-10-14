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

**Baseline (shipped, 2026-10-14):** `tiny_jepa/planning.py` has a correct,
tested CEM optimizer (`cem_plan`) — it provably finds the optimum against a
known reward landscape. Against orby_v1's learned reward head it scores
**0.0 pellets/episode** (greedy: 8.8). Measured diagnosis: the reward head
separates eat/non-eat on real latents (0.86 vs 0.31) but collapses on imagined
latents, and imagined-score ranking carries no signal about true outcomes
(rank corr +0.05). The planner is exonerated; the value head is the suspect.
Start there.

**Win condition:** > 4 pellets/episode (100 seeds) with the position heads
disabled. This is the hardest challenge in the set and the most important
one: it's the difference between a world model and a detector with a
steering wheel.

**Stretch:** Beat the greedy 8.6. Do that and you've earned the top row of
the leaderboard permanently.
