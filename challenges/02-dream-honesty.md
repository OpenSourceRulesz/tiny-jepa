# Challenge 02 — Dream honesty

**The quest:** Make the model's imagined futures stay honest for longer.

**Baseline:** Roll the predictor forward 8 steps from a real frame and
measure cosine similarity between the imagined latent and the true future
latent: **0.43 at 8 steps** (random: 0.15). The dreams blur — that's the
compounding error every world model fights.

**Your move:** Anything that sharpens long-horizon imagination. Ideas:
recurrent predictor, multi-step training loss, hierarchical timescales,
larger latent, better encoder. Keep it numpy if you can — elegance counts.

**How to measure:**
```python
from tiny_jepa import TinyJEPA, OrbyWorld
import numpy as np
model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
sims = []
for seed in range(20):
    env = OrbyWorld(seed=seed)
    f = env.render(); z = model.encode(f)
    actions = [4, 4, 1, 4, 3, 2, 4, 4]
    imagined = model.imagine(z, actions)
    for a in actions: env.step(a)
    true = model.encode(env.render())
    # compare imagined[-1] vs encode of true frame after each prefix...
print("8-step cosine:", ...)
```

**Win condition:** 8-step cosine > 0.43, reproducible from your PR.

**Stretch:** > 0.60 — that's the regime where MPC starts working (see the
`workflow_mpc.py` example for why it matters).
