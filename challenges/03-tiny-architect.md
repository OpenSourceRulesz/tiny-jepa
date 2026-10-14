# Challenge 03 — Tiny architect

**The quest:** Shrink the model without lobotomizing it.

**Baseline:** ~200k params, 32-dim latent, hunts 8.6 pellets/episode.

**Your move:** Train a smaller architecture that still hunts. Ideas: 16-dim
latent, fewer layers, shared weights, quantized numpy. How small can the
world model get before it stops understanding its world?

**Win condition:** < 50k params AND > 5 pellets/episode (100 seeds).
Report both numbers in your PR.

**Stretch:** < 10k params. At that point you're flirting with the
Kolmogorov complexity of the game itself — genuinely interesting.
