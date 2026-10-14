# Challenge 05 — Blindfold

**The quest:** Hunt pellets when you can't see the whole world.

**Setup:** Mask 50% of the input frame's 8×8 patches (at random, each step)
before encoding — the V-JEPA masking idea applied at test time. The shipped
checkpoint never trained for this.

**Your move:** Make it work anyway, or train a model that does. Ideas:
fine-tune the encoder with masked inputs, add a latent "filling-in"
mechanism, or show that the predictor's temporal context compensates.

**Win condition:** > 4 pellets/episode (100 seeds) with 50% of patches
masked. Report the masking scheme in your PR.

**Why it matters:** Real sensors are occluded, dirty, and partial. A world
model that collapses the moment it can't see everything isn't a world
model — it's a camera calibrator.
