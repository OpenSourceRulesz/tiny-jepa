# Challenge 07 — ONNX sprint

**The quest:** Fastest single-step inference, fairly measured.

**Setup:** Export with `tiny_jepa.export_onnx.export`, run with
onnxruntime. Task: one `encode` + one `predictor` step, batch size 1,
float32, CPU.

**Win condition:** Lowest median latency over 1000 runs on your machine —
but you must report the machine (CPU model, RAM, OS) alongside the number.
The leaderboard tracks (latency, machine) pairs, not just latency.

**Why it matters:** World models only matter if they can think faster than
the world moves. This challenge keeps the project honest about deployment.

**Stretch:** Quantize to int8 via ONNX and show the pellet-hunting score
doesn't collapse. Speed *and* accuracy — that's the game.
