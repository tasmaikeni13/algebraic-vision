# Production phases 2–5

`algebraic_v2` implements the **selected** Phase 1 candidate from
[the analytical report](results/phase1/REPORT.md): retained E8 attention,
bounded rank-2 projective Bi-Cayley positions, pre-LayerNorm inverse-energy
centroid, rational gate, and the genuine 7/8 power score on `rho(z/4)^8`.
The original standard/v1 equations and initialization stay intact. Tests
compare their forward values and all parameter VJPs against the original
Git source, including a nonzero classification head.

The original proposal's root-free attention was rejected in Phase 1. It is
available separately in `algebraic_vision/rational_kernel.py`, with Klein
preprocessing outside Pallas and zero roots in its inner forward/VJP kernels.
It is an ablation, not the selected v2. The selected model still has quadratic
E8 attention roots; reductions come from its gate, positions, pooling and head.
This is the explicit consequence of the completed Phase 1 self-correction.

`algebraic_vision.production` trains all three arms on one data-parallel mesh:
four hosts, sixteen TPU v4 chips (32 TensorCores), bfloat16 matrix products,
float32 parameters, accumulation and residuals. B/16 uses 12×768, 12 heads,
MLP 3072 and global batch 1024; L/16 uses 24×1024, 16 heads, MLP 4096 and
global batch 512, with rematerialization. Both use 196 patches at 224 pixels.
L/16 applies a matched RandAugment policy, Beta Mixup/CutMix and stochastic
depth 0.2. Every arm for a seed shares data order, crop/flip/mixing streams,
shared parameter initialization and token budgets. Sweep budgets are 30
epochs; main budgets are 300, excluding the 10,000 minival images.
Pallas grids mark their batch/head axes parallel, enabling execution on both
TensorCores of each v4 JAX device as described in the
[JAX Megacore guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).

AdamW bias-correction constants and warmup/cosine learning rates are computed
on the host and supplied as scalar arguments. The compiled algebraic update
includes model, loss, VJP, clipping and optimizer, with no transcendental or
square-root opcode. The audit checks optimized TPU HLO and recursively checks
the nested Pallas programs, since custom calls hide their bodies in HLO.
Instruction-site counts and scalar operation counts are distinct; Phase 1
contains the latter. Evaluation NLL/entropy uses logarithms separately.

Fused whole-row algebraic Pallas attention keeps the short ViT score matrix
in VMEM, never in HBM. Production standard runs use an online tiled
FlashAttention forward with 128×128 score tiles and the exact recomputing
backward for short sequences. The legacy standard backend remains available;
the benchmark records both comparisons. Kernel
verification checks Q/K/V and denominator/margin gradients in float32 and
bfloat16 at the B/L production loads on all chips.
FP32 verification explicitly requests float32 matmul precision, since
[JAX's TPU default may use reduced precision](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

Checkpoints atomically store parameters, optimizer moments, step and history
on every host, with configuration/code hashes. Resume reconstructs the same
data and regularization stream. Completed sweep checkpoints are removed;
main checkpoints persist for later evaluation. Use durable host storage for
main runs. Smoke checkpoints use `/dev/shm/algebraic-vision-smoke`.

## Verification commands

```bash
source .venv/bin/activate
python -m pip install -r requirements-tpu.txt
JAX_PLATFORMS=cpu python -m pytest -q
(cd formal && lake build)
python -m experiments.production smoke --out results/configs/smoke.json
python -m scripts.smoke_production
python -m analysis.production_fairness results/configs/smoke.json --results results
python -m analysis.production_report results/configs/smoke.json --output results/smoke/report
JAX_PLATFORMS=cpu python -m analysis.production.verify
```

Smoke jobs are synthetic, use the full B/L batch sizes, run three steps and
evaluate/checkpoint/serialize diagnostics. Their accuracies are pipeline
checks only. Pod runtime packages must be installed in the same `.venv` on
`w1`, `w2`, `w3`; the launcher syncs code/configs, not Python environments.
It checks device occupancy on every host before launch and waits for an
existing workload to release the pod. The smoke driver includes a one-step
checkpoint followed by resume to the configured three steps.

## Operator commands: Phase 4

Prepare the identical ImageNet directory on each host first. The existing
`scripts/prepare_imagenet.py` supports JPEG/parquet and decoded array layouts:

```bash
python scripts/prepare_imagenet.py --source /path/to/imagenet256 --out /dev/shm/data/in256 --layout jpeg
```

The generated sweep has 108 jobs: two scales × three seeds × three arms ×
three independent LR candidates × two weight decays. Selection requires every
result and uses only the paired minival Top-1 mean. Validation is not evaluated
during the sweep. In a tie, lower LR then lower weight decay wins.

```bash
python -m experiments.production sweep --out results/configs/sweep.json
python -m analysis.production_fairness results/configs/sweep.json
python -m scripts.pod results/configs/sweep.json --execute-long-runs --timeout 172800
python -m analysis.production_fairness results/configs/sweep.json --results results
python -m analysis.production_select results/configs/sweep.json --output results/selections.json
```

## Operator commands: Phase 5

The selections file generates both nine-run main matrices. The queue skips
completed matching outputs and resumes interrupted runs from each host's
checkpoint. Each main run evaluates validation Top-1/NLL/ECE and records
per-block attention entropy and score magnitude.

```bash
python -m experiments.production main --selections results/selections.json --out results/configs/main.json
python -m scripts.pod results/configs/main.json --execute-long-runs --timeout 604800
python -m analysis.production_fairness results/configs/main.json --results results
python -m analysis.production_report results/configs/main.json --output results/main/report
```

To run Stage 5A or 5B separately, add `--scales vit-b16` or
`--scales vit-l16` when generating the main matrix. Reports include all three
pairwise comparisons for each scale, per-seed differences, 95% Student-t
intervals, one-sided paired tests, and standalone PDF/PNG metric and attention
figures. Synthetic selections cannot generate production main runs.

**Execution boundary:** preparation and bounded smoke tests are authorized.
Full sweeps and main runs are left for the human operator, as `prompt.md`
requires; the launcher enforces the explicit `--execute-long-runs` flag.
