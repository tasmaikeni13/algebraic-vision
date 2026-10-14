# Algebraic Vision

Phase 1 v2 research is available in [`analysis/phase1/`](analysis/phase1/).
The [analytical report](results/phase1/REPORT.md) records the final candidate,
its scoped comparison gate, reproducible simulations and Lean proofs.
[theory.md](theory.md) explains the core v1/v2 equations and design decisions.
The existing training models below remain frozen; no production v2 arm or
later-phase pipeline has been added.

A Vision Transformer whose forward pass, backward pass and training loss use
only arithmetic, square roots and comparisons: no exponential, logarithm,
error function, sine or cosine. It is compared with a standard ViT on
ImageNet-1k under matched data, architecture, optimizer and tuning effort.

## The two models

| Component | Standard ViT | Algebraic ViT |
| --- | --- | --- |
| Attention | softmax of `q.k / sqrt(d)` | `E_8(s) / sum_k E_8(s_k)` with `E_n(s) = (s/n + sqrt(1 + s^2/n^2))^n` |
| MLP gate | GELU | radical logistic `x / (1 + E_8(-1.702 x))` |
| Positions | learned absolute embeddings | learned absolute embeddings and learned 2-D Cayley rotations |
| Loss | cross-entropy | power score `a = 63/64` on the link `E_64(z) / sum_k E_64(z_k)` |

Everything else is shared: patch embedding, pre-norm blocks with LayerNorm,
widths and depth, global average pooling, label smoothing 0.1, AdamW with a
warmup-cosine schedule, data and augmentation.

`E_n` agrees with `e^s` to second order but grows polynomially. For positive
scores the ratio of two attention weights is at most `(s_1/s_2)^n` however
large the scores become, so attention cannot collapse to one-hot weights as
query-key norms grow, and no running maximum is needed. The Cayley rotation
`R(w) = [[1-w^2, -2w], [2w, 1-w^2]] / (1+w^2)` turns each feature pair with
the patch column and row, so attention logits depend only on the 2-D offset
between patches. The power score is strictly proper and tends to
cross-entropy as `a -> 1`; its fractional powers are nested square roots.

## Verification

* [`analysis/symbolic_checks.py`](analysis/symbolic_checks.py): every
  identity, derivative, bound and limit used in Algebraic Vision, verified
  with sympy and 50-digit mpmath.
* [`formal/`](formal/): Lean 4 proofs over the reals (ratio bound and
  attention floor, Jacobian bounds, tiled accumulation, gate properties,
  relative positions of the Cayley rotations, strict propriety of the power
  score).
* [`scripts/audit_purity.py`](scripts/audit_purity.py): compiles the
  training step for TPU and scans every instruction; the algebraic graph
  contains no transcendental operation.
* [`analysis/precision.py`](analysis/precision.py),
  [`analysis/stress.py`](analysis/stress.py),
  [`analysis/propagation.py`](analysis/propagation.py): finite precision,
  attention under growing score scale, signal propagation at
  initialisation.
* [`analysis/fairness_check.py`](analysis/fairness_check.py): validates that
  paired runs differ only in arm-defining fields and selected learning rates.
* [`tests/`](tests/): unit tests on CPU.

## Layout

| Path | Contents |
| --- | --- |
| `algebraic_vision/` | primitives, attention, Pallas TPU kernels, model, losses, input pipeline, training loop |
| `experiments/` | experiment scripts for main runs and learning-rate sweeps |
| `analysis/` | mathematical and numerical analysis, learning-rate selection, reports, fairness checks |
| `scripts/` | ImageNet preparation, TPU launchers, purity audit, kernel benchmark |
| `formal/` | Lean 4 proofs |
| `results/` | experimental records and verification outputs ([`results/README.md`](results/README.md)) |
| `tests/` | unit tests |
| `AGENTS.md` | repository instructions for paired experiments, validation and shared hardware |

## Running Experiments

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # TPU hosts: requirements-tpu.txt
export PYTHONPATH=$PWD
python -m pytest                      # unit tests on CPU
python analysis/symbolic_checks.py    # symbolic checks
(cd formal && lake exe cache get && lake build)   # Lean proofs
```

### ImageNet Preparation

ImageNet-1k as parquet shards of JPEG bytes (64 px and 256 px copies) is
converted once on every host:

```bash
python scripts/prepare_imagenet.py --source DIR_64PX  --out /dev/shm/data/in64  --layout pixels
python scripts/prepare_imagenet.py --source DIR_256PX --out /dev/shm/data/in256 --layout jpeg
```

### Training

Single-chip runs (e.g. on host):

```bash
python -m algebraic_vision.train arm=algebraic \
    'model={"patch_size":16,"width":768,"heads":12,"mlp_dim":3072}' \
    total_images=384350100 lr=2e-3 warmup_steps=5000 output=example.json
```

Full-pod training runs (e.g. TPU v4 slice):

```bash
# 1. Learning-rate sweep on ImageNet-1k
python experiments/sweep.py > jobs.json && python scripts/pod_queue.py jobs.json
python -m analysis.select_lr

# 2. Standard main runs (300 epochs)
python experiments/main.py 5e-4 2e-3 --epochs 300 > jobs.json && python scripts/pod_queue.py jobs.json
python -m analysis.main_report
python -m analysis.fairness_check results/sweep results/main
```

## Citation

```bibtex
@misc{keni2026algebraicvision,
  title  = {Algebraic Vision: a Vision Transformer Without Transcendental Functions},
  author = {Keni, Tasmai},
  year   = {2026},
  url    = {https://github.com/tasmaikeni13/algebraic-vision}
}
```
