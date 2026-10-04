# Algebraic Vision

A Vision Transformer whose forward pass, backward pass and training loss use
only arithmetic, square roots and comparisons: no exponential, logarithm,
error function, sine or cosine. It is compared with a standard ViT on
ImageNet-1k under matched data, parameters, token budget, optimizer and
tuning effort, and it is more accurate.

Paper: [`paper/main.pdf`](paper/main.pdf) (source in [`paper/`](paper/)).

## Results

ImageNet-1k validation top-1 (50,000 images). Seeds are paired: both models
see the same images, crops and flips in the same order and start from the
same values of every shared tensor.

Main comparison, ViT-B/16 at 224 px, 2.5B patch tokens (about 10 epochs),
seeds 42-44, each model at the learning rate it selected on a held-out split
of the training set:

| | Standard ViT | Algebraic ViT |
| --- | --- | --- |
| Top-1, seeds 42 / 43 / 44 | 58.35 / 58.28 / 58.34 | 68.31 / 67.74 / 68.51 |
| Top-1, mean | 58.32 | **68.19** (+9.86; 95% CI +8.97 to +10.75; p = 2.2e-4) |
| NLL | 1.871 | 1.410 |
| Calibration error (ECE) | 0.072 | 0.068 |
| Peak learning rate | 5e-4 | 2e-3 |
| Training throughput, 16 TPU v4 chips | 11,895 img/s | 10,311 img/s |

A standard ViT with sine/cosine 2-D rotary positions (RoPE-2D), trained
with the standard model's hyperparameters, reaches 65.87: relative
positions explain most of the gap. The algebraic model, whose Cayley
rotations are the algebraic counterpart of these rotary positions, still
leads it on every seed (+2.31; 95% CI +1.25 to +3.37; p = 0.006).

The four stages of the study:

| Stage | Setting | Standard | Algebraic |
| --- | --- | --- | --- |
| Small-scale trials | ViT-S/8, 64 px, 4 epochs, each model at its best learning rate, 4 seeds | 28.64 | 36.49 (+7.85, p = 2.7e-5) |
| Pilot | ViT-S/4, 64 px, 10 epochs (3.25B patch tokens), 2 seeds | 48.88 | 59.93 (+11.05) |
| Sweep | ViT-B/16, 224 px, 600M patch tokens, best learning rate, 3 seeds | 26.08 | 39.82 (+13.74) |
| Main runs | ViT-B/16, 224 px, 2.5B patch tokens, 3 seeds | 58.32 | 68.19 (+9.86, p = 2.2e-4) |

The algebraic model trains stably at learning rates where the standard ViT
degrades or collapses (the standard model's best learning rate is 5e-4;
the algebraic model's is 2e-3), and at 2.5B tokens it is still ahead on
every seed.

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
  identity, derivative, bound and limit used in the paper, with sympy and
  50-digit mpmath.
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
* [`analysis/fairness_check.py`](analysis/fairness_check.py): paired runs
  differ only in the model-defining fields and each model's selected
  learning rate.
* [`tests/`](tests/): unit tests (CPU).

## Layout

| Path | Contents |
| --- | --- |
| `algebraic_vision/` | primitives, attention, Pallas TPU kernels, model, losses, input pipeline, training loop |
| `experiments/` | the configuration of every training run, by stage |
| `analysis/` | mathematical and numerical analysis, learning-rate selection, reports, figures, the paper's numbers |
| `scripts/` | ImageNet preparation, TPU launchers, purity audit, kernel benchmark |
| `formal/` | Lean 4 proofs |
| `results/` | the JSON record of every run and analysis ([`results/README.md`](results/README.md)) |
| `paper/` | LaTeX source, figures and generated numbers |
| `tests/` | unit tests |

## Reproducing

Commands run from the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # TPU hosts: requirements-tpu.txt
export PYTHONPATH=$PWD
python -m pytest                      # unit tests on CPU
python analysis/symbolic_checks.py    # symbolic checks
(cd formal && lake exe cache get && lake build)   # Lean proofs
```

ImageNet-1k as parquet shards of JPEG bytes (64 px and 256 px copies) is
converted once on every host:

```bash
python scripts/prepare_imagenet.py --source DIR_64PX  --out /dev/shm/data/in64  --layout pixels
python scripts/prepare_imagenet.py --source DIR_256PX --out /dev/shm/data/in256 --layout jpeg
```

One run on the chips of one host (here the small-scale algebraic model):

```bash
python -m algebraic_vision.train arm=algebraic \
    'model={"patch_size":8,"width":384,"heads":6,"mlp_dim":1536}' \
    total_images=5084668 lr=1.5e-3 warmup_steps=2500 output=example.json
```

The experiments ran on a TPU v4-32 slice (four hosts; the workers are
reachable as `w1`, `w2`, `w3` from host 0). Single-chip jobs go through
`scripts/grid.py`, whole-slice jobs through `scripts/pod_queue.py`:

```bash
python experiments/small_scale.py round1 > jobs.json && python scripts/grid.py jobs.json
python experiments/pilot.py 5e-4 1.5e-3 > jobs.json && python scripts/pod_queue.py jobs.json
python experiments/sweep.py > jobs.json && python scripts/pod_queue.py jobs.json
python analysis/select_lr.py                     # results/sweep/selection.json
python experiments/main.py 5e-4 2e-3 > jobs.json && python scripts/pod_queue.py jobs.json
python analysis/main_report.py
python analysis/fairness_check.py results/sweep results/pilot results/main
python analysis/figures.py && python analysis/paper_numbers.py
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

The language-model counterpart of this study is
[algebraic-transformer](https://github.com/tasmaikeni13/algebraic-transformer).
