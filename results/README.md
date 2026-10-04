# Results

Every training run writes one JSON record: the full training and model
configuration, training metrics at regular intervals (with minival top-1
where evaluated), final top-1, NLL and calibration error on minival and on
the 50,000-image validation set, throughput, device count, JAX version and
the Git commit that produced it.

| Directory | Contents | Defined by |
| --- | --- | --- |
| `small_scale/round1` ... `round6` | ViT-S/8 on 64 px ImageNet, 4 epochs: component screen, learning-rate and warmup searches, redesigns (`<config>_s<seed>.json`) | `experiments/small_scale.py` |
| `smoke/` | ViT-B/16 smoke runs (parameter counts, throughput) | |
| `pilot/` | ViT-S/4 on 64 px ImageNet, 10 epochs, seeds 42-43 | `experiments/pilot.py` |
| `sweep/` | ViT-B/16 at 224 px, 600M patch tokens, learning-rate grid, seeds 42-44; `candidates.json` is the grid and `selection.json` the chosen learning rates | `experiments/sweep.py`, `analysis/select_lr.py` |
| `main/` | ViT-B/16 at 224 px, 2.5B patch tokens, seeds 42-44, and the RoPE-2D control; `summary.json` holds the paired comparison | `experiments/main.py`, `analysis/main_report.py` |
| `analysis/` | symbolic checks, finite precision, signal propagation, stress test, purity audit, attention statistics of trained models | `analysis/*.py`, `scripts/audit_purity.py` |
| `kernels/` | attention kernel benchmark on one TPU v4 chip | `scripts/bench_attention.py` |

The experiment scripts regenerate the configuration of every recorded run.
During the study the stages were numbered, so the `name` and `output` fields
inside the records keep their original paths (`p3` = small-scale trials,
`p5` = smoke, `p7` = pilot, `p8` = sweep, `p9` = main runs); the measured
values are unchanged.
