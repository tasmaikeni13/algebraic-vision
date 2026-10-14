# Results

[`phase1/REPORT.md`](phase1/REPORT.md) contains the v2 Phase 1 investigation,
its reproducible numerical evidence and figures, formal/test validation, and
the final candidate's scoped gate decision. Rejected iterations remain in
[`phase1/iterations_0_2.md`](phase1/iterations_0_2.md). Later phases are unstarted.

Every training run writes one JSON record: the full training and model
configuration, training metrics at regular intervals (with minival top-1
where evaluated), final top-1, NLL and calibration error on minival and on
the 50,000-image validation set, throughput, device count, JAX version and
the Git commit that produced it.

| Directory | Contents | Defined by |
| --- | --- | --- |
| `smoke/` | ViT smoke runs (parameter counts, throughput) | `algebraic_vision/train.py` |
| `sweep/` | ImageNet-1k learning-rate sweep records | `experiments/sweep.py`, `analysis/select_lr.py` |
| `main/` | Full ImageNet-1k training runs | `experiments/main.py`, `analysis/main_report.py` |
| `analysis/` | Symbolic checks, finite precision, signal propagation, stress test, purity audit | `analysis/*.py`, `scripts/audit_purity.py` |
| `kernels/` | Attention kernel benchmark on TPU v4 | `scripts/bench_attention.py` |
