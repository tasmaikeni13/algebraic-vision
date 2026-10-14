# Results

[`phase1/REPORT.md`](phase1/REPORT.md) contains the v2 Phase 1 investigation,
its reproducible numerical evidence and figures, formal/test validation, and
the final candidate's scoped gate decision. Rejected iterations remain in
[`phase1/iterations_0_2.md`](phase1/iterations_0_2.md). Production phases 2–5
are documented in [PHASES.md](../PHASES.md).

Every training run writes one JSON record: the full training and model
configuration, training metrics at regular intervals (with minival top-1
where evaluated), final top-1, NLL and calibration error on minival and on
the 50,000-image validation set, throughput, device count, JAX version and
the Git commit that produced it.

| Directory | Contents | Defined by |
| --- | --- | --- |
| `smoke/` | Legacy checks and the complete three-arm B/L pod smoke matrix | `algebraic_vision/production.py`, `scripts/smoke_production.py` |
| `sweep/` | Operator-triggered learning-rate/weight-decay sweeps | `experiments/production.py`, `analysis/production_select.py` |
| `main/` | Operator-triggered main runs and paired reports | `experiments/production.py`, `analysis/production_report.py` |
| `configs/` | Sweep matrix, main template and executed smoke configs | `experiments/production.py` |
| `verification/` | CPU/formal proofs, full TPU HLO/kernel checks and completion audit | `analysis/production/verify.py`, `scripts/audit_production.py` |
| `analysis/` | Symbolic checks, finite precision, signal propagation, stress test, purity audit | `analysis/*.py`, `scripts/audit_purity.py` |
| `kernels/` | Attention kernel benchmark on TPU v4 | `scripts/bench_attention.py` |

# Production phases 2–5

The current [verification report](verification/REPORT.md) and
[completion audit](verification/completion.json) cover all eighteen full-pod
synthetic smoke runs. `smoke/report/` contains paired statistics and PDF/PNG
figures; `configs/` contains the 108-job sweep and blocked main template.
Full sweeps and main training remain for the operator. These synthetic metrics
are pipeline verification, not ImageNet accuracy evidence.
