# Archived exploratory screens

These are the exact temporary scripts executed during self-correction, with
their original output paths under `/tmp` retained. Run from the repository root:

```bash
.venv/bin/python -m analysis.phase1.exploration.projected
.venv/bin/python -m analysis.phase1.exploration.column_energy
.venv/bin/python -m analysis.phase1.exploration.rational
```

The corresponding archived outputs are in
[`results/phase1/exploration/`](../../../results/phase1/exploration/).
Its manifest hashes both scripts and outputs. These screens use selection
seeds 0/1 and precision seeds 1000/1001, ordinary AD, and small diagnostic
grids. They are exploratory telemetry, **not** the final gate study,
and do not have the complete version/source provenance of the final runner.

The grids contain 75 projected-kernel candidates, 225 projected kernels with
column-energy factors, and 96 rational-tail/Padé candidates. Raw ratios and
every rejected configuration remain available. Optimized exact custom
derivatives in `revisions.py` were validated separately against ordinary AD;
these archived outputs are not misrepresented as executions of those later
implementations. The selected candidate retains E8 attention, so no numerical
equivalence tolerance from these screens is used to pass its attention gate.
