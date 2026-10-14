# Production verification record

Phase 2 model registration, B/L scale schemas, matched strong augmentation,
whole-update purity auditing and the Phase 3 kernels are implemented.
The 108-job sweep and 18-job main template are prepared. Full sweeps and main
training have not been launched. Distributed training/checkpoint/evaluation
smoke verification is the remaining execution gate.

Evidence recorded so far:

* `architectures.json`: all six full-width model shapes and parameter counts.
* `pod_check.log` and worker logs: four hosts, sixteen TPU v4 devices and a
  successful collective (sum 40).
* `purity_tpu.json`: actual TPU compilation of all four B/L algebraic updates,
  including optimizer and nested Pallas programs, with no forbidden operations.
  Compiled HLO rsqrt sites: B/16 v1=334, v2=226; L/16 v1=826, v2=490.
* `kernels_tpu.json`: 20 full-pod, production-load FP32/BF16 kernel comparisons
  with forward and Q/K/V/margin gradient errors, timing and purity telemetry.
* `tests.log`: full CPU suite, 112 tests passed.
* Lean: `lake build` succeeded (8,891 build jobs); Phase 1 proofs unchanged.

The accepted Phase 1 candidate retains E8 attention. The separately verified
root-free rational proposal remains an ablation. FP32 kernel checks explicitly
request float32 matmul precision; BF16 checks use the production precision and
record both absolute errors and complete relative error norms. The selected
E8 kernels are about 7–8% slower than tiled FlashAttention in BF16, and about
27% slower than the legacy whole-row softmax kernel. Both comparisons are
kept visible; this is not a claim of beating the fastest standard kernel.

These initial records precede the final source snapshot and smoke run. Final
completion requires current-source evidence, successful bounded training at
both scales for all arms, checkpoint resume, selection/report serialization,
and an explicit requirement-by-requirement audit.
