# Production verification record

**Phases 2–5 complete within the requested execution boundary.** The completed
Phase 1 selection is implemented at B/16 and L/16. Full sweeps and main runs
have not been launched; their exact operator commands are in [PHASES.md](../../PHASES.md).

[completion.json](completion.json) records all 46 requirement checks and their
authoritative evidence. Every check passed against the current training source.

* All **18** three-step synthetic pod cases passed: three arms × seeds 42/43/44
  × B/16 and L/16, on four hosts and sixteen TPU v4 chips. B uses batch 1024;
  L uses batch 512, rematerialization, RandAugment, Mixup/CutMix and stochastic depth.
* The first B/16 standard case saved a checkpoint after step 1, then resumed
  on all hosts to step 3. Checkpoints serialize parameters, optimizer and history.
* All **12 algebraic** runs passed the optimized TPU HLO and recursive Pallas
  purity audit, including model, loss, VJP, clipping and AdamW. A separate typed
  audit confirms an exhaustive arithmetic/structural allowlist for real-valued
  operations; sign/remainder instructions are integer indexing.
* Full CPU suite: **113 tests passed**, including exact frozen-baseline forward
  and VJP equivalence, mixed-target heads, checkpoint replay and a minival-only
  selection test that deliberately gives another candidate better validation scores.
* Lean build succeeded, all **88** current local theorems/lemmas were axiom-audited,
  and there are no sorry admissions. Only standard Lean axioms were disclosed.
* All **20** full-pod B/L FP32/BF16 kernel comparisons passed forward and analytic
  Q/K/V/mass/margin VJP checks. FP32 explicitly requests float32 matmul precision.
* The 108-job sweep and 18-job main template pass fairness validation. Synthetic
  selection and paired reporting completed, including 95% intervals, one-sided
  paired tests, per-block diagnostics and visually checked PDF/PNG figures.

## Root counts and timing

Compiled whole-update HLO rsqrt instruction sites are **334 → 226** for B/16
and **826 → 490** for L/16. These are static instruction sites; nested Pallas
program counts and Phase 1 scalar operation budgets are recorded separately.

The selected E8 kernel is **8.3%** slower than tiled standard FlashAttention
at the B production BF16 load, and **6.6%** slower at L. It is **28.0%** and
**26.3%** slower than the legacy whole-row softmax kernel, respectively.
Both comparisons remain in [kernels_tpu.json](kernels_tpu.json).

Short smoke throughput excludes the first timed invocation. Memory is the
compiled argument + temporary + output − alias estimate per logical device:

| Scale | Arm | Parameters | Mean smoke images/s | Memory GiB |
| --- | --- | ---: | ---: | ---: |
| vit-b16 | standard | 86,566,120 | 8,519 | 8.39 |
| vit-b16 | algebraic | 86,575,336 | 7,457 | 7.72 |
| vit-b16 | algebraic_v2 | 86,570,728 | 6,431 | 8.58 |
| vit-l16 | standard | 304,324,584 | 1,703 | 5.16 |
| vit-l16 | algebraic | 304,349,160 | 1,662 | 5.28 |
| vit-l16 | algebraic_v2 | 304,336,872 | 1,487 | 5.37 |

These short runs verify execution and resource fit. They are not ImageNet
accuracy results or long-run throughput estimates. Fewer roots do not imply
faster whole-model training; the measured v2 smoke throughput is lower here.

## Selection and remaining operator work

V2 follows Phase 1's **retained-E8** design: bounded rank-2 projective positions,
pre-norm inverse-energy centroid, rational gate and the genuine 7/8 power score
on rho(z/4)^8. It still has quadratic attention roots and E8 score attenuation.
The root-free Klein/Cayley proposal is separately verified as a rejected
ablation; it is not used in the selected model.

The operator must prepare ImageNet on each host, execute the budgeted sweeps,
select real minival hyperparameters, then generate and execute the two main
matrices. Synthetic results cannot select production hyperparameters or unlock
the main template. The post-run pipelines are prepared and tested; full sweeps,
300-epoch main training and their real accuracy conclusions remain deliberately
unexecuted under the prompt's stop gate.
