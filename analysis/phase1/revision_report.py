"""Evidence and gate for the retained-E8 Phase 1 refinement.

This accepts only the stated finite experiment matrix and exact component
properties. It never asserts population, ImageNet or TPU latency dominance.
"""

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SECTIONS=["symbolic","attention","positions","pool_precision","propagation","convergence","purity"]
SELECTED="v2_retained_e8"
ARMS=["standard","algebraic_v1",SELECTED]


def read(path):return json.loads(path.read_text())


def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def source_issues(record,expected,label):
    saved=record.get("provenance",{}).get("source_sha256",{})
    return [f"{label}: stale or missing {name}" for name in expected
        if name not in saved or not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=saved[name]]


def matrix_issues(rows,keys,axes,label):
    observed=[tuple(r[k] for k in keys) for r in rows]
    expected=set(itertools.product(*axes))
    missing=expected-set(observed)
    return [f"{label}: {len(missing)} missing cells or duplicates"] if missing or len(observed)!=len(set(observed)) else []


def audit(out,data,costs):
    issues,coverage=[],[]
    deps=["analysis/phase1/"+n for n in ["revision_study.py","revisions.py","references.py","convergence.py","propagation.py","simulations.py","purity.py"]]
    deps += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"algebraic_vision").glob("*.py"))]
    for name,record in data.items():issues+=source_issues(record,deps,"revision_"+name)
    cost_deps=["analysis/phase1/"+n for n in ["costs.py","operation_count.py","revisions.py","references.py","purity.py"]]
    cost_deps += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"algebraic_vision").glob("*.py"))]
    issues+=source_issues(costs,cost_deps,"costs")
    a=data["attention"]
    coverage+=matrix_issues(a["records"],["distribution","seed","alpha"],
        [["gaussian","student_t3"],range(42,50),[1,2,4,8,10,16,32,64,100,128,256]],"revision stress")
    coverage+=matrix_issues(a["precision"],["n","seed","scale","dtype"],
        [[32,197,256],range(42,50),[1,3,10,30,100],["float32","bfloat16"]],"revision precision")
    coverage+=matrix_issues(a["radial"],["alpha"],[[10,32,100,256,1024]],"revision radial")
    expected_positions=set(itertools.product([14,16,32,64],["float32","bfloat16"],[(0.,0.),(1.,1.),(0.6,1.6),(1e4,1e4),(0.,1e4)]))
    observed=[(r["grid"],r["dtype"],(r["raw_x"],r["raw_y"])) for r in data["positions"]["records"]]
    if not expected_positions<=set(observed) or len(observed)!=len(set(observed)):coverage.append("revision positions: incomplete or duplicate matrix")
    coverage+=matrix_issues(data["pool_precision"]["records"],["scale","dtype"],[[1,10,100,1e3,1e4,1e6],["float32","bfloat16"]],"revision pool precision")
    for kind in ["forward","spectral"]:
        rows=data["propagation"][kind]
        coverage+=matrix_issues(rows,["seed","variant","layer"],[[42,43,44],ARMS,range(1,25)],"revision "+kind)
        if kind=="spectral" and any(len(r["singular_values"])!=64 or (r["layer"] in [12,24] and len(r.get("stack_singular_values",[]))!=64) for r in rows):
            coverage.append("revision spectra: missing singular values")
    c=data["convergence"]
    coverage+=matrix_issues(c["records"],["seed","depth","variant","lr"],[[42,43,44],[2,4],ARMS,np.geomspace(1e-4,1e-2,5)],"revision convergence")
    coverage+=matrix_issues(c["ntk"],["seed","depth","variant"],[[42,43,44],[2,4],ARMS],"revision NTK")
    if c["scope"]["steps"]<200 or any(len(r["loss_history"])<200 or len(r.get("train_accuracy_history",[]))<200 for r in c["records"]):coverage.append("revision convergence: incomplete history")
    if not all(data["symbolic"].get("checks",{}).values()) or len(data["symbolic"]["checks"])<9:coverage.append("revision symbolic checks incomplete")
    if set(data["purity"]["records"])!={"selected_complete_step","rational_gate_value_and_gradient"}:coverage.append("revision purity incomplete")
    coverage+=matrix_issues(costs["records"],["shape","tokens"],[["B/16","L/16"],[197,256]],"revision costs")
    for row in costs["records"]:
        expected={"B/16":(12,768,12),"L/16":(24,1024,16)}[row["shape"]]
        if (row["depth"],row["width"],row["heads"])!=expected or row["grid"]!={197:14,256:16}[row["tokens"]]:
            coverage.append("revision costs: mislabeled model shape")
    return issues,coverage


def compare_pool(initial):
    rows=initial["outliers"]["records"]
    ratios=[]
    for n,count,scale in itertools.product([197,256],[1,3,5],[5,10,25,50,100]):
        cell=[r for r in rows if (r["n"],r["outliers"],r["scale_sqrt_d"])==(n,count,scale)]
        new=np.mean([r["angle_degrees"] for r in cell if r["pool"]=="v2_inverse_energy"])
        old=np.mean([r["angle_degrees"] for r in cell if r["pool"]=="standard_v1_actual_ln_gap"])
        ratios.append({"n":n,"outliers":count,"scale_sqrt_d":scale,"ratio_of_means":float(new/old)})
    return ratios


def evaluate(data,costs,initial):
    attention=data["attention"]
    gradient=bool(all(r["finite"] and r["gradient_ratio"]==1 and r["q_gradient_max_difference"]==r["k_gradient_max_difference"]==0 for r in attention["records"])
                  and all(r["gradient_ratio"]==1 and r["absolute_difference"]==0 for r in attention["radial"]))
    precision=bool(all(r["finite"] and r["baseline_forward_max_difference"]==r["baseline_backward_max_difference"]==0 for r in attention["precision"])
                   and all(r["finite"] for r in data["pool_precision"]["records"]))
    positions=bool(all(r["min_offset_distance"]>0 and r["rank_determinant"]!=0 and r["maximum_phase"]<4 and r["unique_rounded_grid_matrices"]==r["grid"]**2 and r["table_orthogonality_error"]<1e-5 for r in data["positions"]["records"]))
    ratios=compare_pool(initial)
    pool=bool(max(r["ratio_of_means"] for r in ratios)<=1)
    cost=True
    for r in costs["records"]:
        budgets=r["batch_budgets"]
        cost &= [b["batch"] for b in budgets]==[1,2,3]
        for metric,keys in [("floating_work_including_selections",["arithmetic","matmul","minmax","comparisons","selects"]),("rsqrt",["rsqrt"])]:
            deltas=[sum(b["candidate"][k]-b["baseline"][k] for k in keys) for b in budgets]
            slope=deltas[1]-deltas[0]
            certificate=r["batch_affine_certificate"][metric]
            cost &= bool(deltas[0]<=0 and slope<=0 and deltas[2]==deltas[0]+2*slope
                and certificate["delta_at_batch_one"]==deltas[0] and certificate["delta_per_extra_example"]==slope
                and certificate["batch_three_delta"]==deltas[2] and certificate["all_positive_batches_nonregressing"])
        cost &= r["optimizer_parameter_delta"]==-(r["depth"]*r["width"]//2)
        for metric in ["arithmetic","matmul","rsqrt"]:
            old=budgets[0]["baseline"][metric]
            new=budgets[0]["candidate"][metric]
            reported=r["comparisons"]["retained_e8"]["static_"+metric]
            cost &= bool(new<=old and reported["candidate_minus_baseline"]==new-old)
    cost=bool(cost)
    purity=bool(all(not r["forbidden"] for r in data["purity"]["records"].values()))
    c=data["convergence"]["records"]
    mslr=[]
    for seed,depth in itertools.product([42,43,44],[2,4]):
        def largest(variant):
            return max([r["lr"] for r in c if r["seed"]==seed and r["depth"]==depth and r["variant"]==variant and r["stable"]],default=0.)
        mslr.append({"seed":seed,"depth":depth,"v1":largest("algebraic_v1"),"v2":largest(SELECTED)})
    convergence=bool(all(r["finite"] for r in c) and all(r["v2"]>=r["v1"]>0 for r in mslr))
    return {"gradient_pass":gradient,"precision_pass":precision,"position_pass":positions,"pool_pass":pool,
        "training_operation_count_pass":cost,"purity_pass":purity,"mslr_pass":convergence,
        "pool_cells":ratios,"worst_pool_corruption_ratio":max(r["ratio_of_means"] for r in ratios),"paired_mslr":mslr,
        "attention_gradient_ratio":min(r["gradient_ratio"] for r in attention["records"]),
        "attention_max_forward_difference":max(r["baseline_forward_max_difference"] for r in attention["precision"]),
        "attention_max_backward_difference":max(r["baseline_backward_max_difference"] for r in attention["precision"]),
        "tolerance_used_for_attention_nonregression":0,
        "scope":"Five requested comparisons on the declared Phase 1 corpus and exact component identities, plus purity and synthetic MSLR. No population, full-width optimization, ImageNet accuracy or TPU latency dominance theorem."}


def figures(out,data):
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    rows=data["propagation"]["forward"]
    for ax,key,title in zip(axes,["variance_out","cosine_out","spectral_gap"],["Residual variance","Mean patch cosine","PF spectral gap"]):
        for arm in ARMS:
            ys=[np.mean([r[key] for r in rows if r["variant"]==arm and r["layer"]==l]) for l in range(1,25)]
            ax.plot(range(1,25),ys,label=arm)
        ax.set(xlabel="Layer",ylabel=title);ax.axvline(12,color="gray",ls=":");ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Final revision: paired 64D forward heads, three seeds; not full-width ViT-B/L")
    for suffix in ["png","pdf"]:fig.savefig(out/"figures"/f"revision_propagation.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    runs=data["convergence"]["records"]
    for ax,depth in zip(axes,[2,4]):
        for arm in ARMS:
            rows=[r for r in runs if r["depth"]==depth and r["variant"]==arm and r["lr"]==.01]
            ys=np.mean([r["train_accuracy_history"] for r in rows],axis=0)
            ax.plot(range(len(ys)),ys,label=arm)
        ax.set(xlabel="SGD step",ylabel="Training accuracy",title=f"{depth} layers, LR=0.01",ylim=(0,1.02));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched synthetic task; objectives differ; held-out results retained separately")
    for suffix in ["png","pdf"]:fig.savefig(out/"figures"/f"revision_convergence.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def finalize(out,initial_gate,initial,validation):
    missing=["revision_"+n for n in SECTIONS if not (out/f"revision_{n}.json").is_file()]
    if not (out/"costs.json").is_file():missing.append("costs")
    if missing:
        initial_gate["missing_revision_evidence"]=missing
        return initial_gate
    data={n:read(out/f"revision_{n}.json") for n in SECTIONS}
    costs=read(out/"costs.json")
    integrity,coverage=audit(out,data,costs)
    integrity=initial_gate["evidence_integrity_issues"]+integrity
    coverage=initial_gate["coverage_issues"]+coverage
    selected=evaluate(data,costs,initial)
    checks=[selected[k] for k in ["gradient_pass","precision_pass","position_pass","pool_pass","training_operation_count_pass","purity_pass","mslr_pass"]]
    passed=bool(all(checks) and initial_gate["formal_verified"] and initial_gate["tests_verified"] and not integrity and not coverage)
    gate={"decision":"GO" if passed else "NO-GO","strict_gate_satisfied":passed,"selected_variant":SELECTED,
        "selected_candidate":selected,"initial_candidates":initial_gate["candidates"],
        "evidence_integrity_issues":integrity,"coverage_issues":coverage,
        "formal_verified":initial_gate["formal_verified"],"tests_verified":initial_gate["tests_verified"],
        "validation_current":initial_gate["validation_current"],"dominance_proven":False,
        "dominance_scope":"Phase 1 acceptance is the explicit finite matrix plus exact-real component properties and stated source-program operation convention. Universal statistical/full-model dominance is not claimed.",
        "reason":"The retained-E8 candidate passes the explicit Phase 1 comparisons, formal checks and operation convention." if passed else "At least one required comparison, current validation, source-integrity check or coverage check has not passed.",
        "phase_boundary":"Phase 1 only. No production third-arm registration, TPU kernels, multi-host smoke tests, sweeps or main runs.",
        "cost_certificates":[{"shape":r["shape"],"tokens":r["tokens"],"certificate":r["batch_affine_certificate"]} for r in costs["records"]]}
    figures(out,data)
    c=data["convergence"]["records"]
    lrrows=[]
    for depth,arm in itertools.product([2,4],ARMS):
        rows=[r for r in c if r["depth"]==depth and r["variant"]==arm and r["lr"]==.01]
        hits=[r["first_train_accuracy_95_step"] for r in rows]
        hit=f"{np.mean(hits):.2f}" if all(h is not None for h in hits) else "not all reached"
        lrrows.append(f"| {depth} | {arm} | {sum(r['stable'] for r in rows)}/3 | {hit} | {np.mean([r['heldout_accuracy'] for r in rows]):.4f} |")
    costrows=[]
    for r in costs["records"]:
        f=r["batch_affine_certificate"]["floating_work_including_selections"]
        s=r["batch_affine_certificate"]["rsqrt"]
        costrows.append(f"| {r['shape']} | {r['tokens']} | {f['delta_at_batch_one']:,} | {f['delta_per_extra_example']:,} | {s['delta_at_batch_one']:,} |")
    names=[("gradient_pass","High-scale attention gradients"),("precision_pass","FP32/BF16 attention stability and finite pool boundaries"),
        ("position_pass","Rank and grid positional injectivity"),("pool_pass","Outlier angular robustness"),
        ("training_operation_count_pass","Per-step theoretical work and roots"),("purity_pass","Algebraic training graph"),("mslr_pass","Synthetic stable learning-rate endpoint")]
    checkrows="\n".join(f"| {label} | {'PASS' if selected[key] else 'FAIL'} |" for key,label in names)
    report=f"""# Phase 1 analytical report

**Decision: {gate['decision']} for the refined `v2_retained_e8` research candidate on the declared Phase 1 checks.** The initial proposal and root-free attention candidates remain rejected. The final revision retains E8 attention, which proves an exact attention-gradient and precision match to v1; it does **not** eliminate E8 score attenuation or quadratic attention roots. All later phases remain unstarted.

This is a scoped research gate, not a theorem that one complete model dominates another on every distribution. `dominance_proven=false` in [gate.json](gate.json) explicitly records that limit. No full-width training, ImageNet accuracy or TPU latency result is implied. User authorization allowed choosing an appropriate numerical equivalence criterion; the selected attention ultimately needs **zero tolerance**.

## Selected equations and design decision

The accessible derivation is in [theory.md](../../theory.md). The frozen `standard` and `algebraic` implementations, experiments, original tests and original proofs are unchanged.

* **Attention:** keep exactly the existing normalized E8 score link, no Klein lift or column balancing. The same expression, precision policy and real-valued derivatives make its preservation and perturbation behavior identical to v1 in the same Q/K coordinates.
* **Positions:** two SO(2) planes in SO(4), with phases `c*theta_x+r*theta_y` and `c*theta_x-r*theta_y`. Learn each frequency through `w=(1/G)*(1/4+3/(4*(1+raw^2)))`. Positive bounded frequencies maintain rank two and prevent phase wrapping on the grid. Homogeneous projective recurrence removes table-renormalization roots.
* **Pool:** `LN(sum a_i*h_i/sum a_i)`, with `a_i=d/(d+||h_i||^2)`, applied to the raw residual tokens before the final LN. This replaces the Lorentz proposal with a redescending rational centroid.
* **MLP:** `f(x)=x/2*(1+c*x/(1+c*|x|))`, c=0.851. This C2 rational gate has derivative strictly between zero and one, matches rgelu's value/slope/curvature at zero, and uses no roots. Its exact analytic JVP is checked against ordinary differentiation; it is not a surrogate gradient.
* **Head:** genuine strictly proper power score a=7/8 on `rho(z/4)^8`, with three sequential roots of the row normalizer. Exact CE's forward logarithm and the non-conservative quarter-power field are rejected. Raw loss values from different proper scores are not directly comparable.

## Checks and coverage

| Requirement | Result |
| --- | --- |
{checkrows}

Evidence integrity issues: **{integrity or 'none'}**. Coverage issues: **{coverage or 'none'}**. Verification current: **{gate['validation_current']}**. Formal build/axioms verified: **{gate['formal_verified']}**. Tests and frozen baselines verified: **{gate['tests_verified']}**.

The original nine-section experiment matrix is preserved in its JSON files and [the initial iteration report](iterations_0_2.md). The final revision additionally runs both stress distributions at eight paired seeds 42–49, all 11 scales 1–256, radial counterexamples through 1024, and FP32/BF16 errors at N=32/197/256 and scales 1/3/10/30/100. [revision_attention.json](revision_attention.json) records every result: minimum paired query-gradient ratio **{selected['attention_gradient_ratio']}**, maximum forward difference **{selected['attention_max_forward_difference']}**, and maximum backward difference **{selected['attention_max_backward_difference']}**. Matching a kernel to itself is deliberate retention of a sound component, not evidence that the failed new kernels improved.

Crowding comparisons at N=197/256 and foreground counts 1/3/5 include all original balanced and unbalanced candidates. The selected attention inherits exactly v1's foreground mass and gradient SNR for identical Q/K; unchanged evidence is reused explicitly rather than counted as new independent samples. Original u/kappa/light-cone boundary cases and 50-digit references remain in [precision.json](precision.json). The selected pool is separately finite, in value and VJP, through scale 1e6 in both dtypes: [revision_pool_precision.json](revision_pool_precision.json).

Learned positions are checked at G=14/16/32/64, initialization, equal and mixed frequency extremes, and FP32/BF16 output boundaries. Every rounded grid matrix is distinct in each declared test. The exact argument holds for all finite raw parameters: positive angles give nonzero rank; each relative phase has magnitude below 4<2*pi; both phases zero force both offsets zero. Rounded rotations are not machine-exact orthogonal matrices. [Position data](revision_positions.json).

The selected inverse-energy pool is the identical previously tested `v2_inverse_energy` component. On every N/count/scale cell, its mean angular corruption is no larger than actual v1 **LN-before-GAP**, with worst ratio **{selected['worst_pool_corruption_ratio']:.6f}**. All paired samples remain in [outliers.json](outliers.json); raw GAP remains a separate diagnostic. Positive weights and fixed points are formalized. Isolated-outlier influence is O(1/||h||), derivative O(1/||h||^2), with other tokens fixed. This does not guarantee smaller error on every arbitrary token collection.

## Deep propagation, convergence and NTK

[revision_propagation.json](revision_propagation.json) records F(q), G(c), PF gaps and spectra at every layer for standard, v1 and the selected revision, at three paired seeds. Depths are 12 and 24. Forward heads are 64D over 64 tokens; exact backward singular spectra use the complete 64D state of four 16D tokens, with composed spectra and Lyapunov exponents at 12/24. The synthetic widths are explicit: these are neither full-width ViT-B/L Jacobians nor an infinite-width mean-field theorem. [Propagation figure](figures/revision_propagation.pdf).

The complete 2/4-layer, width-32 classification study uses three paired seeds, five LRs from 1e-4 to 1e-2, 200 SGD-momentum steps per run, 64 training and 64 held-out examples. Every candidate run is finite. All six paired seed/depth comparisons retain v1's stable endpoint of 0.01, so MSLR remains right-censored there. Complete loss and task-accuracy traces, held-out results, and empirical 32-output NTK spectra/conditioning are in [revision_convergence.json](revision_convergence.json).

| Layers | Arm | Stable at LR=.01 | Mean step reaching 95% train accuracy | Mean held-out accuracy at .01 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(lrrows)}

Task-accuracy convergence is measured directly, separately from each objective's loss. Head link, loss scale, gate, pool and positions change together, so these runs do not isolate one cause of any speed difference. No larger MSLR or universal accuracy improvement is asserted. [Accuracy traces](figures/revision_convergence.pdf). Earlier rgelu/ALU/Brier and exact/quarter-gradient diagnostics remain in the original studies.

## Theoretical per-step operation comparison

[costs.json](costs.json) contains forward plus complete VJP counts at B/L widths, depths and N=197/256. Matrix FLOPs use 2*M*N*K; scalar arithmetic and division count one each. Roots are counted separately. The conservative table below also charges every min/max, comparison and select as one unit, and every fixed scan iteration is included. It does not hide these operations in the FLOP saving.

Common embedding, QKVO/MLP/classifier linear products, residual operations and internal norms cancel. The changed pool's norm, head loss, all position-frequency gradients and full attention/PV backward are included. The selected position parameters decrease by D/2 per layer, so an identical elementwise SGD/Adam-style optimizer costs no more. The proof assumes the same storage/recomputation convention, no extra rematerialization, and dense shapes. CPU lowered/compiled estimates are supplied separately and also favor this selection; they are not TPU timing predictions. See [JAX's AOT documentation](https://docs.jax.dev/en/latest/aot.html) for compiler-specific cost analysis.

For each shape the source-program count is affine in batch size B>=1: pointwise counts are linear, batch reductions cost B-1 per fixed group, and no operation multiplies two independent batch dimensions. The count difference is `delta(1)+(B-1)*slope`. Both terms are nonpositive, with a separate B=3 regression crosscheck; the general affine inequality is also proved in Lean.

| Shape | N | Work difference at B=1 | Work difference per extra example | Root difference at B=1 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(costrows)}

These are **v2 minus v1**, so negative means less work. Attention roots remain; the large saving comes from replacing rgelu, removing positional-table roots, pooling once after the centroid and reducing head root depth. The static source count is an audited operation convention, not a formally verified compiler or hardware cycle model.

## Formal verification and self-correction record

All original v1 proofs remain included. New Lean modules prove rational equivalence/bounds, projective orthogonality/determinant/powers, paired SO(4) relative shifts, conditional rank, the Lorentz denominator bound, inverse-energy positivity/fixed points, learned-frequency bounds/rank, paired-phase injectivity, rational gate slope bounds and the affine cost inequality. Gate derivative identities and C2 matching are additionally checked by SymPy, 50-digit oracles and numerical JVP tests. Every local theorem/lemma is axiom-audited; only standard Lean axioms are permitted. [Formal inventory](../../formal/README.md), [validation](validation.json), [verification log](verification.log).

The initial iterations corrected the advertised margin range, arbitrary-frequency injectivity, floating-point exactness, unqualified N=1 outlier attenuation, the quarter-gradient integrability claim and the pure exact-MLE forward claim. Sharpening and column balancing did not cure all gradient/precision failures. Rational-tail attention improved several metrics but retained concrete regressions. Details and every original failed cell remain in [iterations_0_2.md](iterations_0_2.md).

Additional screens cover 75 higher-order projected candidates, 225 with column-energy priors, and 96 rational/Padé candidates, using separate selection seeds 0/1 and precision seeds 1000/1001. Exact scripts, raw ratios and limits of their exploratory provenance are archived in [the exploration directory](../../analysis/phase1/exploration/README.md). Their later optimized analytic differentials were tested independently. Padé/Newton links approach E8 closely and save roots, but their audited direct arithmetic cost can regress; compiler-specific estimates alone were not used to override that finding.

The final self-correction is conservative retention: keep the attention component that already meets the gradient/precision checks, and select independently justified changes elsewhere. This achieves the declared five-axis Phase 1 acceptance without claiming a successful root-free attention replacement, curing all middle-block attenuation, or obtaining universal model dominance.

## Reproduce

Install [requirements-phase1.txt](../../requirements-phase1.txt), then run `python -m analysis.phase1.run` followed by `python -m analysis.phase1.verify`. The full runner includes the original studies, final revision and operation counts. Use `python -m analysis.phase1.run --report-only --require-pass` to check cached, source-validated evidence; missing/stale evidence or a failed gate exits 2. Individual final studies are available through `python -m analysis.phase1.revision_study --sections ...`; cost analysis through `python -m analysis.phase1.costs`. CPU is selected explicitly, so none initializes pod training. [manifest.json](manifest.json) hashes artifacts. Reduced matrices cannot pass the coverage check.
"""
    (out/"REPORT.md").write_text(report)
    return gate
