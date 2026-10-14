"""Build an evidence-linked Phase 1 report; never infer GO from test success."""

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SECTIONS=["symbolic","stress","crowding","outliers","precision","positions","propagation","convergence","purity"]
LABELS={"standard":"Standard","algebraic_v1":"v1","v2_proposed":"v2 proposed","v2_sharp":"v2 sharp",
        "v2_balanced":"v2 balanced","v2_rational_tail":"v2 rational tail"}


def mean(rows,key):
    return float(np.mean([r[key] for r in rows]))


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def dependencies(section):
    p="analysis/phase1/"
    deps={"symbolic":["symbolic.py"],"stress":["simulations.py","references.py"],
          "crowding":["simulations.py","references.py"],"outliers":["simulations.py","references.py"],
          "precision":["simulations.py","references.py"],"positions":["simulations.py","references.py"],
          "propagation":["propagation.py","simulations.py","references.py"],
          "convergence":["convergence.py","propagation.py","simulations.py","references.py"],
          "purity":["purity.py","references.py","convergence.py","propagation.py","simulations.py"]}
    return [p+a for a in deps[section]]+["algebraic_vision/primitives.py","algebraic_vision/attention.py","algebraic_vision/losses.py"]


def evidence_integrity(data):
    issues=[]
    for section,record in data.items():
        saved=record.get("provenance",{}).get("source_sha256",{})
        for name in dependencies(section):
            path=ROOT/name
            if name not in saved or not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=saved[name]:
                issues.append(f"{section}: stale or missing source {name}")
    return issues


def validation_is_current(validation):
    saved=validation.get("verification_source_sha256",{})
    return bool(saved and validation.get("verification_inputs_unchanged") and all(
        (ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        for name,digest in saved.items()))


def theoretical_work():
    """Shape-derived model costs, separating shared matmuls and elementwise work.

    Root counts are scalar elements, not vector HLO instructions. Position
    tables are shared across the batch. Optimizer/data/remat costs are excluded.
    """
    rows=[]
    for name,depth,width,heads,hidden in [("B/16",12,768,12,3072),("L/16",24,1024,16,4096)]:
        for n,grid in [(197,14),(256,16)]:
            batch=1
            shared=depth*batch*(24*n*width*width+4*n*n*width)
            norms=depth*2*batch*n
            mlp=depth*batch*n*hidden
            positions=depth*grid*width
            v1_attention=depth*batch*heads*n*n
            v2_attention=depth*batch*heads*2*n
            rows.append({"shape":name,"depth":depth,"width":width,"heads":heads,"mlp_dim":hidden,"tokens":n,"grid":grid,"batch":batch,
                         "shared_forward_block_matmul_flops":shared,
                         "shared_forward_backward_block_matmul_flops":3*shared,
                         "v1_model_loss_forward_root_elements":norms+mlp+positions+v1_attention+batch*n+batch*(1000+6),
                         "v2_projected_model_loss_forward_root_elements":norms+mlp+v2_attention+batch*(n+2)+batch*(1000+3),
                         "v2_tail_energy_model_loss_forward_root_elements":norms+mlp+batch+batch*(1000+3),
                         "v1_attention_forward_backward_root_elements":2*v1_attention,
                         "v2_projected_attention_forward_backward_root_elements":v2_attention,
                         "v2_tail_attention_forward_backward_root_elements":0})
    return {"rows":rows,"formula_shared_forward_block_matmul_flops":"B*L*(24*N*D^2 + 4*N^2*D); linear QKVO + 4D MLP + QK/PV",
            "formula_v1_forward_roots":"L*(2*B*N + B*N*M + G*D + B*H*N^2) + B*N + B*(K+6)",
            "formula_v2_projected_forward_roots":"L*(2*B*N + B*N*M + 2*B*H*N) + B*(N+2) + B*(K+3)",
            "formula_v2_tail_energy_forward_roots":"L*(2*B*N + B*N*M) + B + B*(K+3)",
            "head_correction":"Frozen v1 roots the row normalizer six times, not each probability: K+6 scalar root elements per example; the v2 power head uses K+3.",
            "assumptions":"K=1000, batch=1 samples; position tables shared across batch; recurrence counted as written (compiler may prune the unused final carry update); rgelu/ALU each have one forward root per activation; analytic AD reuses v2 lift roots; frozen v1 attention recomputes slope roots backward; no rematerialization, optimizer, embedding or classifier matmul costs included in block-only FLOPs.",
            "total_step_flop_dominance":"unproven: elementwise arithmetic, reductions, divisions and optimizer work must also be counted; leading shared FLOPs cannot certify total-step dominance"}


def coverage_issues(data):
    """A partial or duplicated experiment matrix cannot support a full gate."""
    issues=[]
    variants=set(LABELS)
    specs={
        "stress":(["distribution","seed","alpha","variant"],
                  [["gaussian","student_t3"],range(42,50),[1,2,4,8,10,16,32,64,100,128,256],variants]),
        "crowding":(["n","foreground","scenario","seed","variant"],
                    [[197,256],[1,3,5],["mixed_queries","all_signal_queries"],range(42,50),variants]),
        "outliers":(["n","seed","outliers","scale_sqrt_d","pool"],
                    [[197,256],range(42,50),[1,3,5],[5,10,25,50,100],
                     ["raw_gap_diagnostic","standard_v1_actual_ln_gap","v2_lorentz","v2_inverse_energy"]]),
        "precision":(["seed","scale","variant","dtype"],
                     [range(42,50),[1,3,10,30,100],variants,["float32","bfloat16"]]),
        "positions":(["grid","dtype"],[[14,16,32,64],["float32","bfloat16"]]),
    }
    for section,(keys,axes) in specs.items():
        rows=data[section].get("records",[])
        observed=[tuple(r[k] for k in keys) for r in rows]
        missing=set(itertools.product(*axes))-set(observed)
        if missing or len(observed)!=len(set(observed)):
            issues.append(f"{section}: {len(missing)} missing cells or duplicate experiment keys")
    for kind in ["forward","spectral"]:
        rows=data["propagation"].get(kind,[])
        observed={(r["seed"],r["variant"],r["gate"],r["layer"]) for r in rows}
        expected={(seed,variant,"rgelu",layer) for seed in [42,43,44] for variant in variants for layer in range(1,25)}
        expected|={(seed,"v2_sharp","alu",layer) for seed in [42,43,44] for layer in range(1,25)}
        if not expected<=observed or len(rows)!=len(observed):
            issues.append(f"propagation: incomplete or duplicate {kind} matrix")
    c=data["convergence"]
    names={r["variant"] for r in c.get("records",[])}
    expected_names={"standard/rgelu/power/lorentz","algebraic_v1/rgelu/power/lorentz",
                    "v2_proposed/rgelu/power/lorentz","v2_sharp/rgelu/power/lorentz",
                    "v2_sharp/alu/power/inverse_energy","v2_rational_tail/alu/power/inverse_energy",
                    "v2_sharp/alu/brier/inverse_energy"}
    observed={(r["seed"],r["depth"],r["variant"],r["lr"]) for r in c.get("records",[])}
    expected=set(itertools.product([42,43,44],[2,4],expected_names,np.geomspace(1e-4,1e-2,5)))
    if not expected<=observed or len(c.get("records",[]))!=len(observed) or not expected_names<=names:
        issues.append("convergence: incomplete or duplicate paired LR/depth/variant matrix")
    if c.get("scope",{}).get("steps",0)<200 or any(len(r.get("loss_history",[]))<200 for r in c.get("records",[])):
        issues.append("convergence: fewer than 200 recorded optimization steps")
    if len(c.get("ntk",[]))<42 or len(c.get("head_gradient_diagnostics",[]))<60:
        issues.append("convergence: missing NTK or head-gradient diagnostics")
    if not all(data["symbolic"].get("checks",{}).values()) or len(data["symbolic"].get("checks",{}))<30:
        issues.append("symbolic: assertions failed or correction checks missing")
    if len(data["precision"].get("kernel_boundary_50_digit",[]))<12 or len(data["precision"].get("pool_boundary",[]))<24:
        issues.append("precision: missing boundary cases")
    for variant in ["algebraic_v1","v2_proposed","v2_rational_tail"]:
        if "two_layer_step_"+variant not in data["purity"].get("records",{}):
            issues.append("purity: missing complete synthetic step for "+variant)
    return issues


def comparison(data,variant,pool):
    stress=data["stress"]["records"]
    baseline={(r["distribution"],r["seed"],r["alpha"]):r for r in stress if r["variant"]=="algebraic_v1"}
    gradient=[]
    for r in stress:
        if r["variant"]==variant and 10<=r["alpha"]<=100:
            b=baseline[(r["distribution"],r["seed"],r["alpha"])]
            gradient.append(r["q_gradient_norm"]/b["q_gradient_norm"])
    radial=[]
    for r in data["stress"]["radial_counterexample"]:
        if r["alpha"]<=100 and variant in r:
            radial.append(r[variant]/r["algebraic_v1"])
    precision=[]
    raw=data["precision"]["records"]
    for dtype in ["float32","bfloat16"]:
        for scale in sorted({r["scale"] for r in raw}):
            a=[r for r in raw if r["variant"]==variant and r["dtype"]==dtype and r["scale"]==scale]
            b=[r for r in raw if r["variant"]=="algebraic_v1" and r["dtype"]==dtype and r["scale"]==scale]
            for metric in ["tv_max","backward_relative_error"]:
                precision.append({"dtype":dtype,"scale":scale,"metric":metric,"ratio_of_means":mean(a,metric)/max(mean(b,metric),1e-30)})
    raw=data["outliers"]["records"]
    pooling=[]
    for n in [197,256]:
        for count in [1,3,5]:
            for scale in [5,10,25,50,100]:
                a=[r for r in raw if r["pool"]==pool and r["n"]==n and r["outliers"]==count and r["scale_sqrt_d"]==scale]
                b=[r for r in raw if r["pool"]=="standard_v1_actual_ln_gap" and r["n"]==n and r["outliers"]==count and r["scale_sqrt_d"]==scale]
                pooling.append(mean(a,"angle_degrees")/mean(b,"angle_degrees"))
    return {"random_high_scale_gradient_min_paired_ratio":min(gradient),
            "random_high_scale_gradient_mean_paired_ratio":float(np.mean(gradient)),
            "radial_gradient_min_ratio":min(radial) if radial else None,
            "precision_cells":precision,"worst_precision_ratio":max(r["ratio_of_means"] for r in precision),
            "worst_pool_corruption_ratio":max(pooling),
            "gradient_pass":bool(min(gradient)>=1 and (not radial or min(radial)>=1)),
            "precision_pass":bool(max(r["ratio_of_means"] for r in precision)<=1 and all(r["finite"] for r in data["precision"]["records"] if r["variant"]==variant)),
            "pool_pass":bool(max(pooling)<=1),
            "criterion_note":"No tolerance chosen after seeing the data; exact sample non-regression is not a population dominance proof."}


def save(fig,path):
    fig.savefig(path.with_suffix(".png"),dpi=180,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def plot_series(ax,rows,xkey,ykey,variant,**kwargs):
    xs=sorted({r[xkey] for r in rows if r["variant"]==variant and r[ykey] is not None})
    means,errors=[],[]
    for x in xs:
        a=np.array([r[ykey] for r in rows if r["variant"]==variant and r[xkey]==x and r[ykey] is not None])
        means.append(a.mean());errors.append(a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0.)
    line,=ax.plot(xs,means,label=LABELS.get(variant,variant),**kwargs)
    ax.fill_between(xs,np.array(means)-errors,np.array(means)+errors,color=line.get_color(),alpha=0.12)


def figures(out,data):
    figdir=out/"figures"
    figdir.mkdir(exist_ok=True)
    rows=[r for r in data["stress"]["records"] if r["distribution"]=="gaussian"]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    keys=[("entropy","Attention entropy / log N"),("q_saturated_rows","Fraction of rows with ||dP/dQ||F < 1e-4"),
          ("q_gradient_norm","Query VJP Frobenius norm"),("score_saturated_rows","Score-space saturation (different units)")]
    for ax,(key,label) in zip(axes.flat,keys):
        for name in LABELS:
            plot_series(ax,rows,"alpha",key,name)
        ax.set_xscale("log",base=2)
        if key=="q_gradient_norm":ax.set_yscale("log")
        ax.set(xlabel="Pre-projection Q,K scale alpha",ylabel=label)
        ax.grid(alpha=0.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f"Figure 5 stress: 197 patches, 64 channels, {len({r['seed'] for r in rows})} paired seeds; bands are SE")
    save(fig,figdir/"figure5_score_runaway")

    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    rows=[r for r in data["outliers"]["records"] if r["n"]==197 and r["outliers"]==3]
    for name in sorted({r["pool"] for r in rows}):
        xs=sorted({r["scale_sqrt_d"] for r in rows})
        ys=[mean([r for r in rows if r["pool"]==name and r["scale_sqrt_d"]==x],"angle_degrees") for x in xs]
        axes[0].plot(xs,ys,"o-",label=name)
    axes[0].set(xscale="log",xlabel="Outlier norm / sqrt(d)",ylabel="Angular corruption (degrees)",title="3 outliers, N=197, d=768")
    axes[0].legend(fontsize=7)
    raw=[r for r in data["precision"]["records"] if r["dtype"]=="bfloat16"]
    for name in LABELS:plot_series(axes[1],raw,"scale","tv_max",name)
    axes[1].set(xscale="log",xlabel="Q,K scale",ylabel="Maximum row total variation error",title="BF16 inputs, FP32 arithmetic vs FP64")
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=0.2)
    save(fig,figdir/"robustness_precision")

    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    rows=[r for r in data["propagation"]["forward"] if r["gate"]=="rgelu"]
    for ax,(key,label) in zip(axes,[("variance_out","Residual variance"),("cosine_out","Mean patch cosine"),("spectral_gap","1 - |lambda2(P)|")]):
        for name in LABELS:plot_series(ax,rows,"layer",key,name)
        ax.set(xlabel="Layer",ylabel=label);ax.axvline(12,color="gray",ls=":");ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Matched-depth synthetic signal propagation (64 tokens, one 64D head)")
    save(fig,figdir/"propagation")

    fig,axes=plt.subplots(1,2,figsize=(13,4),layout="constrained")
    rows=data["convergence"]["records"]
    for ax,depth in zip(axes,[2,4]):
        for name in sorted({r["variant"] for r in rows}):
            xs=sorted({r["lr"] for r in rows})
            ys=[mean([r for r in rows if r["variant"]==name and r["depth"]==depth and r["lr"]==x],"heldout_accuracy") for x in xs]
            ax.plot(xs,ys,"o-",label=name,markersize=3)
        ax.set(xscale="log",xlabel="Learning rate",ylabel="Held-out accuracy",title=f"{depth} layers, {data['convergence']['scope']['steps']} steps, {len({r['seed'] for r in rows})} paired seeds",ylim=(0,1.05))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6)
    save(fig,figdir/"micro_convergence")


def build(out):
    out=Path(out)
    missing=[name for name in SECTIONS if not (out/f"{name}.json").is_file()]
    if missing:
        gate={"decision":"NO-GO","missing_evidence":missing}
        dump(out/"gate.json",gate)
        return gate
    data={name:json.loads((out/f"{name}.json").read_text()) for name in SECTIONS}
    integrity=evidence_integrity(data)
    coverage=coverage_issues(data)
    validation=json.loads((out/"validation.json").read_text()) if (out/"validation.json").is_file() else {}
    candidates={name:comparison(data,name,pool) for name,pool in [
        ("v2_proposed","v2_lorentz"),("v2_sharp","v2_lorentz"),
        ("v2_balanced","v2_lorentz"),("v2_rational_tail","v2_inverse_energy")]}
    current=validation_is_current(validation)
    formal=bool(current and validation.get("lean_build_passed") and validation.get("axioms_standard_only"))
    tests=bool(current and validation.get("tests_passed") and validation.get("frozen_baselines_unchanged"))
    positions=all(r["min_offset_distance"]>0 and r["rank_determinant"]!=0 and r["orthogonality_error"]<1e-5 for r in data["positions"]["records"])
    purity=all(not r["forbidden"] for r in data["purity"]["records"].values())
    for c in candidates.values():
        c["position_pass"]=positions and formal
        c["research_arithmetic_purity_pass"]=purity
        c["theoretical_attention_rsqrt_reduction"]=True
        c["total_training_step_flop_dominance_proven"]=False
        c["all_empirical_nonregression_checks_pass"]=all(c[k] for k in ["gradient_pass","precision_pass","pool_pass","position_pass","research_arithmetic_purity_pass"])
    # Tests prove implementation properties. They are not evidence of a
    # universal optimization or statistical dominance theorem.
    empirical_match=any(c["all_empirical_nonregression_checks_pass"] for c in candidates.values())
    decision="REVIEW-REQUIRED" if empirical_match and formal and tests and not integrity and not coverage else "NO-GO"
    gate={"decision":decision,"dominance_proven":False,"candidates":candidates,
          "evidence_integrity_issues":integrity,"formal_verified":formal,"tests_verified":tests,
          "coverage_issues":coverage,"validation_current":current,
          "reason":"No tested candidate satisfies every non-regression comparison; exact CE forward also violates algebraic purity. No population/full-model dominance theorem is established.",
          "phase_boundary":"Phase 1 research only. Phases 2-5 were not implemented or launched.",
          "strict_gate_satisfied":False}
    gate["theoretical_work"]=theoretical_work()
    dump(out/"gate.json",gate)
    figures(out,data)

    def stress_mean(variant,key,alpha=100):
        return mean([r for r in data["stress"]["records"] if r["variant"]==variant and r["distribution"]=="gaussian" and r["alpha"]==alpha],key)
    def pool_mean(pool):
        return mean([r for r in data["outliers"]["records"] if r["pool"]==pool and r["scale_sqrt_d"]==100],"angle_degrees")
    g=candidates
    rows=[]
    for name in candidates:
        c=candidates[name]
        rows.append(f"| {name} | {c['random_high_scale_gradient_min_paired_ratio']:.4g} | {c['radial_gradient_min_ratio'] if c['radial_gradient_min_ratio'] is not None else 'not measured'} | {c['worst_precision_ratio']:.4g} | {c['worst_pool_corruption_ratio']:.4g} | NO-GO |")
    lrrows=[]
    runs=data["convergence"]["records"]
    for depth in [2,4]:
        for name in sorted({r["variant"] for r in runs}):
            a=[r for r in runs if r["variant"]==name and r["depth"]==depth]
            stable=[r["lr"] for r in a if r["stable"]]
            bylr={lr:mean([r for r in a if r["lr"]==lr],"heldout_accuracy") for lr in sorted({r["lr"] for r in a})}
            lrrows.append(f"| {depth} | {name} | {max(stable) if stable else 'none'} | {max(bylr.values()):.3f} |")
    propagation_rows=[]
    for depth in [12,24]:
        for name in LABELS:
            a=[r for r in data['propagation']['forward'] if r['variant']==name and r['gate']=='rgelu' and r['layer']==depth]
            b=[r for r in data['propagation']['spectral'] if r['variant']==name and r['gate']=='rgelu' and r['layer']==depth]
            propagation_rows.append(f"| {name} | {depth} | {mean(a,'variance_out'):.4g} | {mean(a,'cosine_out'):.4g} | {mean(a,'spectral_gap'):.4g} | {np.mean([min(r['stack_singular_values']) for r in b]):.4g} | {np.mean([max(r['stack_singular_values']) for r in b]):.4g} |")
    workrows=[]
    for r in gate['theoretical_work']['rows']:
        workrows.append(f"| {r['shape']} | {r['tokens']} | {r['shared_forward_backward_block_matmul_flops']:,} | {r['v1_model_loss_forward_root_elements']:,} | {r['v2_projected_model_loss_forward_root_elements']:,} | {r['v2_tail_energy_model_loss_forward_root_elements']:,} |")
    monte_seeds=data['stress']['provenance']['seeds']
    prototype_seeds=data['convergence']['provenance']['seeds']
    text=f"""# Phase 1 analytical report

**Decision: NO-GO. The requested strict dominance gate is not satisfied.** This repository contains a completed set of Phase 1 investigations and executable counterexamples, not an approved production `algebraic_v2`. Tests passing do not turn failed mathematical or statistical gates into a pass. No Phase 2-5 implementation, TPU kernel engineering, sweeps, or main runs were performed.

## Evidence and scope

Coverage issues: **{coverage or 'none'}**. Algorithm/source integrity issues: **{integrity or 'none'}**. Verification matches current sources: **{current}**.

{len(monte_seeds)} paired Monte Carlo seeds ({', '.join(str(s) for s in monte_seeds)}) cover Gaussian and Student-t stress, N=197/256 crowding with k=1/3/5 foreground patches, d=768/1024 outlier pooling, and FP32/BF16 perturbations through scale 100. {len(prototype_seeds)} paired seeds ({', '.join(str(s) for s in prototype_seeds)}) cover 12/24-layer signal propagation and 2/4-layer, {data['convergence']['scope']['steps']}-step synthetic patch classification on five learning rates from 1e-4 to 1e-2. Every JSON records source hashes, versions, seeds, backend and elapsed time. Research runs use CPU and FP64 where stated; these are not TPU speed or full ViT-B/L accuracy measurements.

The exact spectral study uses the complete 64 by 64 Jacobian of a 4-token, 16-channel state; the separate forward propagation study uses 64 tokens and 64-dimensional heads. The layer counts match ViT-B/L. Width, training data and parameter counts do not. F(q) and G(c) are empirical trajectories, not an infinite-width proof. [Propagation data](propagation.json) includes every layer's singular spectrum, stack spectra and Lyapunov exponents at depths 12 and 24.

## Axis 1: exact algebra, derivatives and necessary corrections

All {len(data['symbolic']['checks'])} symbolic and high-precision assertions passed. [Full symbolic record](symbolic.json).

* For kappa>|u|, the positive square root is (kappa²+u²)/(kappa²-u²), so the radical base equals (kappa+u)/(kappa-u). Raising to n proves the claimed identity. For n=8 the Taylor series is `1 + 16u/kappa + 128u²/kappa² + 688u³/kappa³ + 2816u⁴/kappa⁴ + O(u⁵)`. For |u|<=1 and kappa>1 the explicit bounds in the prompt hold. The log derivatives are `16*kappa/(kappa²-u²)` in u and `-16*u/(kappa²-u²)` in kappa. Query/key derivatives include the ball-lift differential `I/r - qqᵀ/r³` and the orthogonal relative position map; complete formulas are in the JSON. At 50 digits the identity's maximum relative residual is {data['symbolic']['telemetry']['relative_identity_error']}.
* `1+1/rho(t)` ranges over (1,infinity), not (1.05,2]. The research correction is `1.05+0.95/(1+t²)`. A nonzero initial t is needed if this parameter is to learn immediately; its derivative vanishes at zero.
* Homogeneous recurrence has squared norm `(1+w²)^c`; it never reaches zero from (1,0). The polynomial identity proves orthogonality over the reals. Floating-point rotations are **not machine-exact**. Common binary rescaling preserves the represented projective point over the reals and prevents overflow for the tested bounded frequencies.
* Arbitrary SO(4) frequencies need not have rank two. The phase Jacobian has determinant `-4*(atan(wx1)*atan(wy2)+atan(wx2)*atan(wy1))`. Tying wx1=wx2=wx>0 and wy1=wy2=wy>0 makes it `-8*atan(wx)*atan(wy) != 0`. With wx<=1/(4G), wy<=1/(5G), all tested grid offsets stay within the non-wrapping region. Local rank alone is insufficient to rule out global periodic aliases. Also, independent planes in the frozen v1 model can jointly have rank two; a null-line in one plane does not imply a whole-head null-line.
* Lorentz pooling preserves identical tokens, and for nonempty finite tokens with d>0 its denominator obeys `D >= N²*d/(d+max ||h_i||²)>0`. The isolated-outlier derivative is O(1/||h_i||) when N>=2 and the other finite tokens are fixed. N=1 is a counterexample to the unqualified attenuation claim because pooling is exactly the identity.
* Cancellation correction: with b_i=h_i/sqrt(d+||h_i||²), the **same real-valued** denominator is `D = N*sum_i d/(d+||h_i||²) + N*sum_i ||b_i-mean(b)||²`. This O(Nd) form uses nonnegative summands. Tests cover its true VJP, not a clamped or surrogate backward pass.
* The exact CE gradient is conservative: off-diagonal Hessian entries are `-p_i*p_j*b_i*b_j`, where b_i=(n/tau)/sqrt(1+(z_i/tau)²). Replacing only the outside factor by the quarter power produces curl `p_i*p_j*(a_j*b_i-a_i*b_j)`, equal to {data['symbolic']['telemetry']['curl_max']['0.25']} in a concrete 50-digit example. Reject the quarter-power field as the gradient of the advertised objective, regardless of whether a toy optimizer converges.
* CE is strictly proper in p (KL excess risk); smoothed labels target the smoothed distribution. Its value contains log. A finite graph of arithmetic, square roots and piecewise comparisons is piecewise algebraic, so a custom arithmetic VJP cannot supply the exact non-algebraic CE value. [JAX's custom derivative contract](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html) separates primal values from derivative rules. The HLO negative control detects the forward log.
* Research alternatives use the genuine algebraic Brier loss or the existing strictly proper power-score family with a=7/8, n=8, tau=4, requiring three nested roots rather than six. These are changed objectives, explicitly named and differentiated normally; they are not described as exact MLE.

## Axis 2: propagation, spectra and score runaway

The prompt's attenuation number is reproduced: g8(94)={data['symbolic']['telemetry']['g8_at_94']}. This verifies the formula, not a new trained-model Figure 5 replication.

At alpha=100 on Gaussian Q,K, mean query VJP norms are v1={stress_mean('algebraic_v1','q_gradient_norm'):.6g}, proposed v2={stress_mean('v2_proposed','q_gradient_norm'):.6g}, sharpened v2={stress_mean('v2_sharp','q_gradient_norm'):.6g}, and rational-tail v2={stress_mean('v2_rational_tail','q_gradient_norm'):.6g}. Proposed v2 retains {stress_mean('v2_proposed','q_gradient_norm')/stress_mean('algebraic_v1','q_gradient_norm'):.3%} of v1's gradient norm despite high attention entropy. Bounding scores is not by itself gradient preservation. The lift's radial and tangential derivatives are d/(d+||q||²)^(3/2) and 1/sqrt(d+||q||²).

Both pre-projection Q derivatives and score-space derivatives are recorded. Score coordinates differ between kernels and cannot be compared as if they had the same units. Balanced attention couples rows; its complete VJP is measured, while inapplicable row-local Jacobian statistics are null. [Stress data](stress.json), [Figure 5 plot](figures/figure5_score_runaway.pdf), [propagation plot](figures/propagation.pdf).

The following means summarize the separate forward and exact-small-Jacobian studies. The singular values belong to the composed small state, not the 64-token forward study. Full per-layer spectra and Lyapunov exponents are retained in the data.

| Variant | Depth | Forward variance | Patch cosine | PF gap | Mean minimum stack SV | Mean maximum stack SV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(propagation_rows)}

## Axis 3: paired Monte Carlo, precision and convergence

Column balancing improves foreground recovery when most queries are background, but loses it when all queries want the same sparse foreground. For N=197,k=1, sharpening alone recovers about 0.9998 foreground mass; balancing falls to about 0.8303 in the all-signal-query scenario. Thus A3 is not selected as a general default: its extra N² reduction pass has no universal SNR benefit. [Crowding data and explicit gradient SNR definition](crowding.json).

The actual baseline applies final LayerNorm **per token before GAP**. At outlier norm 100*sqrt(d), averaged over all tested counts, sizes and seeds, raw GAP corruption is {pool_mean('raw_gap_diagnostic'):.3f} degrees, actual baseline corruption is {pool_mean('standard_v1_actual_ln_gap'):.3f}, proposed Lorentz pooling is {pool_mean('v2_lorentz'):.3f}, and inverse-energy pooling is {pool_mean('v2_inverse_energy'):.3f}. Comparing v2 only to raw GAP would have hidden its regression against v1. The second pooling correction is the rational centroid `sum_i a_i*h_i/sum_i a_i`, a_i=d/(d+||h_i||²). It preserves fixed points and has redescending isolated-outlier influence O(1/||h_i||), derivative O(1/||h_i||²); it changes the geometric pooling objective. [Outlier data](outliers.json), [robustness and precision plot](figures/robustness_precision.pdf).

Finite-precision tests compare each kernel with its own FP64 reference, not one kernel's answers with another's. Boundary u near +/-1 and kappa near 1.05 have a 50-digit mpmath probability and diagonal-derivative oracle. Native BF16 rational arithmetic and FP32 nonlinear accumulation are reported separately. Nearly collinear high-norm pooling is compared to an exact fixed point and a separate FP64 VJP. The naive denominator becomes nonfinite or grossly inaccurate at large scales; the algebraically identical positive-sum denominator remains finite through 1e6. [Precision data](precision.json). Relative errors near a zero reference gradient can be large even when absolute errors are small; they must not be interpreted as task accuracy.

The table reports the largest stable learning rate observed anywhere in the declared grid for each configuration and the best *mean* held-out accuracy across the grid; it does not certify stability at every seed or a larger LR. Full per-seed results, complete loss traces, empirical 32-output NTK eigenvalues, numerical ranks and positive-subspace condition numbers are in [convergence.json](convergence.json). The MSLR estimate is right-censored at 0.01 whenever that endpoint is stable. Loss values from different proper scores are not comparable. [Convergence plot](figures/micro_convergence.pdf).

| Depth | Synthetic configuration | Largest observed stable LR | Best mean held-out accuracy |
| --- | --- | ---: | ---: |
{chr(10).join(lrrows)}

Both rgelu and the coupled algebraic ALU gate are evaluated. These small prototypes do not select a gate for a 304M-parameter ViT-L. Head-only diagnostics compare tau=2/4 and both exponents over the same LR grid; optimization observations do not override the curl counterexample.

## Axis 4: Lean and arithmetic verification

Formal build verified: **{formal}**. Tests and frozen-baseline check verified: **{tests}**. See [validation.json](validation.json) and [verification log](verification.log). The pinned Lean/Mathlib versions are unchanged. V1 modules are unchanged; new V2 modules prove rational equivalence, positivity, explicit bounds/floor, projective orthogonality/determinant and power recurrence, the actual paired SO(4) relative-offset identity, the conditional rank determinant, and the vector-norm centroid denominator bound. Axiom auditing covers every locally declared theorem and lemma, including v1. Only Lean's standard axioms are allowed; no `sorryAx` is accepted. See [Lean instructions](../../formal/README.md).

[Primitive HLO audit](purity.json) covers forward and backward research primitives and two-layer synthetic loss/gradient graphs on CPU, with a transcendental negative control. This is intentionally not the Phase 2 full-size TPU model audit. Per head at N=197 the original attention requires 38,809 scalar root evaluations versus 394 for the ball lift (591 with column balancing), or zero for the rational-tail correction. All retain the shared 4*N²*d_h leading QK/PV matmul FLOPs. Projective tables remove recurrence roots; the decoupled power head halves the sequential root depth. Division, reduction, MXU scheduling and compiler fusion still matter: these counts do not prove a TPU latency improvement or total training-step FLOP dominance.

The existing v1 head already avoids per-class nested roots: it roots the **row normalizer** six times. For K classes its forward loss therefore uses K+6 scalar root elements, versus K+3 for the corrected power head. A claim of removing three roots from every class would overstate the actual saving. The formulas in [gate.json](gate.json) account for this implementation detail, forward LayerNorm/MLP/pooling roots and shared positional tables.

The table gives batch-one model/loss forward root elements and the leading **block matmul** forward/backward FLOPs, derived at the requested B/L shapes without implementing production models. Shared leading matmuls are `3*L*(24*N*D²+4*N²*D)`; the factor three counts the forward and two linear backward products. Optimizer, embedding/head matmuls, rematerialization and remaining elementwise work are excluded and explicitly prevent a total-step FLOP certificate. In particular, the full model's MLP roots remain even when attention becomes root-free.

| Shape | N | Shared block train matmul FLOPs | v1 forward roots | Projected v2 forward roots | Tail + inverse-energy forward roots |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(workrows)}

## Axis 5: self-correction record and strict gate

Iteration 0 verifies the proposed formula and rejects the advertised margin range, unconditional positional injectivity, machine-exact floating-point orthogonality, unqualified outlier attenuation, quarter-power conservative gradient, and pure exact CE forward claims.

Iteration 1 tests kappa=1.1 and gain=2, tied grid-safe frequencies, the positive-sum Lorentz denominator and genuine algebraic proper losses. Foreground sharpness and numerical stability improve, but pre-projection gradient preservation and actual-baseline pooling comparisons still fail.

Iteration 2 tests a root-free unbounded rational-tail score link: for t=|s|/8, f(t)=1+t+t²/(2+t); use f on positive scores and 1/f on negative scores, followed by three squarings and normalization. It matches E8 through second order at zero and avoids the lift's radial saturation. With inverse-energy pooling and the ALU/power-head ablations, it improves the random high-scale gradient study and outlier study. It still regresses in individual finite-precision cells and an explicit radial query case. All affected symbolic, numerical, prototype, purity and formal checks are rerun; no failed metric is suppressed.

An implementation regression test also exposed the absolute-value subgradient at exactly s=0: writing the positive branch entirely through abs(s) incorrectly returned zero slope. The final implementation keeps its linear positive term signed, while both evaluated branches remain finite. The analytic derivative test includes exactly zero, and all numerical studies were rerun after this correction. No surrogate derivative hides the defect.

Ratios below one are bad for gradients; ratios above one are bad for errors/corruption. These are sample gates, strictly compared without a post-hoc tolerance. They do not constitute a population dominance theorem.

| Candidate | Minimum random paired gradient ratio, alpha 10-100 | Minimum radial ratio, alpha 10-100 | Worst FP error ratio of means | Worst pool corruption ratio of means | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

The self-correction loop has produced improved candidates and concrete remaining counterexamples. The directive's demand to keep iterating *until provable domination* has **not been fulfilled**. A finite Monte Carlo study cannot turn a false unconditional claim into a theorem, and this report does not assert a successful phase gate. The honest result is NO-GO, with all failures retained and no later phase started. Implementing a production candidate requires a mathematically consistent objective and an explicit, defensible statistical acceptance criterion.

## Reproduction and artifact integrity

From the repository root, create an environment using `requirements-phase1.txt`, then run `python -m analysis.phase1.run` and `python -m analysis.phase1.verify`. Run `python -m analysis.phase1.run --require-pass` to receive exit status 2 when the research gate is not satisfied. Results are deterministic for the recorded seeds on the recorded CPU/JAX versions; execution times and plots' PDF metadata may differ. Full references and a requirement/evidence checklist are in [the research README](../../analysis/phase1/README.md). [manifest.json](manifest.json) hashes generated artifacts; each data file independently hashes its research dependencies.
"""
    (out/"REPORT.md").write_text(text)
    files=[p for p in out.rglob("*") if p.is_file() and p.name!="manifest.json" and not p.name.endswith(".tmp")]
    dump(out/"manifest.json",{"artifacts":{str(p.relative_to(out)):{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size} for p in sorted(files)},
                              "source_integrity_issues":integrity,"coverage_issues":coverage,"decision":gate["decision"]})
    return gate


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"results/phase1")
    args=parser.parse_args()
    result=build(args.out)
    print(result["decision"])
