#!/usr/bin/env python3
"""Monte Carlo stress test of attention under growing score scale.

Rows of 197 scores alpha * t, with t drawn from a Gaussian or a Student-t(3)
distribution, are pushed through softmax and radical attention. For each
alpha we record the normalised entropy of the visible weights, the largest
weight, and the Frobenius norm of the Jacobian dp/ds (exact, via the closed
form diag(g) (diag(p) - p p^T) applied row-wise). Writes
results/analysis/stress.json.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from algebraic_vision.primitives import (  # noqa: E402
    radical_exp, radical_exp_slope)

OUT = Path(__file__).resolve().parents[1] / "results/analysis/stress.json"
ROWS, KEYS = 10_000, 197
ALPHAS = (1, 2, 4, 8, 16, 32, 64, 128)


def softmax_stats(s):
    p = jax.nn.softmax(s, -1)
    g = jnp.ones_like(s)
    return p, g


def radical_stats(order, sink):
    def f(s):
        e = radical_exp(s, order)
        p = e / (sink + jnp.sum(e, -1, keepdims=True))
        return p, radical_exp_slope(s, order)
    return f


def summarise(p, g):
    mass = jnp.sum(p, -1, keepdims=True)
    q = p / mass
    ent = -jnp.sum(q * jnp.log(jnp.maximum(q, 1e-300)), -1) / np.log(KEYS)
    # ||J||_F^2 for J_ij = g_j p_i (delta_ij - p_j) equals
    # sum_j g_j^2 p_j^2 [(1 - p_j)^2 + sum_{i != j} p_i^2]; the second
    # bracket term is clamped at 0 because it cancels when p_j -> 1.
    pp = jnp.sum(p * p, -1, keepdims=True)
    rest = jnp.maximum(pp - p * p, 0.0)
    jf2 = jnp.sum((g * g) * (p * p) * ((1 - p) ** 2 + rest), -1)
    jf = jnp.sqrt(jf2)
    return {"entropy": float(jnp.mean(ent)),
            "entropy_p01": float(jnp.quantile(ent, 0.01)),
            "max_weight": float(jnp.mean(jnp.max(p, -1))),
            "jacobian_fro": float(jnp.mean(jf)),
            "jacobian_median": float(jnp.median(jf)),
            "saturated_rows": float(jnp.mean(jf < 1e-4))}


def main():
    rng = np.random.default_rng(0)
    base = {"gaussian": rng.normal(size=(ROWS, KEYS)),
            "student_t3": rng.standard_t(3, size=(ROWS, KEYS))}
    kinds = {"softmax": softmax_stats,
             "radical_n8": radical_stats(8, 0.0),
             "radical_n8_sink1": radical_stats(8, 1.0),
             "radical_n16": radical_stats(16, 0.0)}
    out = {}
    for dist, t in base.items():
        for name, f in kinds.items():
            rows = []
            for a in ALPHAS:
                rows.append({"alpha": a, **summarise(*f(jnp.asarray(a * t)))})
            out.setdefault(dist, {})[name] = rows
            print(dist, name, " ".join(
                f"a={r['alpha']}:H={r['entropy']:.3f},"
                f"Jmean={r['jacobian_fro']:.3f},"
                f"Jmed={r['jacobian_median']:.1e},"
                f"sat={r['saturated_rows']:.2f}"
                for r in rows if r['alpha'] in (1, 8, 32, 128)), flush=True)
    # Two criteria were set before the test: an entropy floor at the largest
    # scale, and a mean Jacobian norm ten times that of softmax. The second
    # fails: the mean is dominated by the few softmax rows that are not
    # saturated, so the fraction of saturated rows is reported as well.
    algs = ("radical_n8", "radical_n16")
    out["entropy_floor"] = bool(all(
        out[d][k][-1]["entropy"] > 0.05 for d in out for k in algs))
    out["mean_jacobian_10x_softmax"] = bool(all(
        out[d][k][-1]["jacobian_fro"] > 10 *
        out[d]["softmax"][-1]["jacobian_fro"]
        for d in out for k in algs))
    print("entropy floor:", out["entropy_floor"],
          "| mean Jacobian 10x softmax:", out["mean_jacobian_10x_softmax"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
