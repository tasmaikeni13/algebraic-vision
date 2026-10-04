#!/usr/bin/env python3
"""Finite-precision behaviour of standard and algebraic primitives.

For softmax against radical attention (no sink, as in the final model),
GELU against the ALU and radical-logistic gates, and cross-entropy against
the power scores, at several input scales, we measure

* own error: float32 implementation vs float64 reference on the same input;
* rounding effect: float64 evaluation on bfloat16-rounded inputs vs the
  float64 reference, i.e. how strongly the primitive propagates the input
  rounding that dominates a bfloat16 training graph.

Writes results/analysis/precision.json. Runs on CPU with x64 enabled.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from algebraic_vision.attention import radical_weights  # noqa: E402
from algebraic_vision.losses import cross_entropy, power_score  # noqa: E402
from algebraic_vision.primitives import alu, radical_gelu  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results/analysis/precision.json"
rng = np.random.default_rng(0)


def bf16_round(x):
    return jnp.asarray(x).astype(jnp.bfloat16).astype(jnp.float64)


def softmax_rows(s):
    return jax.nn.softmax(s, axis=-1)


def radical_rows(order, sink=0.0):
    def f(s):
        omega = jnp.full(s.shape[:-1] + (1,), sink, s.dtype)
        return radical_weights(s, omega, order, 1.0)
    return f


def tv(p, q):
    return float(jnp.max(0.5 * jnp.sum(jnp.abs(p - q), axis=-1)))


def rel_l2(a, b):
    return float(jnp.linalg.norm((a - b).ravel())
                 / max(float(jnp.linalg.norm(b.ravel())), 1e-300))


def finite(*arrays):
    return all(bool(jnp.all(jnp.isfinite(a))) for a in arrays)


def attention_study():
    rows = {}
    kinds = {"softmax": softmax_rows, "radical_n8": radical_rows(8),
             "radical_n16": radical_rows(16)}
    for scale in (0.3, 1.0, 3.0, 10.0, 30.0):
        s = jnp.asarray(rng.normal(0, scale, (256, 197)))
        g = jnp.asarray(rng.normal(0, 1, (256, 197)))
        for name, f in kinds.items():
            ref, vjp_ref = jax.vjp(f, s)
            p32, vjp32 = jax.vjp(f, s.astype(jnp.float32))
            pin = f(bf16_round(s))
            d_ref = vjp_ref(g)[0]
            d32 = vjp32(g.astype(jnp.float32))[0]
            d_in = jax.vjp(f, bf16_round(s))[1](g)[0]
            rows.setdefault(name, []).append({
                "scale": scale,
                "own_tv": tv(p32.astype(jnp.float64), ref),
                "round_tv": tv(pin, ref),
                "own_grad": rel_l2(d32.astype(jnp.float64), d_ref),
                "round_grad": rel_l2(d_in, d_ref),
                "finite": finite(p32, d32),
            })
    return rows


def activation_study():
    rows = {}
    kinds = {"gelu": lambda x: jax.nn.gelu(x, approximate=False),
             "alu": lambda x: alu(x),
             "rgelu": lambda x: radical_gelu(x)}
    for scale in (1.0, 3.0, 10.0):
        x = jnp.asarray(rng.normal(0, scale, 200_000))
        for name, f in kinds.items():
            df = jax.vmap(jax.grad(f))
            ref, dref = f(x), df(x)
            y32, d32 = f(x.astype(jnp.float32)), df(x.astype(jnp.float32))
            yin, din = f(bf16_round(x)), df(bf16_round(x))
            rows.setdefault(name, []).append({
                "scale": scale,
                "own_rel": rel_l2(y32.astype(jnp.float64), ref),
                "round_rel": rel_l2(yin, ref),
                "own_grad": rel_l2(d32.astype(jnp.float64), dref),
                "round_grad": rel_l2(din, dref),
                "finite": finite(y32, d32),
            })
    return rows


def loss_study():
    rows = {}
    labels = jnp.asarray(rng.integers(0, 1000, 256))
    kinds = {"cross_entropy": lambda z: cross_entropy(z, labels, 0.1,
                                                      reduction="none"),
             "power_7_8": lambda z: power_score(z, labels, -3, 8, 0.1,
                                                reduction="none"),
             "power_63_64": lambda z: power_score(z, labels, -6, 64, 0.1,
                                                  reduction="none")}
    for scale in (1.0, 3.0, 10.0, 30.0):
        z = jnp.asarray(rng.normal(0, scale, (256, 1000)))
        for name, f in kinds.items():
            # The production losses compute internally in float32; the
            # reference re-evaluates the same formula in float64.
            ref64 = _float64_reference(name, z, labels)
            val32, g32 = jax.value_and_grad(
                lambda v: jnp.sum(f(v)))(z.astype(jnp.float32))
            val32 = f(z.astype(jnp.float32))
            vref, gref = ref64
            vin, gin = _float64_reference(name, bf16_round(z), labels)
            rows.setdefault(name, []).append({
                "scale": scale,
                "own_rel": _rms_rel(val32.astype(jnp.float64), vref),
                "round_rel": _rms_rel(vin, vref),
                "own_grad": rel_l2(g32.astype(jnp.float64), gref),
                "round_grad": rel_l2(gin, gref),
                "finite": finite(val32, g32),
            })
    return rows


def _rms_rel(a, b):
    """Root-mean-square of per-sample relative errors."""
    return float(jnp.sqrt(jnp.mean(((a - b) / jnp.abs(b)) ** 2)))


def _float64_reference(name, z, labels):
    y = jax.nn.one_hot(labels, 1000) * 0.9 + 0.1 / 1000
    if name == "cross_entropy":
        def f(zz):
            return -jnp.sum(y * jax.nn.log_softmax(zz, -1), -1)
    else:
        a, n = (7.0 / 8.0, 8) if name == "power_7_8" else (63.0 / 64.0, 64)

        def f(zz):
            r = zz / n + jnp.sqrt(1 + (zz / n) ** 2)
            e = (r / jnp.max(r, -1, keepdims=True)) ** n
            p = e / jnp.sum(e, -1, keepdims=True)
            return (jnp.sum(p ** a, -1) / a
                    - jnp.sum(y * p ** (a - 1), -1) / (a - 1)
                    + jnp.sum(y ** a, -1) / (a * (a - 1)))
    return f(z), jax.grad(lambda v: jnp.sum(f(v)))(z)


def extremes():
    s = jnp.linspace(-1e4, 1e4, 20001, dtype=jnp.float32)[None, :]
    x = jnp.linspace(-1e6, 1e6, 200001, dtype=jnp.float32)
    z = jnp.linspace(-1e3, 1e3, 1000, dtype=jnp.float32)[None, :]
    lab = jnp.zeros((1,), jnp.int32)
    p = radical_rows(8)(s)
    gs = jax.grad(lambda v: jnp.sum(radical_rows(8)(v) ** 2))(s)
    a, ga = alu(x), jax.vmap(jax.grad(alu))(x)
    r, gr = radical_gelu(x), jax.vmap(jax.grad(radical_gelu))(x)
    lv, lg = jax.value_and_grad(lambda v: power_score(v, lab, -3, 8))(z)
    lv64, lg64 = jax.value_and_grad(
        lambda v: power_score(v, lab, -6, 64))(z)
    return {"radical_attention_|s|<=1e4": finite(p, gs),
            "alu_|x|<=1e6": finite(a, ga),
            "rgelu_|x|<=1e6": finite(r, gr),
            "power_score_7_8_|z|<=1e3": finite(lv, lg),
            "power_score_63_64_|z|<=1e3": finite(lv64, lg64)}


def criteria(att, act, los, ext):
    """Own float32 error below a tenth of the bfloat16 rounding effect (and
    finite at extreme inputs); rounding effect at most twice the standard
    primitive's."""
    pairs = [("attention", att, "softmax", ["radical_n8", "radical_n16"],
              ("own_tv", "round_tv"), ("own_grad", "round_grad")),
             ("activation", act, "gelu", ["alu", "rgelu"],
              ("own_rel", "round_rel"),
              ("own_grad", "round_grad")),
             ("loss", los, "cross_entropy", ["power_7_8", "power_63_64"],
              ("own_rel", "round_rel"), ("own_grad", "round_grad"))]
    verdict = {"own_error_small": True, "rounding_at_most_2x": True,
               "details": []}
    for group, table, std, algs, val_keys, grad_keys in pairs:
        for alg in algs:
            for i, row in enumerate(table[alg]):
                srow = table[std][i]
                for own, rnd in (val_keys, grad_keys):
                    g1 = row[own] <= 0.1 * row[rnd] and row["finite"]
                    g2 = row[rnd] <= 2.0 * srow[rnd] + 1e-12
                    verdict["own_error_small"] &= g1
                    verdict["rounding_at_most_2x"] &= g2
                    if not (g1 and g2):
                        verdict["details"].append(
                            f"{group}/{alg} scale={row['scale']} {own}="
                            f"{row[own]:.2e} {rnd}={row[rnd]:.2e} "
                            f"std {rnd}={srow[rnd]:.2e}")
    verdict["own_error_small"] &= all(ext.values())
    return verdict


def main():
    att, act, los, ext = (attention_study(), activation_study(), loss_study(),
                          extremes())
    verdict = criteria(att, act, los, ext)
    record = {"attention": att, "activation": act, "loss": los,
              "extremes_finite": ext, "criteria": verdict}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    for group, table in (("attention", att), ("activation", act),
                         ("loss", los)):
        for name, rows in table.items():
            for r in rows:
                vals = " ".join(f"{k}={v:.2e}" for k, v in r.items()
                                if isinstance(v, float) and k != "scale")
                print(f"{group:10s} {name:14s} scale={r['scale']:<5} {vals}")
    print("extremes finite:", ext)
    print("own error small:", verdict["own_error_small"],
          "rounding at most 2x standard:", verdict["rounding_at_most_2x"])
    for d in verdict["details"]:
        print("  ", d)


if __name__ == "__main__":
    main()
