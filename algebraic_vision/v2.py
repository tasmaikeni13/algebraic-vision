"""Production equations selected by the completed Phase 1 study.

E8 attention is retained. The rejected Klein/Cayley kernel is an explicit
ablation in rational_kernel.py, never silently used as the selected model.
"""

from functools import partial

import jax
import jax.numpy as jnp

from algebraic_vision import losses


def frequency(raw, grid):
    return (0.25 + 0.75 / (1.0 + raw * raw)) / grid


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def projective_tables(w, grid):
    v, u = jnp.ones_like(w), jnp.zeros_like(w)
    cs, ss = [], []
    for _ in range(grid):
        vv, uu = v * v, u * u
        den = vv + uu
        cs.append((vv - uu) / den)
        ss.append(2.0 * u * v / den)
        v, u = v - w * u, u + w * v
    return jnp.stack(cs), jnp.stack(ss)


@projective_tables.defjvp
def _projective_jvp(grid, primals, tangents):
    (w,), (dw,) = primals, tangents
    c, s = projective_tables(w, grid)
    index = jnp.arange(grid, dtype=w.dtype).reshape((grid,) + (1,) * w.ndim)
    dtheta = 2.0 * index * dw / (1.0 + w * w)
    return (c, s), (-s * dtheta, c * dtheta)


def rotation_init(cfg, key):
    kx, ky = jax.random.split(key)
    shape = (cfg.heads, cfg.head_dim // 4)
    return {"raw_x": jax.random.uniform(kx, shape, minval=0.6, maxval=1.4),
            "raw_y": jax.random.uniform(ky, shape, minval=0.8, maxval=1.6)}


def rotary_tables(cfg, rot):
    cx, sx = projective_tables(frequency(rot["raw_x"], cfg.grid), cfg.grid)
    cy, sy = projective_tables(frequency(rot["raw_y"], cfg.grid), cfg.grid)
    row, col = jnp.arange(cfg.num_patches) // cfg.grid, jnp.arange(cfg.num_patches) % cfg.grid
    cp, sp = cx[col] * cy[row] - sx[col] * sy[row], sx[col] * cy[row] + cx[col] * sy[row]
    cm, sm = cx[col] * cy[row] + sx[col] * sy[row], sx[col] * cy[row] - cx[col] * sy[row]
    return tuple(a.transpose(1, 0, 2) for a in (cp, sp, cm, sm))


def rotate(x, tables):
    cp, sp, cm, sm = tables
    a, b, c, d = jnp.split(x.astype(jnp.float32), 4, -1)
    return jnp.concatenate((cp * a - sp * b, sp * a + cp * b,
                            cm * c - sm * d, sm * c + cm * d), -1).astype(x.dtype)


def inverse_energy_pool(x):
    x = x.astype(jnp.float32)
    d = x.shape[-1]
    a = d / (d + jnp.sum(x * x, -1, keepdims=True))
    return jnp.sum(a * x, -2) / jnp.sum(a, -2)


def _gate_parts(x, c):
    y = c * x
    den = jnp.where(y >= 0, 1.0 + jnp.maximum(y, -0.5),
                    1.0 - jnp.minimum(y, 0.5))
    r = 1.0 / den
    half = 0.5 * x * r
    return jnp.where(y >= 0, x - half, half), r, y


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def rational_gate(x, c=0.851):
    return _gate_parts(x, c)[0]


@rational_gate.defjvp
def _gate_jvp(c, primals, tangents):
    (x,), (dx,) = primals, tangents
    value, r, y = _gate_parts(x, c)
    half = 0.5 * r * r
    return value, jnp.where(y >= 0, 1.0 - half, half) * dx


def power_score(logits, labels, smoothing=0.1, backend="auto"):
    # E8(2*z) = rho(z/4)^8: tau and n are independent.
    z = 2.0 * logits.astype(jnp.float32)
    if backend == "pallas" or (backend == "auto" and jax.default_backend() == "tpu"):
        from algebraic_vision.kernels import fused_power_score
        return fused_power_score(z, labels, -3, 8, smoothing)
    return losses.power_score(z, labels, -3, 8, smoothing)


def probabilities(logits):
    return losses.radical_probabilities(2.0 * logits.astype(jnp.float32), 8)
