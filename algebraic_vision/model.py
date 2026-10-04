"""Vision Transformer with interchangeable standard and algebraic components.

One configuration class describes both models of the comparison. The
standard ViT uses softmax attention, a GELU MLP and learned absolute position
embeddings; the algebraic ViT uses radical attention, the radical-logistic
gate and learned 2-D Cayley rotations on top of the same absolute embeddings.
Everything else (patching, LayerNorm, residual layout, pooling,
initialisation) is shared, so the two models differ only where the
configuration says they do. The remaining options (attention sink, ALU gate,
fixed rotations, RoPE-2D, ...) are the variants studied along the way.

Parameters are float32 pytrees; matrix products run in ``compute_dtype``
(bfloat16 by default) with float32 accumulation, and the residual stream,
normalisation and attention scores stay in float32.
"""

import math
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from algebraic_vision.attention import radical_attention, softmax_attention
from algebraic_vision.primitives import (alu, cayley_frequencies,
                                         cayley_tables, radical_exp,
                                         radical_gelu, rotate_pairs)


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    num_classes: int = 1000
    width: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    pool: str = "gap"            # "gap" or "cls"
    posemb: str = "learned"      # "learned", "sincos2d" or "none"
    # "none", "rope2d" (sine/cosine), "cayley2d" (fixed rational
    # frequencies) or "cayley2d_mixed" (learned per head and pair)
    rotary: str = "none"
    attention: str = "softmax"   # "softmax" or "radical"
    order: int = 8               # n in E_n for radical attention
    temperature: float = 1.0     # scores are multiplied by this before E_n
    sink: str = "none"           # "none", "fixed" or "learned"
    sink_init: float = 0.0       # sink mass is E_n(sink_init); 1.0 at 0
    activation: str = "gelu"     # "gelu", "alu" or "rgelu"
    alu_c: float = math.sqrt(2.0 / math.pi)
    act_beta: float = 1.702      # rgelu: x * sigma_n(beta x)
    act_order: int = 8
    glu: bool = False            # gated MLP: act(x W_g) * (x W_u)
    compute_dtype: str = "bfloat16"
    remat: bool = False

    @property
    def grid(self):
        return self.image_size // self.patch_size

    @property
    def num_patches(self):
        return self.grid ** 2

    @property
    def head_dim(self):
        return self.width // self.heads


def standard_config(**overrides):
    """The canonical ViT: softmax, GELU, learned position embeddings."""
    return replace(ViTConfig(), **overrides)


def algebraic_config(**overrides):
    """The algebraic ViT.

    Radical attention E_8 without a sink, the radical-logistic gate
    x sigma_8(1.702 x), learned absolute embeddings plus learned 2-D Cayley
    rotations. It is trained with the power score a = 63/64 on an E_64 link
    (the loss is part of the training configuration).
    """
    base = ViTConfig(attention="radical", sink="none", activation="rgelu",
                     rotary="cayley2d_mixed")
    return replace(base, **overrides)


# --------------------------------------------------------------------- init

def _xavier(key, fan_in, fan_out, shape):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, jnp.float32, -limit, limit)


def _lecun(key, shape):
    std = 1.0 / math.sqrt(shape[0])
    return std * jax.random.truncated_normal(key, -2.0, 2.0, shape,
                                             jnp.float32) / 0.87962566


def _norm_params(width):
    return {"scale": jnp.ones((width,), jnp.float32),
            "bias": jnp.zeros((width,), jnp.float32)}


def init_params(cfg, key):
    """Initialise parameters following the big_vision ViT conventions."""
    if cfg.width % cfg.heads or (cfg.head_dim % 4 and cfg.rotary != "none"):
        raise ValueError("width/heads must divide, and rotary needs "
                         "head_dim divisible by 4")
    keys = iter(jax.random.split(key, 4 + 6 * cfg.depth))
    d = cfg.width
    patch_dim = cfg.patch_size ** 2 * cfg.channels
    hidden = cfg.mlp_dim * (2 if cfg.glu else 1)
    params = {
        "patch": {"w": _lecun(next(keys), (patch_dim, d)),
                  "b": jnp.zeros((d,), jnp.float32)},
        "norm": _norm_params(d),
        "head": {"w": jnp.zeros((d, cfg.num_classes), jnp.float32),
                 "b": jnp.zeros((cfg.num_classes,), jnp.float32)},
        "blocks": [],
    }
    if cfg.posemb == "learned":
        params["posemb"] = (jax.random.normal(next(keys), (cfg.num_patches, d))
                            / math.sqrt(d))
    else:
        next(keys)
    if cfg.pool == "cls":
        params["cls"] = jnp.zeros((1, d), jnp.float32)
    for _ in range(cfg.depth):
        kq, kk, kv = jax.random.split(next(keys), 3)
        block = {
            "ln1": _norm_params(d),
            "qkv": {"w": jnp.concatenate([_xavier(kx, d, d, (d, d))
                                          for kx in (kq, kk, kv)], axis=1),
                    "b": jnp.zeros((3 * d,), jnp.float32)},
            "proj": {"w": _xavier(next(keys), d, d, (d, d)),
                     "b": jnp.zeros((d,), jnp.float32)},
            "ln2": _norm_params(d),
            "fc1": {"w": _xavier(next(keys), d, cfg.mlp_dim, (d, hidden)),
                    "b": jnp.zeros((hidden,), jnp.float32)},
            "fc2": {"w": _xavier(next(keys), cfg.mlp_dim, d,
                                 (cfg.mlp_dim, d)),
                    "b": jnp.zeros((d,), jnp.float32)},
        }
        if cfg.sink == "learned":
            block["sink"] = jnp.full((cfg.heads,), cfg.sink_init, jnp.float32)
        if cfg.rotary == "cayley2d_mixed":
            block["rot"] = mixed_rotation_init(cfg, next(keys))
        else:
            next(keys)
        params["blocks"].append(block)
    return params


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# ------------------------------------------------------------------ layers

def layer_norm(x, p, eps=1e-6):
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(var + eps) * p["scale"] + p["bias"]


def dense(x, p, dtype):
    y = jnp.dot(x.astype(dtype), p["w"].astype(dtype),
                preferred_element_type=jnp.float32)
    return y + p["b"]


def sincos_2d(grid, width, temperature=10000.0):
    """Fixed 2-D sine-cosine position embedding (posemb="sincos2d")."""
    y, x = np.mgrid[:grid, :grid]
    omega = np.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / temperature ** omega
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    emb = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    return jnp.asarray(emb, jnp.float32)


def rotary_tables(cfg):
    """Per-token (cos-like, sin-like) tables of shape (tokens, head_dim/2).

    The first half of the pairs rotates with the column index and the second
    half with the row index. "cayley2d" uses Cayley rotations with a fixed
    frequency grid (algebraic); "rope2d" uses cos/sin (standard RoPE).
    """
    quarter = cfg.head_dim // 4
    if cfg.rotary == "cayley2d":
        cos_1d, sin_1d = cayley_tables(cfg.grid, cayley_frequencies(quarter))
    elif cfg.rotary == "rope2d":
        theta = 100.0 ** (-np.arange(quarter) / quarter)
        angles = np.arange(cfg.grid)[:, None] * theta[None, :]
        cos_1d = jnp.asarray(np.cos(angles), jnp.float32)
        sin_1d = jnp.asarray(np.sin(angles), jnp.float32)
    else:
        raise ValueError(f"no fixed rotation tables for {cfg.rotary!r}")
    rows = np.repeat(np.arange(cfg.grid), cfg.grid)
    cols = np.tile(np.arange(cfg.grid), cfg.grid)
    cos = jnp.concatenate([cos_1d[cols], cos_1d[rows]], axis=1)
    sin = jnp.concatenate([sin_1d[cols], sin_1d[rows]], axis=1)
    if cfg.pool == "cls":       # the class token is not rotated
        cos = jnp.concatenate([jnp.ones((1, 2 * quarter)), cos], axis=0)
        sin = jnp.concatenate([jnp.zeros((1, 2 * quarter)), sin], axis=0)
    return cos, sin


def mixed_rotation_init(cfg, key):
    """Learned 2-D Cayley frequencies: each pair turns along its own direction.

    Pair k gets the magnitude of the fixed grid and a random direction
    (a, b) = ((1 - t^2)/(1 + t^2), 2t/(1 + t^2)), t uniform in [-1, 1], so
    that its rotation is R(a w_k)^col R(b w_k)^row.
    """
    pairs = cfg.head_dim // 2
    w = cayley_frequencies(pairs)
    t = jax.random.uniform(key, (cfg.heads, pairs), minval=-1.0, maxval=1.0)
    a = (1.0 - t * t) / (1.0 + t * t)
    b = 2.0 * t / (1.0 + t * t)
    return {"wx": a * w, "wy": b * w}


def mixed_rotary_tables(cfg, rot):
    """Per-head tables (heads, tokens, head_dim/2) for learned frequencies."""
    cx, sx = cayley_tables(cfg.grid, rot["wx"])      # (grid, heads, pairs)
    cy, sy = cayley_tables(cfg.grid, rot["wy"])
    rows = np.repeat(np.arange(cfg.grid), cfg.grid)
    cols = np.tile(np.arange(cfg.grid), cfg.grid)
    cos = cx[cols] * cy[rows] - sx[cols] * sy[rows]   # angle addition
    sin = sx[cols] * cy[rows] + cx[cols] * sy[rows]
    cos, sin = cos.transpose(1, 0, 2), sin.transpose(1, 0, 2)
    if cfg.pool == "cls":
        h, _, half = cos.shape
        cos = jnp.concatenate([jnp.ones((h, 1, half)), cos], axis=1)
        sin = jnp.concatenate([jnp.zeros((h, 1, half)), sin], axis=1)
    return cos, sin


def sink_mass(cfg, block):
    if cfg.sink == "none":
        return jnp.zeros((cfg.heads,), jnp.float32)
    if cfg.sink == "fixed":
        value = radical_exp(jnp.float32(cfg.sink_init), cfg.order)
        return jnp.full((cfg.heads,), value, jnp.float32)
    return radical_exp(block["sink"], cfg.order)


def activation(cfg, x):
    if cfg.activation == "gelu":
        return jax.nn.gelu(x, approximate=False)
    if cfg.activation == "alu":
        return alu(x, cfg.alu_c)
    if cfg.activation == "rgelu":
        return radical_gelu(x, cfg.act_beta, cfg.act_order)
    raise ValueError(f"unknown activation {cfg.activation!r}")


def attend(cfg, q, k, v, block):
    if cfg.attention == "softmax":
        return softmax_attention(q, k, v)
    if cfg.attention == "radical":
        return radical_attention(q, k, v, sink_mass(cfg, block), cfg.order,
                                 cfg.temperature)
    raise ValueError(f"unknown attention {cfg.attention!r}")


def shared_rotary_tables(cfg):
    """Rotation tables shared by all blocks, or None.

    Fixed rotations use one table for every block; learned rotations are
    built per block from its parameters (see block_qkv).
    """
    if cfg.rotary in ("none", "cayley2d_mixed"):
        return None
    return rotary_tables(cfg)


def block_qkv(cfg, x, block, tables):
    """Queries, keys and values of one block, rotated by position."""
    dtype = jnp.dtype(cfg.compute_dtype)
    b, t, _ = x.shape
    h = layer_norm(x, block["ln1"])
    qkv = dense(h, block["qkv"], dtype).astype(dtype)
    qkv = qkv.reshape(b, t, 3, cfg.heads, cfg.head_dim)
    q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
    if cfg.rotary == "cayley2d_mixed":
        tables = mixed_rotary_tables(cfg, block["rot"])
    if tables is not None:
        q = rotate_pairs(q, *tables)
        k = rotate_pairs(k, *tables)
    return q, k, v


def block_forward(cfg, x, block, tables):
    dtype = jnp.dtype(cfg.compute_dtype)
    b, t, d = x.shape
    q, k, v = block_qkv(cfg, x, block, tables)
    o = attend(cfg, q, k, v, block)
    o = o.transpose(0, 2, 1, 3).reshape(b, t, d)
    x = x + dense(o, block["proj"], dtype)

    h = layer_norm(x, block["ln2"])
    m = dense(h, block["fc1"], dtype)
    if cfg.glu:
        gate, up = jnp.split(m, 2, axis=-1)
        m = activation(cfg, gate) * up
    else:
        m = activation(cfg, m)
    return x + dense(m.astype(dtype), block["fc2"], dtype)


def patchify(images, patch):
    b, h, w, c = images.shape
    x = images.reshape(b, h // patch, patch, w // patch, patch, c)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(
        b, (h // patch) * (w // patch), patch * patch * c)


def embed(cfg, params, images):
    dtype = jnp.dtype(cfg.compute_dtype)
    x = dense(patchify(images, cfg.patch_size), params["patch"], dtype)
    if cfg.posemb == "learned":
        x = x + params["posemb"]
    elif cfg.posemb == "sincos2d":
        x = x + sincos_2d(cfg.grid, cfg.width)
    if cfg.pool == "cls":
        cls = jnp.broadcast_to(params["cls"], (x.shape[0], 1, cfg.width))
        x = jnp.concatenate([cls, x], axis=1)
    return x


def forward(cfg, params, images, return_features=False):
    """Logits for a batch of images in [-1, 1], shape (B, H, W, C)."""
    x = embed(cfg, params, images)
    tables = shared_rotary_tables(cfg)
    step = partial(block_forward, cfg)
    if cfg.remat:
        step = jax.checkpoint(step)
    for block in params["blocks"]:
        x = step(x, block, tables)
    x = layer_norm(x, params["norm"])
    pooled = x[:, 0] if cfg.pool == "cls" else jnp.mean(x, axis=1)
    logits = dense(pooled, params["head"], jnp.float32)
    if return_features:
        return logits, x
    return logits
