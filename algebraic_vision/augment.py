"""On-device image augmentation, identical for both arms.

Augmentation runs inside the jitted training step from a key derived from
(seed, step), so a given seed produces the same crops and flips for the
standard and the algebraic model. It is part of the input pipeline, not of
the model.
"""

import jax
import jax.numpy as jnp


def to_unit_range(images):
    """uint8 [0, 255] -> float32 [-1, 1]."""
    return images.astype(jnp.float32) / 127.5 - 1.0


def _crop_one(key, image, out_size, scale, ratio):
    h, w = image.shape[0], image.shape[1]
    k_area, k_ratio, k_y, k_x = jax.random.split(key, 4)
    area = jax.random.uniform(k_area, (), minval=scale[0], maxval=scale[1])
    aspect = jax.random.uniform(k_ratio, (), minval=ratio[0], maxval=ratio[1])
    target = area * h * w
    ch = jnp.clip(jnp.sqrt(target / aspect), 1.0, h)
    cw = jnp.clip(jnp.sqrt(target * aspect), 1.0, w)
    y0 = jax.random.uniform(k_y, (), minval=0.0, maxval=h - ch)
    x0 = jax.random.uniform(k_x, (), minval=0.0, maxval=w - cw)
    sy, sx = out_size / ch, out_size / cw
    return jax.image.scale_and_translate(
        image, (out_size, out_size, image.shape[2]), (0, 1),
        jnp.stack([sy, sx]), jnp.stack([-y0 * sy, -x0 * sx]),
        method="linear", antialias=True)


def train_augment(key, images, out_size, scale=(0.35, 1.0),
                  ratio=(0.75, 4.0 / 3.0), flip=True):
    """Random resized crop and horizontal flip; returns floats in [-1, 1]."""
    x = to_unit_range(images)
    k_crop, k_flip = jax.random.split(key)
    keys = jax.random.split(k_crop, x.shape[0])
    crop = jax.vmap(lambda k, im: _crop_one(k, im, out_size, scale, ratio))
    x = crop(keys, x)
    if flip:
        mask = jax.random.bernoulli(k_flip, 0.5, (x.shape[0], 1, 1, 1))
        x = jnp.where(mask, x[:, :, ::-1], x)
    return x


def eval_preprocess(images, out_size):
    """Centre crop to out_size (identity when sizes match)."""
    h = images.shape[1]
    off = (h - out_size) // 2
    x = images[:, off:off + out_size, off:off + out_size]
    return to_unit_range(x)
