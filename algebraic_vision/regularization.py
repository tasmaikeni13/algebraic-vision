"""Paired on-device crop, RandAugment, Mixup and CutMix streams.

Beta draws are replayed on the host; all image arithmetic on the device is
algebraic. These transforms apply identically to every experimental arm.
"""

import jax
import jax.numpy as jnp

from algebraic_vision.augment import to_unit_range
from algebraic_vision.losses import smoothed_targets


def _sqrt(x):
    return x * jax.lax.rsqrt(jnp.maximum(x, 1e-20))


def _crop(key, image, size, minimum):
    h, w = image.shape[:2]
    ka, kr, ky, kx, kf = jax.random.split(key, 5)
    area = jax.random.uniform(ka, (), minval=minimum, maxval=1.0) * h * w
    aspect = jax.random.uniform(kr, (), minval=0.75, maxval=4.0 / 3.0)
    ch, cw = jnp.clip(_sqrt(area / aspect), 1, h), jnp.clip(_sqrt(area * aspect), 1, w)
    y0 = jax.random.uniform(ky) * (h - ch)
    x0 = jax.random.uniform(kx) * (w - cw)
    scale = jnp.stack([size / ch, size / cw])
    x = jax.image.scale_and_translate(image, (size, size, 3), (0, 1), scale,
                       -jnp.stack([y0, x0]) * scale, method="linear", antialias=True)
    return jnp.where(jax.random.bernoulli(kf), x[:, ::-1], x)


def _randaugment(key, image, layers, magnitude):
    # A fixed shared policy with independently sampled brightness, contrast,
    # color, solarization, posterization and integer translations.
    for subkey in jax.random.split(key, layers):
        kop, ksign = jax.random.split(subkey)
        op = jax.random.randint(kop, (), 0, 7)
        sign = jnp.where(jax.random.bernoulli(ksign), 1.0, -1.0)
        level = sign * magnitude
        mean = jnp.mean(image, (0, 1), keepdims=True)
        gray = jnp.mean(image, -1, keepdims=True)
        shift = jnp.int32(level * image.shape[0] * 0.3)
        options = [lambda x: x + level, lambda x: (x - mean) * (1 + level) + mean,
                   lambda x: (x - gray) * (1 + level) + gray,
                   lambda x: jnp.where(x > 1 - 2 * magnitude, -x, x),
                   lambda x: jnp.floor((x + 1) * 8) / 8 - 1,
                   lambda x: _translate(x, shift, 0), lambda x: _translate(x, 0, shift)]
        image = jnp.clip(jax.lax.switch(op, options, image), -1.0, 1.0)
    return image


def _translate(image, dy, dx):
    h, w = image.shape[:2]
    yy, xx = jnp.arange(h) - dy, jnp.arange(w) - dx
    valid = (yy[:, None] >= 0) & (yy[:, None] < h) & (xx[None, :] >= 0) & (xx[None, :] < w)
    moved = image[jnp.clip(yy, 0, h - 1)[:, None], jnp.clip(xx, 0, w - 1)[None, :]]
    return jnp.where(valid[:, :, None], moved, 0.0)


def prepare_batch(cfg, classes, images, labels, key, mixing):
    coefficient = mixing[0] if jnp.ndim(mixing) else mixing
    kc, ka, kp, km, ky, kx = jax.random.split(key, 6)
    x = to_unit_range(images)
    keys = jax.random.split(kc, images.shape[0])
    x = jax.vmap(lambda k, im: _crop(k, im, cfg.image_size, cfg.crop_scale_min))(keys, x)
    if cfg.randaugment_layers:
        keys = jax.random.split(ka, images.shape[0])
        x = jax.vmap(lambda k, im: _randaugment(k, im, cfg.randaugment_layers,
                                      cfg.randaugment_magnitude))(keys, x)
    y = smoothed_targets(labels, classes, cfg.label_smoothing)
    if cfg.mixup_alpha or cfg.cutmix_alpha:
        permutation = jax.random.permutation(kp, x.shape[0])
        other, targets = x[permutation], y[permutation]
        mixed = coefficient * x + (1 - coefficient) * other
        mixed_y = coefficient * y + (1 - coefficient) * targets
        if cfg.cutmix_alpha:
            size = cfg.image_size
            side = jnp.int32(size * _sqrt(1 - coefficient))
            cy, cx = jax.random.randint(ky, (), 0, size), jax.random.randint(kx, (), 0, size)
            y0, y1 = jnp.maximum(0, cy - side // 2), jnp.minimum(size, cy + (side + 1) // 2)
            x0, x1 = jnp.maximum(0, cx - side // 2), jnp.minimum(size, cx + (side + 1) // 2)
            mask = ((jnp.arange(size)[:, None] >= y0) & (jnp.arange(size)[:, None] < y1) &
                    (jnp.arange(size)[None, :] >= x0) & (jnp.arange(size)[None, :] < x1))
            cut = jnp.where(mask[None, :, :, None], other, x)
            effective = 1.0 - (y1 - y0) * (x1 - x0) / float(size * size)
            cut_y = effective * y + (1 - effective) * targets
            choose_cut = mixing[1] > 0 if jnp.ndim(mixing) else (jax.random.bernoulli(km) if cfg.mixup_alpha else jnp.bool_(True))
            mixed, mixed_y = jnp.where(choose_cut, cut, mixed), jnp.where(choose_cut, cut_y, mixed_y)
        x, y = mixed, mixed_y
    return x, y
