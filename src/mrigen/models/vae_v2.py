"""Alternative VAE architecture: fewer parameters, resize-conv decoder.

**Opt-in.** The default stays :class:`mrigen.models.vae.VAE`, because the
mentor's server checkpoints (``CHECKPOINTS.md``) were trained with it and
Equinox can only deserialise into the same layer structure -- and those
checkpoints, trained on the full fastMRI train split, beat anything trainable
on a laptop download by several dB. Use this one only when you are training
your own prior.

Three changes from the given architecture, each aimed at generalisation rather
than capacity (the measured failure mode on ~100 training slices is a ~4 dB
train/test autoencode gap, not a lack of capacity):

* one extra stride-2 stage so the dense head sees 4x4 instead of 8x8, cutting
  the ``fc`` layers from ``256*8*8 <-> 128`` to ``256*4*4 <-> 128``. Those
  layers held 6.3M of the given model's 7.7M parameters and were doing most of
  the memorising;
* GroupNorm after every conv, which lets the summed-ELBO gradients train
  stably from the first step;
* nearest-neighbour upsample + 3x3 conv in the decoder instead of stride-2
  ConvTranspose, which avoids the checkerboard pattern transposed convolutions
  leave behind.

**Measured A/B** (identical loss, 300 epochs, sigma_x=0.1, beta=1.0, same seed
and augmentation; 98 training slices; held-out volume ``file1000593``; MAP via
SVI 2000 steps; zero-filled 24.42 dB at R=4 and 24.08 dB at R=8):

=========  =========  ========  =======  =======  ========  =======  ========
arch       params     train AE  test AE  MAP R=4  +DC R=4   MAP R=8  +DC R=8
=========  =========  ========  =======  =======  ========  =======  ========
given      7,686,081  24.95     21.20    22.10    24.77     22.08    24.33
v2         4,304,577  24.29     20.33    23.71    25.33     23.50    24.73
=========  =========  ========  =======  =======  ========  =======  ========

Note the disagreement: ``given`` autoencodes better, ``v2`` *reconstructs*
better. MAP optimises z under a N(0, I) prior, so what matters is how
well-conditioned the z -> x map is, not raw reconstruction capacity. This is the
architecture-level version of the warning in ``CHECKPOINTS.md``: an
autoencoding score alone should not pick your model. Select on reconstruction
quality from undersampled k-space.

Usage::

    from mrigen.models.vae_v2 import VAE2
    from mrigen.train_vae import train
    model = train("data/processed", model_cls=VAE2, out="checkpoints/vae_v2.eqx")
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _upsample2(h: jnp.ndarray) -> jnp.ndarray:
    """Nearest-neighbour 2x upsample over the last two axes."""
    return jnp.repeat(jnp.repeat(h, 2, axis=-1), 2, axis=-2)


class Encoder2(eqx.Module):
    """Conv encoder: (1, 128, 128) image -> (mu, logvar) of size latent_dim."""

    convs: list
    norms: list
    head_mu: eqx.nn.Linear
    head_logvar: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        keys = jax.random.split(key, 8)
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4, channels 1->32->64->128->256->256
        chans = [(1, 32), (32, 64), (64, 128), (128, 256), (256, 256)]
        self.convs = [
            eqx.nn.Conv2d(i, o, 4, stride=2, padding=1, key=keys[n])
            for n, (i, o) in enumerate(chans)
        ]
        self.norms = [eqx.nn.GroupNorm(min(8, o), o) for _, o in chans]
        flat = 256 * 4 * 4
        self.head_mu = eqx.nn.Linear(flat, latent_dim, key=keys[5])
        self.head_logvar = eqx.nn.Linear(flat, latent_dim, key=keys[6])

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = x[None] if x.ndim == 2 else x
        for conv, norm in zip(self.convs, self.norms):
            h = jax.nn.gelu(norm(conv(h)))
        h = h.reshape(-1)
        # Same logvar bound as the given encoder: it passes through exp() in the
        # sample and the KL, and one unbounded step can NaN the run.
        return self.head_mu(h), jnp.clip(self.head_logvar(h), -10.0, 10.0)


class Decoder2(eqx.Module):
    """Latent z -> (128, 128) non-negative magnitude image, via resize-conv."""

    fc: eqx.nn.Linear
    convs: list
    norms: list
    out: eqx.nn.Conv2d

    def __init__(self, latent_dim: int, *, key):
        keys = jax.random.split(key, 8)
        self.fc = eqx.nn.Linear(latent_dim, 256 * 4 * 4, key=keys[0])
        chans = [(256, 256), (256, 128), (128, 64), (64, 32), (32, 32)]
        self.convs = [
            eqx.nn.Conv2d(i, o, 3, padding=1, key=keys[n + 1])
            for n, (i, o) in enumerate(chans)
        ]
        self.norms = [eqx.nn.GroupNorm(min(8, o), o) for _, o in chans]
        self.out = eqx.nn.Conv2d(32, 1, 3, padding=1, key=keys[6])

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = self.fc(z).reshape(256, 4, 4)
        for conv, norm in zip(self.convs, self.norms):
            h = jax.nn.gelu(norm(conv(_upsample2(h))))  # 4->8->16->32->64->128
        # softplus -> non-negative magnitude image; drop the channel axis
        return jax.nn.softplus(self.out(h))[0]


class VAE2(eqx.Module):
    """Drop-in alternative to :class:`mrigen.models.vae.VAE`."""

    encoder: Encoder2
    decoder: Decoder2
    latent_dim: int = eqx.field(static=True)

    def __init__(self, latent_dim: int = 128, *, key):
        ek, dk = jax.random.split(key)
        self.encoder = Encoder2(latent_dim, key=ek)
        self.decoder = Decoder2(latent_dim, key=dk)
        self.latent_dim = latent_dim
