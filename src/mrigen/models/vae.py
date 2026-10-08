"""Convolutional beta-VAE over 128x128 magnitude slices.

The encoder/decoder architecture is as given. ``reparameterise`` is the one
line that makes the whole thing trainable: sampling z directly is not
differentiable, so we sample epsilon ~ N(0, I) and form z = mu + sigma * eps so
gradients flow through mu and sigma.

The weighting of the two ELBO terms is where this model lives or dies -- see
``vae_loss`` for why both terms are sums, not means, and what ``sigma_x`` and
``beta`` actually trade off.

The decoder is a pure function of z once parameters are frozen, which is exactly
what NumPyro needs (see recon/vae_numpyro.py).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class Encoder(eqx.Module):
    """Conv encoder: (1, 128, 128) image -> (mu, logvar) of size latent_dim."""

    layers: list
    head_mu: eqx.nn.Linear
    head_logvar: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        keys = jax.random.split(key, 6)
        # 128 -> 64 -> 32 -> 16 -> 8, channels 1->32->64->128->256
        self.layers = [
            eqx.nn.Conv2d(1, 32, 4, stride=2, padding=1, key=keys[0]),
            eqx.nn.Conv2d(32, 64, 4, stride=2, padding=1, key=keys[1]),
            eqx.nn.Conv2d(64, 128, 4, stride=2, padding=1, key=keys[2]),
            eqx.nn.Conv2d(128, 256, 4, stride=2, padding=1, key=keys[3]),
        ]
        flat = 256 * 8 * 8
        self.head_mu = eqx.nn.Linear(flat, latent_dim, key=keys[4])
        self.head_logvar = eqx.nn.Linear(flat, latent_dim, key=keys[5])

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = x[None] if x.ndim == 2 else x  # ensure leading channel dim
        for conv in self.layers:
            h = jax.nn.gelu(conv(h))
        h = h.reshape(-1)
        return self.head_mu(h), self.head_logvar(h)


class Decoder(eqx.Module):
    """Latent z -> (128, 128) non-negative magnitude image.

    Pure function of z once frozen -> drops straight into NumPyro.
    """

    fc: eqx.nn.Linear
    layers: list

    def __init__(self, latent_dim: int, *, key):
        keys = jax.random.split(key, 5)
        self.fc = eqx.nn.Linear(latent_dim, 256 * 8 * 8, key=keys[0])
        self.layers = [
            eqx.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, key=keys[1]),
            eqx.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, key=keys[2]),
            eqx.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, key=keys[3]),
            eqx.nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, key=keys[4]),
        ]

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = self.fc(z).reshape(256, 8, 8)
        for conv in self.layers[:-1]:
            h = jax.nn.gelu(conv(h))
        h = self.layers[-1](h)
        # softplus -> non-negative magnitude image; drop channel axis
        return jax.nn.softplus(h)[0]


class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    latent_dim: int = eqx.field(static=True)

    def __init__(self, latent_dim: int = 128, *, key):
        ek, dk = jax.random.split(key)
        self.encoder = Encoder(latent_dim, key=ek)
        self.decoder = Decoder(latent_dim, key=dk)
        self.latent_dim = latent_dim


def reparameterise(mu: jnp.ndarray, logvar: jnp.ndarray, key) -> jnp.ndarray:
    """Sample z ~ N(mu, sigma^2) differentiably (the reparameterisation trick).

    Drawing z straight from ``N(mu, sigma^2)`` is not differentiable in mu and
    sigma -- the randomness sits between them and the loss. Instead draw the
    randomness from a *fixed* distribution, ``eps ~ N(0, I)``, and push it
    through a deterministic map: ``z = mu + sigma * eps``. Now eps is just a
    constant as far as autodiff is concerned, so gradients flow into mu and
    sigma and the encoder is trainable.

    We store ``logvar = log(sigma^2)`` rather than sigma because it is
    unconstrained (any real number is a valid logvar, whereas sigma must stay
    positive), so ``sigma = exp(0.5 * logvar)``.
    """
    sigma = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, mu.shape)
    return mu + sigma * eps


def vae_loss(
    model: VAE,
    x: jnp.ndarray,
    key,
    beta: float = 1.0,
    sigma_x: float = 0.1,
):
    """beta-VAE negative ELBO for a single image x of shape (128, 128).

    Both terms are **sums over their own axes**, and that is what makes this an
    ELBO: a Gaussian log-density summed over the 16384 pixels, plus a KL summed
    over the latent dimensions.

    Averaging each term instead -- ``jnp.mean`` on both, which looks harmless --
    silently reweights the KL by ``n_pixels / latent_dim`` (128x at 128^2 with
    a 128-d latent). beta = 1 then behaves like beta = 128, the posterior
    collapses to the prior, the KL falls to 0.0000, and z stops carrying any
    information about x: every z decodes to the same blurry mean image, which
    makes the "learned prior" useless for reconstruction. If you see the KL go
    to zero in the training log, this is why.

    ``sigma_x`` is the assumed per-pixel observation noise in the decoder's
    Gaussian likelihood, and it sets the real trade-off: a small ``sigma_x``
    trusts the pixels and gives sharp samples with a looser latent, a large one
    trusts the prior and gives smooth samples. ``beta`` then scales the KL on
    top of that, as in the beta-VAE paper.

    Returns ``(loss, (mse, kl))``: ``mse`` is the plain per-pixel
    reconstruction error (comparable across runs and hyperparameters) and
    ``kl`` is the summed KL in nats.
    """
    mu, logvar = model.encoder(x)
    z = reparameterise(mu, logvar, key)
    x_hat = model.decoder(z)

    # Reported separately from the loss so it stays interpretable: the error of
    # one pixel, in the [0, 1] units of the image.
    mse = jnp.mean((x_hat - x) ** 2)

    # -log p(x | z) for N(x_hat, sigma_x^2 I), summed over pixels (constants dropped).
    recon = 0.5 * jnp.sum((x_hat - x) ** 2) / sigma_x**2
    # Closed-form KL[N(mu, sigma^2) || N(0, I)], summed over latent dims.
    kl = -0.5 * jnp.sum(1.0 + logvar - mu**2 - jnp.exp(logvar))

    loss = recon + beta * kl
    return loss, (mse, kl)


def make_decoder_fn(model):
    """Return a pure ``z -> x`` function with the decoder's parameters baked in.

    GIVEN. Inference (NumPyro / `jax.jit` / `jax.grad`) needs a *pure* function
    it can trace and differentiate. We split the decoder into arrays (its trained
    parameters) and static structure with ``eqx.partition``, then recombine
    inside the closure with ``eqx.combine`` (CLAUDE.md Contract 4). The returned
    callable closes over plain arrays only, so it is safe for JIT/grad/NUTS.

    Accepts the full :class:`VAE` (preferred, per the contract) or a bare
    :class:`Decoder`.
    """
    decoder = model.decoder if isinstance(model, VAE) else model
    params, static = eqx.partition(decoder, eqx.is_array)

    def decode(z: jnp.ndarray) -> jnp.ndarray:
        return eqx.combine(params, static)(z)

    return decode
