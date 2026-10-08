"""A power-spectrum (stationary Gaussian) prior, learnt from the training slices.

The simplest data-driven prior: assume the image is a zero-mean stationary
Gaussian field. Its Fourier coefficients are independent with variance ``P(k)``
-- the power spectrum -- learnt from data as the mean ``|fft2c(x)|**2`` over
training slices.

Two ways to plug a model into this repo:

1. As a **decoder** for the existing NumPyro model. A sample from the prior is
   ``x = decode(w) = ifft2c(sqrt(P) * (w_re + i w_im)).real`` with ``w ~ N(0, I)``
   of shape ``(2, H, W)``. Hand ``make_spectrum_decoder(P)`` and
   ``latent_dim=(2, H, W)`` to ``reconstruct_map`` and MAP runs unchanged.
2. As a **standalone reconstructor** with the closed-form posterior -- the
   Wiener filter. With a Gaussian likelihood on observed k-space and a diagonal
   Gaussian prior in k-space, every frequency decouples. ``sigma`` is the noise
   std per real/imaginary component, so the complex noise variance is
   ``2 sigma^2`` and::

       observed   k:  mean = P / (P + 2 sigma^2) * y,   var = 2 P sigma^2 / (P + 2 sigma^2)
       unobserved k:  mean = 0,                         var = P
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mrigen.fourier import fft2c, ifft2c


def estimate_power_spectrum(images, eps: float = 1e-6) -> jnp.ndarray:
    """Mean ``|fft2c(x)|**2`` over a stack of ``(N, H, W)`` training images.

    A floor ``eps`` keeps every frequency strictly positive so the Wiener gain
    and the decoder are well defined everywhere.
    """
    k = fft2c(jnp.asarray(images, dtype=jnp.float32))
    return jnp.mean(jnp.abs(k) ** 2, axis=0) + eps


def make_spectrum_decoder(P):
    """Return ``decode(w) -> (H, W)`` real image for ``w ~ N(0, I)`` of shape ``(2, H, W)``.

    Amplitude ``sqrt(P)`` (not ``sqrt(P/2)``) so the *real part* has the training
    images' power: taking the real part halves the energy of a complex field
    with independent real/imaginary parts.
    """
    amp = jnp.sqrt(jnp.asarray(P))

    def decode(w):
        return ifft2c(amp * (w[0] + 1j * w[1])).real

    return decode


def wiener_reconstruct(y_obs, mask, P, sigma: float, *, n_samples: int = 0, key=None):
    """Closed-form Gaussian posterior under the power-spectrum prior.

    Args:
        y_obs: measured k-space ``(H, W)`` complex (zeros where unobserved).
        mask: ``{0, 1}`` sampling mask ``(H, W)``.
        P: power spectrum ``(H, W)`` from :func:`estimate_power_spectrum`.
        sigma: noise std per real/imag component of the measurement.
        n_samples: if > 1, also draw posterior samples (needs ``key``) and
            estimate the per-pixel std from them; with 0 or 1 samples the
            analytic (spatially constant) std is kept.

    Returns:
        dict with ``mean`` (H, W), ``std`` (H, W) and ``samples`` (N, H, W) or None.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    P = jnp.asarray(P)
    mask = jnp.asarray(mask)

    s2c = 2.0 * sigma ** 2                                  # complex noise variance
    gain = P / (P + s2c)                                    # Wiener shrinkage
    mean_k = mask * gain * y_obs                            # unobserved -> prior mean 0
    var_k = jnp.where(mask > 0, P * s2c / (P + s2c), P)     # per-frequency posterior var
    mean = ifft2c(mean_k).real

    # Real part of a unitary transform of independent complex Gaussians:
    # each pixel has (1/N) * sum(var_k) of complex variance, half survives .real.
    const_std = jnp.sqrt(jnp.sum(var_k) / (2.0 * var_k.size))
    std = jnp.full_like(mean, const_std)

    samples = None
    if n_samples > 1:                       # one sample gives std 0; keep analytic std
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        eps = (
            jax.random.normal(k1, (n_samples,) + mean.shape)
            + 1j * jax.random.normal(k2, (n_samples,) + mean.shape)
        )
        ks = mean_k[None] + jnp.sqrt(var_k / 2.0)[None] * eps
        samples = ifft2c(ks).real
        std = jnp.std(samples, axis=0)

    return {"mean": mean, "std": std, "samples": samples}