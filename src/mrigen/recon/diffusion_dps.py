"""[STRETCH] Diffusion posterior sampling (DPS) for reconstruction.

Optional stretch goal. Alternates an unconditional reverse-diffusion step with a
data-consistency gradient step toward the measured k-space. Requires a trained
score model (models/diffusion.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mrigen.fourier import fft2c
from mrigen.models.diffusion import (
    ScoreUNet,
    vp_marginal_log_mean_coef,
    vp_marginal_std,
)


def _vp_reverse_step(x_t, t, t_prev, eps_pred, beta_min, beta_max):
    """DDIM-style deterministic reverse step for the VP-SDE."""
    a_t = jnp.exp(vp_marginal_log_mean_coef(t, beta_min, beta_max))
    a_p = jnp.exp(vp_marginal_log_mean_coef(t_prev, beta_min, beta_max))
    s_t = jnp.sqrt(1.0 - a_t ** 2)
    s_p = jnp.sqrt(1.0 - a_p ** 2)
    x0_hat = (x_t - s_t * eps_pred) / a_t
    return a_p * x0_hat + s_p * eps_pred


def dps_reconstruct(
    y_obs: jnp.ndarray,
    mask: jnp.ndarray,
    model: ScoreUNet,
    *,
    num_steps: int = 200,
    sigma_n: float = 0.05,
    zeta: float = 1.0,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    key,
) -> jnp.ndarray:
    """Diffusion posterior sampling for MRI reconstruction.

    At each reverse-diffusion step we
        1) predict eps with the score model and take a DDIM step to get x0_hat;
        2) compute the data-consistency gradient
               g = Re( F^H ( M * (F x0_hat - y_obs) ) ) / sigma_n^2
           and correct x0_hat by ``-zeta * g``;
        3) re-noise x0_hat to the next time level.

    Parameters
    ----------
    y_obs   : (H, W) complex k-space samples
    mask    : (H, W) 0/1 sampling mask
    model   : trained ScoreUNet predicting epsilon
    num_steps, sigma_n, zeta : sampler / guidance hyperparameters
    key     : PRNG key
    """
    shape = y_obs.shape
    x = jax.random.normal(key, shape)
    ts = jnp.linspace(1.0, 0.0, num_steps + 1)

    inv_var = 1.0 / (sigma_n ** 2)

    for i in range(num_steps):
        t = ts[i]
        t_prev = ts[i + 1]

        # 1) unconditional reverse step
        eps_pred = model(x[None], t)[0] if x.ndim == 2 else model(x, t)
        x0_hat = _vp_reverse_step(x, t, t_prev, eps_pred, beta_min, beta_max)

        # 2) data-consistency guidance toward measured k-space
        def dc_loss(x0):
            r = mask * (fft2c(x0) - y_obs)
            return 0.5 * inv_var * jnp.sum(jnp.abs(r) ** 2)

        grad = jax.grad(dc_loss)(x0_hat)
        x0_hat = x0_hat - zeta * grad

        # 3) re-noise x0_hat to t_prev
        a_p = jnp.exp(vp_marginal_log_mean_coef(t_prev, beta_min, beta_max))
        s_p = vp_marginal_std(t_prev, beta_min, beta_max)
        x = a_p * x0_hat + s_p * eps_pred

    return x