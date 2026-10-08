"""Classical reconstruction baselines.

``zero_filled`` -- the adjoint of the measurement, i.e. the naive inverse FFT
of the zero-filled k-space. This is the baseline every learned method must beat.

``tv_fista`` -- FISTA for TV-regularised compressed sensing. Solves

    min_x  1/2 || M F x - y ||_2^2  +  lam * TV(x)

where F = fft2c is the (orthonormal) 2-D Fourier transform, M is the sampling
mask, and TV is isotropic total variation. The proximal operator for TV is
computed with Chambolle's dual algorithm.
"""

from __future__ import annotations

import jax.numpy as jnp

from mrigen.fourier import fft2c, ifft2c


def zero_filled(y_obs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Zero-filled reconstruction: real part of ifft2c of the masked k-space."""
    return ifft2c(mask * y_obs).real


def _grad(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward finite differences (Neumann boundary): (dx, dy)."""
    dx = jnp.zeros_like(x).at[:-1, :].set(x[1:, :] - x[:-1, :])
    dy = jnp.zeros_like(x).at[:, :-1].set(x[:, 1:] - x[:, :-1])
    return dx, dy


def _div(px: jnp.ndarray, py: jnp.ndarray) -> jnp.ndarray:
    """Discrete divergence: the *negative* adjoint of ``_grad``.

    ``<grad(x), p> == -<x, div(p)>``, i.e. ``div = -grad^T``, which is the usual
    convention and the one Chambolle's algorithm below is written for. Mind the
    sign: taking this for ``+grad^T`` is what makes the prox run uphill.
    """
    dxx = jnp.zeros_like(px).at[1:, :].set(px[:-1, :]) - px
    dxx = dxx.at[0, :].set(-px[0, :])
    dyy = jnp.zeros_like(py).at[:, 1:].set(py[:, :-1]) - py
    dyy = dyy.at[:, 0].set(-py[:, 0])
    return -(dxx + dyy)


def _tv_prox(b: jnp.ndarray, weight: float, n_iter: int = 20) -> jnp.ndarray:
    """Prox of ``weight * TV`` via Chambolle's dual algorithm.

    Solves ``argmin_x  1/2 ||x - b||^2 + weight * TV(x)`` through its dual,

        p* = argmin_{|p| <= 1} || weight * div(p) - b ||^2,
        x  = b - weight * div(p*),

    by projected gradient on ``p``. With ``x = b - weight * div(p)`` the dual
    gradient works out to ``weight * grad(x)``, so the step is

        p <- proj( p - (tau / weight) * grad(x) )

    -- note the **minus**. With a plus the iteration ascends the dual objective
    and the prox returns something *worse* than its input, diverging further the
    larger ``weight`` gets; downstream that shows up as a TV reconstruction with
    negative PSNR. ``p`` starts at zero, which is inside the feasible set.
    """
    if weight <= 0:
        return b
    px = jnp.zeros_like(b)
    py = jnp.zeros_like(b)
    tau = 0.25  # Chambolle step; stable for 1/8 <= tau <= 1/4 in 2-D
    for _ in range(n_iter):
        x = b - weight * _div(px, py)
        gx, gy = _grad(x)
        px_new = px - tau * gx / weight
        py_new = py - tau * gy / weight
        # project each pixel's dual vector onto the unit l2 ball
        norm = jnp.maximum(1.0, jnp.sqrt(px_new ** 2 + py_new ** 2))
        px, py = px_new / norm, py_new / norm
    return b - weight * _div(px, py)


def tv_fista(
    y_obs: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    lam: float = 1e-2,
    n_iter: int = 50,
    step: float = 1.0,
    inner_iter: int = 20,
) -> jnp.ndarray:
    """TV-regularised CS reconstruction via FISTA.

    Minimises ``1/2 ||M F x - y||^2 + lam * TV(x)``. ``step = 1.0`` is safe
    because F is orthonormal and M is a projection.
    """
    x = zero_filled(y_obs, mask)
    z = x
    t = 1.0

    def data_grad(x_):
        return ifft2c(mask * (fft2c(x_) - y_obs)).real

    for _ in range(n_iter):
        v = z - step * data_grad(z)
        x_new = _tv_prox(v, weight=step * lam, n_iter=inner_iter)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new

    return x