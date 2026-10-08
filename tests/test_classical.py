"""Tests for the classical TV baseline.

These pin the two things that were silently wrong once: the sign of the
divergence relative to ``_grad``, and that the TV prox actually *decreases* the
objective it claims to minimise. A prox that runs uphill still returns an image,
so nothing crashes -- the TV row of the results table just quietly becomes
nonsense (negative PSNR). Hence the tests.
"""

import jax.numpy as jnp
import numpy as np

from mrigen.fourier import fft2c
from mrigen.metrics import psnr
from mrigen.recon.classical import _div, _grad, _tv_prox, tv_fista, zero_filled


def _tv(z):
    dx, dy = _grad(z)
    return float(jnp.sum(jnp.sqrt(dx**2 + dy**2)))


def test_div_is_negative_adjoint_of_grad():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((12, 12)))
    # The algorithm keeps the last row/column of p at zero; _grad is zero there.
    px = jnp.asarray(rng.standard_normal((12, 12))).at[-1, :].set(0.0)
    py = jnp.asarray(rng.standard_normal((12, 12))).at[:, -1].set(0.0)

    dx, dy = _grad(x)
    assert np.isclose(
        float(jnp.sum(dx * px) + jnp.sum(dy * py)),
        float(-jnp.sum(x * _div(px, py))),
        rtol=1e-5,
    )


def test_tv_prox_decreases_its_objective():
    rng = np.random.default_rng(1)
    b = jnp.asarray(rng.standard_normal((32, 32)))

    def obj(z, weight):
        return 0.5 * float(jnp.sum((z - b) ** 2)) + weight * _tv(z)

    for weight in (0.05, 0.2, 1.0):
        out = _tv_prox(b, weight, n_iter=100)
        assert obj(out, weight) < obj(b, weight)
        # and it is a *smoothing* operator: more weight, less total variation
        assert _tv(out) < _tv(b)


def test_tv_prox_is_identity_for_zero_weight():
    b = jnp.asarray(np.random.default_rng(2).standard_normal((16, 16)))
    assert jnp.allclose(_tv_prox(b, 0.0), b)


def test_tv_fista_beats_zero_filled_on_a_piecewise_constant_image():
    # TV's home turf: a piecewise-constant phantom undersampled 2x in columns.
    x = np.zeros((64, 64), dtype=np.float32)
    x[16:48, 16:48] = 1.0
    x[28:36, 28:36] = 0.5
    mask = jnp.zeros((64, 64)).at[:, ::2].set(1.0).at[:, 28:36].set(1.0)

    y = mask * fft2c(jnp.asarray(x))
    zf = np.asarray(zero_filled(y, mask))
    tv = np.asarray(tv_fista(y, mask, lam=1e-3, n_iter=100))

    assert np.all(np.isfinite(tv))
    assert psnr(x, tv, data_range=1.0) > psnr(x, zf, data_range=1.0)
