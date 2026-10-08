"""Forward operator, adjoint, and data-consistency projection.

Short, central, conceptual -- this is the heart of the inverse problem. The
measurement model is

    A(x) = M . F(x)          (undersampled k-space of image x)

where F = fourier.fft2c (centred orthonormal FFT) and M is the binary Cartesian
mask from masks.py. The adjoint maps k-space back to image space, and the
data-consistency (DC) projection forces the reconstruction to agree with the
*measured* k-space samples while keeping the model's estimate everywhere else.

Everything in this file is JAX -- ``jnp`` and the given ``fft2c`` / ``ifft2c``
are imported below, and that is all that is needed.
"""

from __future__ import annotations

import jax.numpy as jnp

from mrigen.fourier import fft2c, ifft2c


def forward(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """A(x) = mask * fft2c(x): image -> undersampled k-space.

    Transform the whole image to k-space, then throw away the lines the scanner
    did not acquire. Zeroing rather than deleting keeps the shape at (H, W),
    which is what makes the operator easy to compose and to differentiate.
    """
    return mask * fft2c(x)


def adjoint(k: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """A^H(k) = ifft2c(mask * k): undersampled k-space -> image.

    The adjoint of ``forward``, not its inverse: ``A`` destroys the unmeasured
    lines, and no operator can put them back. Applied to the measurement it
    gives the zero-filled image, aliasing and all. Returned complex; take
    ``.real`` on the magnitude path.
    """
    return ifft2c(mask * k)


def data_consistency(x_est: jnp.ndarray, y_obs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Replace estimated k-space with measured samples where the mask is on.

    Returns the image
        ifft2c( mask * y_obs + (1 - mask) * fft2c(x_est) )
    i.e. keep the measured entries, trust the model elsewhere.

    This is the one place the measurement gets the last word: wherever the
    scanner actually sampled, the estimate is overwritten by the data, so no
    amount of prior confidence can talk the reconstruction out of what was
    measured. Everywhere else the estimate is all we have, so it is kept.
    """
    k_est = fft2c(x_est)
    k_dc = mask * y_obs + (1.0 - mask) * k_est
    return ifft2c(k_dc).real
