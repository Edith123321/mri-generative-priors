"""Cartesian undersampling masks.

A Cartesian MRI scan acquires k-space one phase-encode line (one column) at a
time, so we accelerate by *skipping columns*. We always keep a fully-sampled
block of central columns -- the auto-calibration signal (ACS) -- because the
low frequencies carry most of the image energy and contrast.

``equispaced_mask`` keeps every R-th column plus an ACS band.
``random_mask`` keeps a random subset of columns plus an ACS band, choosing the
count so the *overall* acceleration is approximately R.

Both return a real {0., 1.} array of shape ``(H, W)`` where whole columns are
on or off. Masks are built once on the host, so randomness uses plain NumPy;
the return value is a JAX array.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def acs_columns(width: int, acs_frac: float) -> np.ndarray:
    """Indices of the central ACS columns.

    Returns the integer column indices of the fully-sampled centre band.
    """
    n_acs = max(1, round(acs_frac * width))
    start = (width - n_acs) // 2
    return np.arange(start, start + n_acs)


def _with_acs(mask: jnp.ndarray, width: int, acs_frac: float) -> jnp.ndarray:
    """Turn on the central ACS columns of a (H, W) column mask in place."""
    n_acs = max(1, int(round(acs_frac * width)))
    start = (width - n_acs) // 2
    return mask.at[:, start : start + n_acs].set(1.0)


def equispaced_mask(
    shape: tuple[int, int],
    acceleration: int,
    acs_frac: float = 0.08,
) -> jnp.ndarray:
    """Equispaced Cartesian mask with a central ACS band.

    Args:
        shape: (H, W) of the k-space image.
        acceleration: R; keep roughly every R-th phase-encode column.
        acs_frac: fraction of columns kept fully-sampled in the centre.

    Returns:
        (H, W) float array of 0./1.; entire columns are on or off.
    """
    H, W = shape
    mask = jnp.zeros((H, W), jnp.float32)
    mask = mask.at[:, ::acceleration].set(1.0)
    return _with_acs(mask, W, acs_frac)


def random_mask(
    shape: tuple[int, int],
    acceleration: int,
    acs_frac: float = 0.08,
    seed: int = 0,
) -> jnp.ndarray:
    """Random Cartesian mask with a central ACS band.

    Keeps the central ACS columns and then draws additional columns at random
    so that the *total* fraction of kept columns is approximately
    ``1 / acceleration``.

    Args:
        shape: (H, W) of the k-space image.
        acceleration: target overall R.
        acs_frac: fraction of columns kept fully-sampled in the centre.
        seed: RNG seed for reproducibility.

    Returns:
        (H, W) float array of 0./1.; entire columns are on or off.
    """
    H, W = shape
    rng = np.random.default_rng(seed)

    # Target number of kept columns for the requested overall acceleration.
    target = max(1, int(round(W / acceleration)))

    # ACS columns are always kept; they count toward the target.
    acs = acs_columns(W, acs_frac)
    n_extra = max(0, target - len(acs))

    # Draw the extra columns from the complement of the ACS band.
    candidates = np.setdiff1d(np.arange(W), acs, assume_unique=True)
    if n_extra >= len(candidates):
        extra = candidates
    else:
        extra = rng.choice(candidates, size=n_extra, replace=False)

    kept = np.union1d(acs, extra)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, kept] = 1.0
    return jnp.asarray(mask)