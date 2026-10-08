"""Training-loop tests on synthetic data (never touches patient data).

The schedule one is a regression test: warmup_cosine_decay_schedule treats
decay_steps as the *total* schedule length, so a warm-up of 5 epochs in a
3-epoch run asked optax for a negative decay length and raised. Long runs were
fine, which is exactly why it survived -- a student doing a quick `epochs=3`
sanity run would have hit it immediately.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from mrigen.data import augment
from mrigen.train_vae import load_model, train


@pytest.fixture
def tiny_data(tmp_path):
    """A shard of smooth synthetic 128x128 images in [0, 1]."""
    rng = np.random.default_rng(0)
    ys, xs = np.mgrid[0:128, 0:128].astype(np.float32)
    out = np.empty((40, 128, 128), dtype=np.float32)
    for i in range(40):
        cy, cx = rng.uniform(0.3, 0.7, 2) * 128
        w = rng.uniform(0.15, 0.3) * 128
        img = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * w**2))
        out[i] = img / img.max()
    np.savez_compressed(tmp_path / "synthvol.npz", slices=out)
    return tmp_path


def test_short_run_does_not_break_the_lr_schedule(tiny_data, tmp_path):
    # 3 epochs is fewer than the nominal 5 warm-up epochs.
    model = train(str(tiny_data), epochs=3, batch_size=8,
                  out=str(tmp_path / "m.eqx"), verbose_every=100)
    assert model.latent_dim == 128


def test_single_epoch_run(tiny_data, tmp_path):
    train(str(tiny_data), epochs=1, batch_size=8,
          out=str(tmp_path / "m1.eqx"), verbose_every=100)
    reloaded = load_model(str(tmp_path / "m1.eqx"), latent_dim=128)
    img = reloaded.decoder(jnp.zeros(128))
    assert img.shape == (128, 128)
    assert jnp.all(jnp.isfinite(img))


def test_augment_stays_in_range_and_changes_the_image():
    rng = np.random.default_rng(1)
    x = rng.random((128, 128)).astype(np.float32)
    outs = [augment(x, rng) for _ in range(10)]
    for a in outs:
        assert a.shape == (128, 128)
        assert a.dtype == np.float32
        assert a.min() >= 0.0 and a.max() <= 1.0   # Contract 2
    # augmentation must actually do something
    assert any(not np.allclose(a, x) for a in outs)
