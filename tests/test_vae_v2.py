"""Contract-4 tests for the opt-in v2 architecture.

Same purity/gradability guarantees the default VAE must satisfy, because
anything that goes into NumPyro has to be a pure, differentiable z -> x map.
Also pins the thing that actually broke when v2 was first wired up:
make_decoder_fn used isinstance(model, VAE), which silently treated a VAE2 as
its own decoder and failed with "object is not callable".
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from mrigen.models.vae import VAE, make_decoder_fn
from mrigen.models.vae_v2 import VAE2

LATENT = 128


@pytest.mark.parametrize("cls", [VAE, VAE2])
def test_decoder_is_jittable_pure_fn(cls):
    decode = make_decoder_fn(cls(latent_dim=LATENT, key=jax.random.PRNGKey(0)))
    z = jnp.zeros(LATENT)
    img = jax.jit(decode)(z)
    assert img.shape == (128, 128)
    assert jnp.all(jnp.isfinite(img))
    assert jnp.all(img >= 0.0)                      # Contract 3: non-negative
    assert jnp.allclose(jax.jit(decode)(z), decode(z))


@pytest.mark.parametrize("cls", [VAE, VAE2])
def test_decoder_is_gradable_through_z(cls):
    decode = make_decoder_fn(cls(latent_dim=LATENT, key=jax.random.PRNGKey(0)))
    g = jax.grad(lambda z: jnp.sum(decode(z) ** 2))(jnp.zeros(LATENT))
    assert g.shape == (LATENT,)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("cls", [VAE, VAE2])
def test_make_decoder_fn_accepts_model_or_bare_decoder(cls):
    model = cls(latent_dim=LATENT, key=jax.random.PRNGKey(0))
    z = jnp.zeros(LATENT)
    assert jnp.allclose(make_decoder_fn(model)(z), make_decoder_fn(model.decoder)(z))


@pytest.mark.parametrize("cls", [VAE, VAE2])
def test_encoder_shapes_and_bounded_logvar(cls):
    model = cls(latent_dim=LATENT, key=jax.random.PRNGKey(0))
    # A deliberately extreme input, to check the logvar clamp holds.
    mu, logvar = model.encoder(jnp.full((128, 128), 50.0))
    assert mu.shape == (LATENT,) and logvar.shape == (LATENT,)
    assert jnp.all(jnp.isfinite(mu)) and jnp.all(jnp.isfinite(logvar))
    assert jnp.all(logvar >= -10.0) and jnp.all(logvar <= 10.0)
    # exp(logvar) must not overflow -- that is what the clamp is for
    assert jnp.all(jnp.isfinite(jnp.exp(logvar)))


def test_v2_is_smaller_than_the_default():
    def n(m):
        return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)))

    given = n(VAE(latent_dim=LATENT, key=jax.random.PRNGKey(0)))
    v2 = n(VAE2(latent_dim=LATENT, key=jax.random.PRNGKey(0)))
    assert v2 < given, (v2, given)
