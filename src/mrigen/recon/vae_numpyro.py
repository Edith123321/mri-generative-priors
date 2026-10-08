"""VAE-prior reconstruction in NumPyro: MAP (SVI) and posterior (NUTS).

Put a standard normal prior on the latent z, push it through the frozen decoder
to get an image, apply the forward operator, and place a Gaussian likelihood on
the *observed* k-space samples. Inference then turns measured k-space into a
posterior over images.

Equinox detail: we ``eqx.partition`` the decoder into arrays + static structure
and recombine inside the model, so ``decoder(z)`` is a pure function that
JIT/NUTS can trace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

from mrigen.fourier import fft2c
from mrigen.models.vae import make_decoder_fn


def recon_model(y_obs, mask, decode, latent_dim, sigma):
    """NumPyro model: z ~ N(0, I); x = decode(z); Gaussian likelihood on k-space.

    Steps:
        1) sample ``z`` from a standard Normal of size ``latent_dim``;
        2) decode it to an image ``x``;
        3) form the forward measurement ``k = mask * fft2c(x)``;
        4) observe the real and imaginary parts of the k-space with a
           Normal(., sigma) likelihood, restricted to the sampled locations.

    The mask is a *traced* array under NUTS/SVI, so boolean indexing
    (``k.real[obs]``) raises ``NonConcreteBooleanIndexError`` -- use
    ``dist.Normal(...).mask(obs)`` over the full array instead.
    """
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), 1.0))
    x = decode(z)
    k = mask * fft2c(x)
    obs = mask.astype(bool)
    numpyro.sample("y_re", dist.Normal(k.real, sigma).mask(obs), obs=y_obs.real)
    numpyro.sample("y_im", dist.Normal(k.imag, sigma).mask(obs), obs=y_obs.imag)


def reconstruct_map(
    y_obs, mask, decoder, latent_dim, sigma=0.01, *, steps=1000, lr=1e-2, seed=0,
    progress_bar=True,
):
    """MAP reconstruction via SVI + AutoDelta.

    Returns ``(image, z_map)``.
    """
    decode = make_decoder_fn(decoder)
    guide = autoguide.AutoDelta(recon_model)
    svi = SVI(recon_model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    result = svi.run(
        jax.random.PRNGKey(seed), steps, y_obs, mask, decode, latent_dim, sigma,
        progress_bar=progress_bar,
    )
    z_map = result.params["z_auto_loc"]
    return decode(z_map), z_map


def reconstruct_posterior(
    y_obs,
    mask,
    decoder,
    latent_dim,
    sigma=0.01,
    *,
    num_samples=200,
    num_warmup=200,
    seed=0,
    max_tree_depth=10,
    progress_bar=True,
):
    """Posterior reconstruction via NUTS over z, with pixel-wise uncertainty.

    Keep ``latent_dim`` around 128-256 so the sampler mixes. Returns a dict with
    ``mean`` and ``std`` images (the std map is the uncertainty) and the raw
    image ``samples``. ``max_tree_depth`` bounds the leapfrog steps per sample
    (NumPyro's default is 10, i.e. up to 1023 decoder evaluations per sample);
    lower it to 6-7 on a CPU, at some cost in mixing.
    """
    decode = make_decoder_fn(decoder)
    kernel = NUTS(recon_model, max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), y_obs, mask, decode, latent_dim, sigma)
    zs = mcmc.get_samples()["z"]
    images = jax.vmap(decode)(zs)
    return {
        "mean": jnp.mean(images, axis=0),
        "std": jnp.std(images, axis=0),
        "samples": images,
        "mcmc": mcmc,
    }