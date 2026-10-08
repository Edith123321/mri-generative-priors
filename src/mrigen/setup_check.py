"""Pre-flight check: print JAX devices and run a 1-step VAE forward.

Run this before lunch on Monday so a group knows the GPU works:

    pixi run check

It does NOT need any data -- it makes a random 128x128 image and pushes it
through a freshly-initialised VAE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def main() -> None:
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())

    from mrigen.models.vae import VAE, vae_loss

    key = jax.random.PRNGKey(0)
    mk, xk, lk = jax.random.split(key, 3)
    model = VAE(latent_dim=128, key=mk)
    x = jax.random.uniform(xk, (128, 128))

    # Encoder
    mu, logvar = model.encoder(x)
    assert mu.shape == (128,), f"mu shape {mu.shape}"
    assert logvar.shape == (128,), f"logvar shape {logvar.shape}"
    print(f"encoder OK: mu {mu.shape}, logvar {logvar.shape}")

    # Full forward + loss
    loss, (recon, kl) = vae_loss(model, x, lk, beta=1.0)
    assert jnp.isfinite(loss), f"non-finite loss: {loss}"
    print(
        f"1-step VAE forward OK: loss={float(loss):.4f} "
        f"recon={float(recon):.4f} kl={float(kl):.4f}"
    )

    # Decoder is a pure function of z (what NumPyro needs)
    z = jnp.zeros(128)
    img = model.decoder(z)
    assert img.shape == (128, 128), f"decoder output shape {img.shape}"
    assert jnp.all(img >= 0), "decoder output must be non-negative (softplus)"
    print(
        f"decoder OK: image {img.shape}, "
        f"min {float(img.min()):.3f} max {float(img.max()):.3f}"
    )
    print("setup check complete.")


if __name__ == "__main__":
    main()