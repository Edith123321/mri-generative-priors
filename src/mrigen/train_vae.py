"""beta-VAE training loop with checkpointing.

Students choose hyperparameters (beta, latent_dim, epochs, lr) but the loop,
batching, and checkpoint I/O are provided. If training is slow, skip this and
load the pre-trained ``checkpoints/vae_128.eqx`` instead (see CHECKPOINTS.md).

Usage:
    python -m mrigen.train_vae --data data/processed --epochs 50 --beta 1.0

Trains on ``split="train"`` by default, i.e. without the held-out volumes in
``mrigen.data.HELDOUT_VOLUMES`` -- so that ``FastMRISlices(split="test")`` is a
fair evaluation set for the resulting checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mrigen.data import FastMRISlices, data_loader
from mrigen.models.vae import VAE, vae_loss


def save_model(path: str | Path, model: VAE) -> None:
    """Serialise an Equinox model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)


def load_model(path: str | Path, latent_dim: int = 128) -> VAE:
    """Load an Equinox VAE; rebuilds the skeleton then fills in saved leaves."""
    skeleton = VAE(latent_dim=latent_dim, key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(str(path), skeleton)


@eqx.filter_jit
def _train_step(model, opt_state, batch, key, optim, beta, sigma_x):
    def batched_loss(m):
        keys = jax.random.split(key, batch.shape[0])
        losses, aux = jax.vmap(lambda x, k: vae_loss(m, x, k, beta, sigma_x))(batch, keys)
        mse, kl = aux
        return jnp.mean(losses), (jnp.mean(mse), jnp.mean(kl))

    (loss, aux), grads = eqx.filter_value_and_grad(batched_loss, has_aux=True)(model)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


@eqx.filter_jit
def _eval_mse(model, batch):
    """Per-pixel MSE of the deterministic reconstruction (z = mu, no sampling).

    Evaluation should not depend on a random draw, so we decode the posterior
    *mean* rather than a sample. This is the number to watch for overfitting.
    """
    def one(x):
        mu, _ = model.encoder(x)
        return jnp.mean((model.decoder(mu) - x) ** 2)

    return jnp.mean(jax.vmap(one)(batch))


def train(
    data_dir: str,
    *,
    latent_dim: int = 128,
    beta: float = 1.0,
    sigma_x: float = 0.1,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    out: str = "checkpoints/vae_128.eqx",
    split: str | None = "train",
    augment_data: bool = True,
    val_frac: float = 0.1,
    verbose_every: int = 10,
) -> VAE:
    """Train the beta-VAE prior. ``split="train"`` keeps the held-out volumes out.

    A fraction ``val_frac`` of the training slices is set aside to choose *when*
    to stop: the checkpoint written is the one with the best validation MSE, not
    the last epoch. Those validation slices come from the training volumes, so
    they are correlated with what the model saw and the number is optimistic --
    they are good enough to pick an epoch, and nothing more. Report real numbers
    on ``split="test"``, which no part of training touches.
    """
    key = jax.random.PRNGKey(seed)
    model_key, key = jax.random.split(key)
    model = VAE(latent_dim=latent_dim, key=model_key)

    dataset = FastMRISlices(data_dir, split=split)

    # Validation slices for early stopping, held out of the gradient updates.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(dataset))
    n_val = max(batch_size, round(val_frac * len(dataset))) if val_frac > 0 else 0
    val_imgs = dataset.slices[perm[:n_val]]
    train_imgs = dataset.slices[perm[n_val:]]

    print(
        f"training on {len(train_imgs)} slices from {len(dataset.volumes)} volume(s) "
        f"(split={split!r}), validating on {len(val_imgs)}",
        flush=True,
    )
    print(
        f"  latent_dim={latent_dim} beta={beta} sigma_x={sigma_x} "
        f"epochs={epochs} batch={batch_size} lr={lr} augment={augment_data}",
        flush=True,
    )

    steps_per_epoch = max(1, len(train_imgs) // batch_size)
    total_steps = max(2, epochs * steps_per_epoch)
    # Warm up then cosine-decay: the first epochs of a VAE are unstable because
    # the encoder and decoder are chasing each other, and decaying at the end
    # buys a visibly sharper decoder. optax wants decay_steps to be the *total*
    # schedule length, so the warm-up has to stay well inside it -- cap it at a
    # tenth of the run, or a 3-epoch smoke test asks for 30 warm-up steps out of
    # 18 and optax raises on a negative decay length.
    warmup_steps = max(1, min(5 * steps_per_epoch, total_steps // 10))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr / 10,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr / 50,
    )
    # Clipping matters here: the summed reconstruction term makes early
    # gradients large, and one bad step can push the decoder into a dead region
    # of softplus from which it does not recover.
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    best_val = float("inf")
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(epochs):
        ep_loss = ep_mse = ep_kl = 0.0
        n = 0
        for batch in data_loader(
            train_imgs, batch_size, seed=seed + epoch, augment_data=augment_data
        ):
            key, sk = jax.random.split(key)
            batch = jnp.asarray(batch)
            model, opt_state, loss, (mse, kl) = _train_step(
                model, opt_state, batch, sk, optim, beta, sigma_x
            )
            ep_loss += float(loss)
            ep_mse += float(mse)
            ep_kl += float(kl)
            n += 1

        val_mse = float(_eval_mse(model, jnp.asarray(val_imgs))) if n_val else float("nan")
        history.append(
            {
                "epoch": epoch,
                "loss": ep_loss / n,
                "mse": ep_mse / n,
                "kl": ep_kl / n,
                "val_mse": val_mse,
            }
        )

        if n_val and val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            save_model(out, model)

        if epoch % verbose_every == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:4d}  loss {ep_loss / n:10.1f}  "
                f"mse {ep_mse / n:.5f}  kl {ep_kl / n:7.2f}  "
                f"val_mse {val_mse:.5f}" + ("  *" if best_epoch == epoch else ""),
                flush=True,
            )

    if n_val:
        print(
            f"best val_mse {best_val:.5f} at epoch {best_epoch} "
            f"(PSNR {10 * np.log10(1.0 / best_val):.2f} dB) -> {out}",
            flush=True,
        )
        model = load_model(out, latent_dim=latent_dim)
    else:
        save_model(out, model)
        print(f"saved checkpoint -> {out}")

    model_history = history
    train.history = model_history  # type: ignore[attr-defined]
    return model


def main():
    p = argparse.ArgumentParser(description="Train the beta-VAE prior.")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--sigma-x", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", default="checkpoints/vae_128.eqx")
    p.add_argument(
        "--split", default="train",
        help="'train' (default), 'val', 'test', or 'all'",
    )
    args = p.parse_args()
    train(
        args.data,
        latent_dim=args.latent_dim,
        beta=args.beta,
        sigma_x=args.sigma_x,
        epochs=args.epochs,
        augment_data=not args.no_augment,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
        split=None if args.split == "all" else args.split,
    )


if __name__ == "__main__":
    main()