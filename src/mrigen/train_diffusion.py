"""[STRETCH] Train the score model. See models/diffusion.py and section 9 of the plan.

Optional. Mirrors train_vae.py but with a denoising score-matching objective.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mrigen.data import FastMRISlices
from mrigen.models.diffusion import train_score_model


def train(
    root: str | Path = "data/processed",
    *,
    steps: int = 5000,
    batch_size: int = 8,
    lr: float = 2e-4,
    base_ch: int = 32,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    seed: int = 0,
    out: str | Path = "checkpoints/score.eqx",
):
    """Train a ScoreUNet on the training split and save the result.

    Args:
        root: directory of ``.npz`` shards (from ``data/preprocess.py``).
        steps, batch_size, lr, base_ch: training hyperparameters.
        beta_min, beta_max: VP-SDE schedule bounds.
        seed: PRNG seed.
        out: where to write the trained Equinox module (``eqx.tree_serialise_leaves``).
    """
    import equinox as eqx

    ds = FastMRISlices(root, normalize=True, split="train")
    print(f"training on {len(ds)} slices from {len(ds.volumes)} volumes")

    model = train_score_model(
        ds.slices,
        steps=steps,
        lr=lr,
        seed=seed,
        base_ch=base_ch,
        beta_min=beta_min,
        beta_max=beta_max,
    )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out, model)
    print(f"saved {out}")
    return model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a [STRETCH] score model.")
    p.add_argument("--root", default="data/processed")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--beta-max", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/score.eqx")
    return p.parse_args()


if __name__ == "__main__":
    train(**_parse_args().__dict__)