# Checkpoints

Pre-trained model weights ship here so a team can start reconstructing even if
GPU training is slow. **These are model weights, not patient data**, and are
safe to redistribute under the fastMRI Data Sharing Agreement.

## `vae_128.eqx`

| Field | Value |
|-------|-------|
| Model | convolutional β-VAE (`mrigen.models.vae.VAE`) |
| Resolution | 128 × 128 magnitude |
| Latent dim | 128 |
| β | 1.0 |
| Training data | knee magnitude slices from `knee_singlecoil_val`, **train split only** — the volumes in `mrigen.data.HELDOUT_VOLUMES` (`file1000593`, `file1002067`, the first two in the archive) were excluded |
| Training data (server) | on the school server, `singlecoil_train` **train split only** (see `SERVER.md`); `singlecoil_val` is the untouched test set |
| Optimiser | Adam, lr 1e-3, gradient norm clipped at 1.0; batch 32, 50 epochs |
| Held-out recon PSNR / SSIM | server checkpoint, autoencoding (`decoder(encoder mean)`) on the server `test` split (all 7,135 `singlecoil_val` slices; SSIM on every 10th): **23.74 ± 2.39 dB / 0.578 ± 0.140** |

### β sweep (school server only)

Trained identically on the server `train` split. All four are in
`/shared/checkpoints/`, and `pixi run server-setup` copies them into your clone.
Load one with `load_model("checkpoints/vae_128_beta0.1.eqx", latent_dim=128)`.

| file | β | val PSNR (dB) | test PSNR (dB) | test SSIM |
|------|---|---------------|----------------|-----------|
| `vae_128.eqx` = `vae_128_beta1.0.eqx` | 1.0 | 23.73 | 23.74 | 0.578 |
| `vae_128_beta0.3.eqx` | 0.3 | 25.17 | 25.14 | 0.622 |
| `vae_128_beta0.1.eqx` | 0.1 | 26.18 | 26.10 | 0.651 |
| `vae_128_beta0.03.eqx` | 0.03 | 26.48 | 26.40 | 0.660 |

Smaller β reconstructs better but regularises the latent space less, and a
reconstruction *prior* needs that regularisation: an autoencoding score alone
should not pick β. Choose on `val` by reconstruction quality *from undersampled
k-space* (notebooks 03/04) and report `test` once.

Load it:

```python
from mrigen.train_vae import load_model
vae = load_model("checkpoints/vae_128.eqx", latent_dim=128)
```

Compare your own training run against these numbers — if you can beat them by
tuning β / latent dim / epochs, even better. Train with
`python -m mrigen.train_vae` (it uses `split="train"` by default) and **evaluate
only on `FastMRISlices(split="test")`** — see `06_evaluate_models.ipynb`.

## Locally trained `vae_128.eqx` (laptop download, 4 volumes)

What `pixi run lab` + notebook 01 actually produces from a 5-volume personal
download, for comparison against the server numbers above. **Not** a substitute
for them: the server checkpoints see the full `singlecoil_train` split and score
2.7-5.3 dB better held out. Prefer them when you have them.

| Field | Value |
|-------|-------|
| Architecture | `mrigen.models.vae.VAE` (the default; see `models/vae_v2.py` for the opt-in alternative) |
| Loss | summed-ELBO form with explicit `sigma_x` (`beta_upstream = 2 * sigma_x**2 * beta`) |
| Hyperparameters | latent_dim 128, beta 1.0, **sigma_x 0.1** (= upstream beta 0.02), 300 epochs, batch 16 |
| Optimiser | Adam, warm-up + cosine 1e-3, gradient norm clipped at 1.0 |
| Augmentation | flip / rotate / zoom / shift / gamma (`mrigen.data.augment`) |
| Training data | 98 slices (114 minus a 16-slice validation carve-out) from 3 volumes |
| Held-out volume | `file1000593` (36 slices). `file1002067`, the second held-out volume, was a **truncated download** here |
| Best validation | MSE 0.00358 at epoch 297 (24.46 dB) |

### What it scores

| metric | train | held out |
|--------|-------|----------|
| autoencode PSNR, `z = mu` | 25.65 dB | **20.98 dB** |
| autoencode PSNR, `z` sampled | 25.51 dB | 20.93 dB |
| KL | 254.6 nats (1.99/dim) | 251.2 nats (1.96/dim) |

The **+4.7 dB train/held-out gap is overfitting**, and it is a data limit, not a
hyperparameter one: ~100 training slices against a 7.7M-parameter model. More
volumes is the only real fix. Sample diversity (mean pairwise 1-SSIM over 16
samples) is 0.674 and a sample costs 2.7 ms.

**Check the KL before you trust any checkpoint.** `kl -> 0.0000` in the training
log means the posterior has collapsed: `q(z|x) = N(0, I)` for every x, z carries
no information, and every z decodes to the same average knee. A prior in that
state is useless for reconstruction while still showing a happily falling loss.
Notebook 01 ends with a health check that reports this.

### Honest reconstruction result

The prior **does not beat the classical baselines at this data scale.** On the
held-out slice with sigma = 0.01 measurement noise (notebook 03):

| method | PSNR R=4 | PSNR R=8 |
|--------|----------|----------|
| zero-filled | 23.80 | 23.35 |
| TV/L1 (FISTA, lam 1e-3) | **25.36** | **24.21** |
| VAE MAP | 20.77 | 20.77 |
| VAE MAP + data consistency | 23.47 | 22.99 |

Two things to say out loud rather than hide:

* **MAP scores the same at R=4 and R=8** (20.77 dB both). It has saturated
  against what the decoder can draw, so doubling the measurements changes
  nothing. `||z_map||` ~ 9 against the 11.3 expected of a N(0, I) draw in 128
  dimensions. The 21 dB autoencoding ceiling *is* the reconstruction ceiling.
* **Zero-filled is a strong baseline here.** The mask keeps a fully-sampled ACS
  band and a knee's energy is concentrated in those low frequencies, so there is
  not much for a prior to add at R=4-8. Data consistency recovers most of the
  gap precisely because it hands the measured lines back to the data.

### Architecture A/B

`models/vae_v2.py` is a smaller decoder (4.30M vs 7.69M parameters; resize-conv
instead of ConvTranspose, GroupNorm throughout). Trained identically, 300
epochs, noiseless measurements, 4 held-out slices, MAP via 2000 SVI steps:

| arch | params | train AE | held-out AE | MAP R=4 | +DC R=4 | MAP R=8 | +DC R=8 |
|------|--------|----------|-------------|---------|---------|---------|---------|
| `VAE` (default) | 7,686,081 | 24.95 | **21.20** | 22.10 | 24.77 | 22.08 | 24.33 |
| `VAE2` (opt-in) | 4,304,577 | 24.29 | 20.33 | **23.71** | **25.33** | **23.50** | **24.73** |
| *zero-filled* | - | - | - | - | *24.42* | - | *24.08* |

They disagree: `VAE` autoencodes better, `VAE2` *reconstructs* better. MAP
optimises z under a N(0, I) prior, so what matters is how well-conditioned the
z -> x map is, not raw capacity. `VAE` stays the default only because the
server checkpoints deserialise into it. This is the architecture-level version
of the warning above: **do not pick a model on its autoencoding score.**

## `score_128.eqx` *(stretch)*

Small UNet score model, same data and resolution. Only present if the diffusion
stretch goal was pre-trained.

> **Note:** checkpoints are produced by the mentor before the school and dropped
> into this directory. They are git-ignored by default (see `.gitignore`) so the
> repo stays small; the mentor distributes them out of band or via a release.
