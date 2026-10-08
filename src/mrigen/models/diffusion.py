"""[STRETCH] Small UNet score model + VP-SDE utilities.

Optional stretch goal (see notebooks/05_diffusion_stretch.ipynb). Only attempt
after the VAE pipeline (train -> MAP -> posterior) works end to end. Provided as
a thin skeleton; teams that take this on flesh out the UNet blocks and the SDE
noise schedule.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------------
# Time embedding: sinusoidal features followed by a small MLP
# ----------------------------------------------------------------------------
def sinusoidal_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Map scalar t (or batch) to a ``dim``-vector of sinusoidal features.

    Standard transformer/DDPM embedding: for frequencies
    ``f_i = 2 ** (-i / (dim/2 - 1))`` we return
    ``[sin(2*pi*f_i*t), cos(2*pi*f_i*t)]`` concatenated along the last axis.
    ``t`` should be a scalar or shape ``(B,)`` in ``[0, 1]``.
    """
    half = dim // 2
    freqs = 2.0 ** (-jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1))
    args = 2.0 * jnp.pi * t[..., None] * freqs  # (..., half)
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class TimeEmbed(eqx.Module):
    """Sinusoidal embedding -> 2-layer MLP, output dim = ``dim``."""

    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, *, key):
        k1, k2 = jax.random.split(key)
        self.lin1 = eqx.nn.Linear(dim, dim, key=k1)
        self.lin2 = eqx.nn.Linear(dim, dim, key=k2)
        self.dim = dim

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        h = sinusoidal_embedding(t, self.dim)
        h = jax.nn.gelu(self.lin1(h))
        return self.lin2(h)


# ----------------------------------------------------------------------------
# Building blocks: a residual conv block and a self-attention block
# ----------------------------------------------------------------------------
class ResBlock(eqx.Module):
    """Pre-activation residual block with FiLM-style time conditioning."""

    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    time_proj: eqx.nn.Linear
    skip: eqx.nn.Conv2d | None  # 1x1 conv when in/out channels differ

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        groups = min(8, in_ch)
        self.norm1 = eqx.nn.GroupNorm(groups, in_ch)
        self.norm2 = eqx.nn.GroupNorm(groups, out_ch)
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, 3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, 3, padding=1, key=k2)
        self.time_proj = eqx.nn.Linear(t_dim, out_ch, key=k3)
        self.skip = (
            eqx.nn.Conv2d(in_ch, out_ch, 1, key=k4) if in_ch != out_ch else None
        )

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        h = self.conv1(jax.nn.gelu(self.norm1(x)))
        # FiLM: add a per-channel bias from the time embedding
        h = h + self.time_proj(t_emb)[:, None, None]
        h = self.conv2(jax.nn.gelu(self.norm2(h)))
        skip = x if self.skip is None else self.skip(x)
        return skip + h


class SelfAttention(eqx.Module):
    """Single-head self-attention over spatial positions (cheap at 16x16)."""

    norm: eqx.nn.GroupNorm
    qkv: eqx.nn.Conv2d
    proj: eqx.nn.Conv2d
    heads: int = eqx.field(static=True)

    def __init__(self, ch: int, *, heads: int = 1, key):
        k1, k2 = jax.random.split(key)
        self.norm = eqx.nn.GroupNorm(min(8, ch), ch)
        self.qkv = eqx.nn.Conv2d(ch, 3 * ch, 1, key=k1)
        self.proj = eqx.nn.Conv2d(ch, ch, 1, key=k2)
        self.heads = heads

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(3, C, H * W)  # (3, C, N)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = jax.nn.softmax(q.T @ k / jnp.sqrt(C), axis=-1)  # (N, N)
        out = (v @ attn.T).reshape(C, H, W)
        return x + self.proj(out)


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, ch: int, *, key):
        self.conv = eqx.nn.Conv2d(ch, ch, 3, stride=2, padding=1, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x)


class Upsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, ch: int, *, key):
        self.conv = eqx.nn.Conv2d(ch, ch, 3, padding=1, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2),
                             method="nearest")
        return self.conv(x)


# ----------------------------------------------------------------------------
# Score UNet
# ----------------------------------------------------------------------------
class ScoreUNet(eqx.Module):
    """[STRETCH] Time-conditioned UNet predicting the score s_theta(x, t).

    Input ``x`` has shape ``(1, 128, 128)`` (C, H, W) and ``t`` is a scalar in
    ``[0, 1]``. Output has the same shape as ``x``. Predicts the *noise*
    epsilon, which for the VP-SDE equals ``-sigma(t) * score``.

    Architecture: 3 resolution levels (128 -> 64 -> 32), channel multipliers
    (1, 2, 4) of ``base_ch``, ResBlocks with time conditioning, a
    SelfAttention at the bottleneck, and a symmetric decoder with skip
    connections.
    """

    t_embed: TimeEmbed
    in_conv: eqx.nn.Conv2d
    down_blocks: list
    down_samples: list
    mid_block1: ResBlock
    mid_attn: SelfAttention
    mid_block2: ResBlock
    up_samples: list
    up_blocks: list
    out_norm: eqx.nn.GroupNorm
    out_conv: eqx.nn.Conv2d
    base_ch: int = eqx.field(static=True)

    def __init__(self, *, key, base_ch: int = 32, t_dim: int = 128):
        k = jax.random.split(key, 32)
        c1, c2, c4 = base_ch, base_ch * 2, base_ch * 4

        self.t_embed = TimeEmbed(t_dim, key=k[0])
        self.in_conv = eqx.nn.Conv2d(1, c1, 3, padding=1, key=k[1])

        # Encoder: 3 levels, 2 ResBlocks per level
        self.down_blocks = [
            ResBlock(c1, c1, t_dim, key=k[2]),
            ResBlock(c1, c1, t_dim, key=k[3]),
            ResBlock(c1, c2, t_dim, key=k[4]),
            ResBlock(c2, c2, t_dim, key=k[5]),
            ResBlock(c2, c4, t_dim, key=k[6]),
            ResBlock(c4, c4, t_dim, key=k[7]),
        ]
        self.down_samples = [
            Downsample(c1, key=k[8]),
            Downsample(c2, key=k[9]),
        ]

        # Bottleneck (32x32)
        self.mid_block1 = ResBlock(c4, c4, t_dim, key=k[10])
        self.mid_attn = SelfAttention(c4, key=k[11])
        self.mid_block2 = ResBlock(c4, c4, t_dim, key=k[12])

        # Decoder: mirror of the encoder
        self.up_samples = [
            Upsample(c4, key=k[13]),
            Upsample(c2, key=k[14]),
        ]
        self.up_blocks = [
            ResBlock(c4 + c4, c4, t_dim, key=k[15]),
            ResBlock(c4 + c4, c4, t_dim, key=k[16]),
            ResBlock(c4 + c2, c2, t_dim, key=k[17]),
            ResBlock(c2 + c2, c2, t_dim, key=k[18]),
            ResBlock(c2 + c1, c1, t_dim, key=k[19]),
            ResBlock(c1 + c1, c1, t_dim, key=k[20]),
        ]
        self.out_norm = eqx.nn.GroupNorm(min(8, c1), c1)
        self.out_conv = eqx.nn.Conv2d(c1, 1, 3, padding=1, key=k[21])
        self.base_ch = base_ch

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t_emb = self.t_embed(jnp.asarray(t, dtype=jnp.float32))

        h = self.in_conv(x)

        # Encoder with skip collection
        skips = []
        i = 0
        for lvl in range(3):
            for _ in range(2):
                h = self.down_blocks[i](h, t_emb)
                skips.append(h)
                i += 1
            if lvl < 2:
                h = self.down_samples[lvl](h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder: pop skips in reverse
        i = 0
        for lvl in range(3):
            for _ in range(2):
                s = skips.pop()
                h = jnp.concatenate([h, s], axis=0)
                h = self.up_blocks[i](h, t_emb)
                i += 1
            if lvl < 2:
                h = self.up_samples[lvl](h)

        h = jax.nn.gelu(self.out_norm(h))
        return self.out_conv(h)


# ----------------------------------------------------------------------------
# SDE schedules
# ----------------------------------------------------------------------------
def marginal_std(t: jnp.ndarray, sigma_min: float = 0.01, sigma_max: float = 50.0):
    """[STRETCH] VE-SDE marginal std at time t in [0, 1]."""
    return sigma_min * (sigma_max / sigma_min) ** t


def vp_beta(t: jnp.ndarray, beta_min: float = 0.1, beta_max: float = 20.0):
    """Linear beta schedule for the VP-SDE: beta(t) = beta_min + t*(beta_max-beta_min)."""
    return beta_min + t * (beta_max - beta_min)


def vp_marginal_log_mean_coef(t: jnp.ndarray, beta_min: float = 0.1,
                              beta_max: float = 20.0):
    """log alpha(t) for the VP-SDE with linear beta.

    alpha(t) = exp(-0.5 * ∫_0^t beta(s) ds)
             = exp(-0.5 * (beta_min*t + 0.5*(beta_max-beta_min)*t**2))
    """
    return -0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2)


def vp_marginal_std(t: jnp.ndarray, beta_min: float = 0.1, beta_max: float = 20.0):
    """sigma(t) = sqrt(1 - alpha(t)^2) for the VP-SDE."""
    log_alpha = vp_marginal_log_mean_coef(t, beta_min, beta_max)
    return jnp.sqrt(1.0 - jnp.exp(2.0 * log_alpha))


def vp_score_from_eps(eps: jnp.ndarray, t: jnp.ndarray,
                      beta_min: float = 0.1, beta_max: float = 20.0):
    """Convert an epsilon-prediction to the score: s = -eps / sigma(t)."""
    return -eps / vp_marginal_std(t, beta_min, beta_max)


def vp_perturb(x0: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray,
               beta_min: float = 0.1, beta_max: float = 20.0):
    """Forward VP-SDE: x_t = alpha(t)*x0 + sigma(t)*eps."""
    log_alpha = vp_marginal_log_mean_coef(t, beta_min, beta_max)
    alpha = jnp.exp(log_alpha)
    sigma = jnp.sqrt(1.0 - alpha ** 2)
    # broadcast scalars over (C, H, W)
    return alpha * x0 + sigma * eps


def vp_reverse_step(x_t, t, t_prev, eps_pred,
                    beta_min: float = 0.1, beta_max: float = 20.0):
    """One DDIM-style deterministic reverse step for the VP-SDE.

    Uses the epsilon-prediction to form the score, then updates
    x_{t_prev} = alpha_{t_prev} * x0_hat + sigma_{t_prev} * eps_pred, where
    x0_hat = (x_t - sigma_t * eps_pred) / alpha_t.
    """
    a_t = jnp.exp(vp_marginal_log_mean_coef(t, beta_min, beta_max))
    a_p = jnp.exp(vp_marginal_log_mean_coef(t_prev, beta_min, beta_max))
    s_t = jnp.sqrt(1.0 - a_t ** 2)
    s_p = jnp.sqrt(1.0 - a_p ** 2)
    x0_hat = (x_t - s_t * eps_pred) / a_t
    return a_p * x0_hat + s_p * eps_pred


# ----------------------------------------------------------------------------
# Loss / training utilities
# ----------------------------------------------------------------------------
def score_matching_loss(model: ScoreUNet, x0: jnp.ndarray, key,
                        beta_min: float = 0.1, beta_max: float = 20.0):
    """Denoising score matching loss for the VP-SDE (epsilon parameterization).

    ``x0`` should be a single (1, 128, 128) image in [0, 1]. We draw a random
    time t ~ U[0, 1], noise eps ~ N(0, I), form x_t, and regress eps.
    """
    k_t, k_eps = jax.random.split(key)
    t = jax.random.uniform(k_t, (), minval=0.0, maxval=1.0)
    eps = jax.random.normal(k_eps, x0.shape)
    x_t = vp_perturb(x0, t, eps, beta_min, beta_max)
    eps_hat = model(x_t, t)
    return jnp.mean((eps_hat - eps) ** 2)


@eqx.filter_jit
def _train_step(model, opt_state, optimizer, x0, key, beta_min, beta_max):
    def loss_fn(m):
        return score_matching_loss(m, x0, key, beta_min, beta_max)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    return eqx.apply_updates(model, updates), opt_state, loss


def train_score_model(images, steps: int = 1000, lr: float = 2e-4,
                      seed: int = 0, base_ch: int = 32,
                      beta_min: float = 0.1, beta_max: float = 20.0):
    """Train a ScoreUNet on a stack of images ``(N, 128, 128)``.

    Uses Adam from optax. Returns the trained model.
    """
    import optax
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    model = ScoreUNet(key=init_key, base_ch=base_ch)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n = images.shape[0]
    for it in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (), 0, n)
        x0 = images[idx][None]  # (1, 128, 128)
        model, opt_state, loss = _train_step(
            model, opt_state, optimizer, x0, sub, beta_min, beta_max
        )
        if it % 100 == 0:
            print(f"[score] step {it:4d}  loss {float(loss):.4f}")
    return model


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def sample_vp(model: ScoreUNet, key, shape=(1, 128, 128), num_steps: int = 100,
              beta_min: float = 0.1, beta_max: float = 20.0):
    """DDIM-style deterministic ancestral-free sampler for the VP-SDE.

    Starts from x(1) ~ N(0, I) and integrates t: 1 -> 0 with ``num_steps``
    uniform steps.
    """
    x = jax.random.normal(key, shape)
    ts = jnp.linspace(1.0, 0.0, num_steps + 1)
    for i in range(num_steps):
        t = ts[i]
        t_prev = ts[i + 1]
        eps_pred = model(x, t)
        x = vp_reverse_step(x, t, t_prev, eps_pred, beta_min, beta_max)
    return x