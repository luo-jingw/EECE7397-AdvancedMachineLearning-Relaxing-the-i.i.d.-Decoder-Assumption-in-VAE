"""VAE visualization: sample grids, t-SNE, prior vs class-conditional latents, training curves from checkpoints."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from ckpt import build_vae_from_checkpoint, load_checkpoint_dict
from model import VAE


def ckpt_output_suffix(ckpt_path: str) -> str:
    """Output file tag = checkpoint filename stem (matches custom --save-path names)."""
    stem = Path(ckpt_path).stem.strip()
    return stem if stem else "ckpt"


# Dark, saturated colors for white background (avoids washed-out Plotly pastels in 3D)
_CLASS_PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
)


def _class_color(class_id: int) -> str:
    return _CLASS_PALETTE[int(class_id) % len(_CLASS_PALETTE)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha)})"


def load_model_from_checkpoint(ckpt: dict, device: torch.device) -> tuple[VAE, dict]:
    model, meta = build_vae_from_checkpoint(ckpt, device)
    return model, meta


def load_model(ckpt_path: str, device: torch.device) -> tuple[VAE, dict]:
    ckpt = load_checkpoint_dict(ckpt_path, device)
    return load_model_from_checkpoint(ckpt, device)


def get_dataset(name: str, data_dir: str, train: bool) -> torch.utils.data.Dataset:
    if name == "mnist":
        return datasets.MNIST(
            data_dir,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
    return datasets.CIFAR10(
        data_dir,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


@torch.inference_mode()
def classwise_mu_mean_var(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    latent_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(num_classes, latent_dim, device=device)
    sq_sums = torch.zeros(num_classes, latent_dim, device=device)
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        mu, _ = model.encode(x)
        sums.index_add_(0, y, mu)
        sq_sums.index_add_(0, y, mu * mu)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.long))

    n = counts.clamp(min=1).unsqueeze(1).to(dtype=sums.dtype)
    mu_bar = sums / n
    mean_sq = sq_sums / n
    var = (mean_sq - mu_bar**2).clamp(min=1e-8)
    return mu_bar, var


@torch.inference_mode()
def save_prior_vs_cond_grid(
    model: VAE,
    mu_c: torch.Tensor,
    var_c: torch.Tensor,
    path: Path,
    device: torch.device,
    class_labels: list[str],
    nrow_prior: int = 4,
    seed: int = 0,
    cond_ncols: int = 5,
) -> None:
    n_prior = nrow_prior * nrow_prior
    latent_dim = model.latent_dim
    num_classes = mu_c.size(0)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    z_prior = torch.randn(n_prior, latent_dim, device=device, generator=g)
    std = torch.sqrt(var_c)
    eps = torch.randn(num_classes, latent_dim, device=device, generator=g)
    z_cond = mu_c + eps * std

    x_prior = model.decode(z_prior).cpu()
    x_cond = model.decode(z_cond).cpu()

    pad = 2
    left = make_grid(x_prior, nrow=nrow_prior, padding=pad, pad_value=1.0)
    c, _, _ = left.shape

    cond_ncols_eff = min(cond_ncols, num_classes) if num_classes > 0 else 1
    nrows_cond = math.ceil(num_classes / cond_ncols_eff)

    fig_w = 4.0 + 2.0 * cond_ncols_eff
    fig = plt.figure(figsize=(fig_w, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35 + 0.65 * (cond_ncols_eff / 5)], wspace=0.14)

    ax_l = fig.add_subplot(gs[0, 0])
    arr_l = left.permute(1, 2, 0).numpy()
    if c == 1:
        ax_l.imshow(arr_l[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        ax_l.imshow(arr_l)
    ax_l.set_axis_off()
    ax_l.set_title(r"$z \sim \mathcal{N}(0, I)$ (4$\times$4)", fontsize=10)

    gs_r = gs[0, 1].subgridspec(nrows_cond, cond_ncols_eff, hspace=0.55, wspace=0.12)
    for idx in range(nrows_cond * cond_ncols_eff):
        r = idx // cond_ncols_eff
        j = idx % cond_ncols_eff
        ax = fig.add_subplot(gs_r[r, j])
        if idx < num_classes:
            img = x_cond[idx]
            arr = img.permute(1, 2, 0).numpy()
            if c == 1:
                ax.imshow(arr.squeeze(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(arr)
            ax.text(
                0.5,
                -0.06,
                class_labels[idx],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
            )
        ax.set_axis_off()

    fig.suptitle(r"$z \sim \mathcal{N}(\mu_c, \mathrm{diag}(\sigma^2_c))$ (one per class)", fontsize=10, y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close()


@torch.inference_mode()
def collect_mu_logvar_labels(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    n = 0
    for x, y in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        labels.append(y.cpu().numpy().astype(np.int64))
        n += x.size(0)
        if max_samples is not None and n >= max_samples:
            break
    mu_arr = np.concatenate(mus, axis=0)
    lv_arr = np.concatenate(logvars, axis=0)
    y_arr = np.concatenate(labels, axis=0)
    if max_samples is not None:
        mu_arr = mu_arr[:max_samples]
        lv_arr = lv_arr[:max_samples]
        y_arr = y_arr[:max_samples]
    return mu_arr, lv_arr, y_arr


def logvar_to_diameter(logvar: np.ndarray) -> np.ndarray:
    """Map total log-variance to Plotly marker *size* (pixels). Clipped so points stay visible."""
    d_raw = np.sum(logvar, axis=1, dtype=np.float64) + 32.0
    d_raw = np.maximum(d_raw, 1e-6)
    # Collapsed posteriors can push sum(logvar) very negative → d_raw ~ 1e-6;
    # Plotly then draws ~invisible dots (often mistaken for "all transparent").
    lo, hi = np.percentile(d_raw, [2.0, 98.0])
    if hi - lo < 1e-12:
        return np.full_like(d_raw, 12.0, dtype=np.float64)
    t = (d_raw - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    return (5.0 + 28.0 * t).astype(np.float64)


def diameter_to_opacity(diameter: np.ndarray) -> np.ndarray:
    """Opacity from diameter; min-max on inv with a hard alpha floor."""
    d = np.asarray(diameter, dtype=np.float64)
    inv = 1.0 / (d**2 + 1e2)
    if not np.all(np.isfinite(inv)):
        return np.full_like(inv, 0.75)
    lo, hi = float(inv.min()), float(inv.max())
    if hi - lo < 1e-15:
        t = np.zeros_like(inv)
    else:
        t = (inv - lo) / (hi - lo)
    opacity = 0.22 + 0.78 * t
    return np.clip(opacity, 0.2, 1.0)


def tsne_features(mu: np.ndarray, random_state: int) -> np.ndarray:
    x = StandardScaler().fit_transform(np.asarray(mu, dtype=np.float64))
    rng = np.random.default_rng(random_state)
    return x + rng.normal(0.0, 1e-8, size=x.shape)


def save_tsne_plotly(
    mu: np.ndarray,
    logvar: np.ndarray,
    labels: np.ndarray,
    out_2d: Path,
    out_3d: Path,
    random_state: int,
    perplexity: float,
) -> None:
    d = logvar_to_diameter(logvar)
    opacity = diameter_to_opacity(d)
    x = tsne_features(mu, random_state)

    ts2 = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate=200.0,
        max_iter=1000,
    )
    z2 = ts2.fit_transform(x)

    fig2 = go.Figure()
    for c in np.unique(labels):
        m = labels == c
        fig2.add_trace(
            go.Scatter(
                x=z2[m, 0],
                y=z2[m, 1],
                mode="markers",
                name=f"class {int(c)}",
                marker=dict(
                    size=d[m],
                    opacity=opacity[m],
                    color=_class_color(int(c)),
                    line=dict(width=0),
                ),
            )
        )
    fig2.update_layout(
        title="t-SNE of mu by class (2D)",
        template="plotly_white",
        width=900,
        height=700,
        legend=dict(itemsizing="constant"),
    )
    fig2.write_html(str(out_2d), include_plotlyjs="cdn")

    ts3 = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate=200.0,
        max_iter=1000,
    )
    z3 = ts3.fit_transform(x)

    fig3 = go.Figure()
    for c in np.unique(labels):
        m = labels == c
        cols = [_hex_to_rgba(_class_color(int(c)), float(a)) for a in opacity[m]]
        fig3.add_trace(
            go.Scatter3d(
                x=z3[m, 0],
                y=z3[m, 1],
                z=z3[m, 2],
                mode="markers",
                name=f"class {int(c)}",
                marker=dict(size=d[m], color=cols, line=dict(width=0)),
            )
        )
    fig3.update_layout(
        title="t-SNE of mu by class (3D)",
        template="plotly_white",
        width=900,
        height=700,
        scene=dict(xaxis_title="dim1", yaxis_title="dim2", zaxis_title="dim3"),
        legend=dict(itemsizing="constant"),
    )
    fig3.write_html(str(out_3d), include_plotlyjs="cdn")


@torch.inference_mode()
def save_latent_grid_png(
    model: VAE,
    path: Path,
    device: torch.device,
    nrow: int = 10,
    seed: int = 0,
) -> None:
    n = nrow * nrow
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    z = torch.randn(n, model.latent_dim, device=device, generator=g)
    out = model.decode(z).cpu()
    grid = make_grid(out, nrow=nrow, padding=2, pad_value=1.0)
    arr = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    if grid.size(0) == 1:
        plt.imshow(arr[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()


@torch.inference_mode()
def save_recon_compare_png(
    model: VAE,
    ds: torch.utils.data.Dataset,
    path: Path,
    device: torch.device,
    n_show: int = 9,
    seed: int = 1,
    gap_px: int = 16,
) -> None:
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_show]
    xs = torch.stack([ds[i][0] for i in idx])
    xs_dev = xs.to(device)
    mu, _ = model.encode(xs_dev)
    recon = model.decode(mu).cpu()

    nrow = int(np.sqrt(n_show))
    pad = 2
    left = make_grid(xs, nrow=nrow, padding=pad, pad_value=1.0)
    right = make_grid(recon, nrow=nrow, padding=pad, pad_value=1.0)
    _, h, _ = left.shape
    gap = torch.ones(3, h, gap_px) if left.size(0) == 3 else torch.ones(1, h, gap_px)
    combined = torch.cat([left, gap, right], dim=2)

    grid_np = combined.permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 5))
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(grid_np)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def plot_train_history_from_checkpoint(
    ckpt: dict, out_dir: Path, suffix: str
) -> list[Path]:
    """
    Plot checkpoint train curves.

    - Always writes a combined figure when possible:
      - subplot A: avg_loss / avg_recon / avg_kl (shared y-axis)
      - subplot B: mse(mon) alone
    - If local_k exists, also writes a separate local_k curve plot.
    - Additionally, writes one figure per 1D numeric series (legacy behavior).
    """
    th = ckpt.get("train_history")
    if not isinstance(th, dict):
        return []
    written: list[Path] = []

    def _get_series(key: str) -> np.ndarray | None:
        v = th.get(key)
        if not isinstance(v, list) or len(v) == 0:
            return None
        if not isinstance(v[0], (int, float, np.floating)):
            return None
        return np.asarray(v, dtype=np.float64)

    # Combined curves (requested layout)
    y_loss = _get_series("loss")
    y_recon = _get_series("recon")
    y_kl = _get_series("kl")
    y_mse = _get_series("monitor_mse")
    if y_loss is not None and y_recon is not None and y_kl is not None and y_mse is not None:
        n = int(min(len(y_loss), len(y_recon), len(y_kl), len(y_mse)))
        x = np.arange(1, n + 1)
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(9, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.12)
        )
        ax0.plot(x, y_loss[:n], label="avg_loss", linewidth=1.2, color="#1f77b4")
        ax0.plot(x, y_recon[:n], label="avg_recon", linewidth=1.2, color="#ff7f0e")
        ax0.plot(x, y_kl[:n], label="avg_kl", linewidth=1.2, color="#2ca02c")
        ax0.set_ylabel("loss / recon / kl")
        ax0.set_title("Training curves (shared y-axis)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best", fontsize=9, frameon=False)

        ax1.plot(x, y_mse[:n], label="mse(mon)", linewidth=1.2, color="#d62728")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("mse(mon)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=9, frameon=False)

        p = out_dir / f"train_curves_{suffix}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # local_k as separate plot (requested)
    y_lk = _get_series("local_k")
    if y_lk is not None:
        x = np.arange(1, len(y_lk) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y_lk, color="#9467bd", linewidth=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("local_k")
        ax.set_title("local_k supervision loss")
        ax.grid(True, alpha=0.3)
        p = out_dir / f"train_local_k_{suffix}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # Per-series plots (legacy behavior; keep for convenience)
    # Skip metrics that are already covered by the combined figure.
    skip_single = {"loss", "recon", "kl"}
    for key, val in th.items():
        if val is None:
            continue
        if str(key) in skip_single:
            continue
        if not isinstance(val, list) or len(val) == 0:
            continue
        if not isinstance(val[0], (int, float, np.floating)):
            continue
        y = np.asarray(val, dtype=np.float64)
        x = np.arange(1, len(y) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y, color="#1f77b4", linewidth=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(f"train_history: {key}")
        ax.grid(True, alpha=0.3)
        safe_key = str(key).replace("/", "_")
        p = out_dir / f"train_{safe_key}_{suffix}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAE visualization: samples, t-SNE, prior vs class-cond grid, checkpoint train curves"
    )
    parser.add_argument("--ckpt", type=str, default="./vae_mnist.pt", help="Checkpoint .pt from train.py")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="./vis_out")
    parser.add_argument(
        "--plot-train-history",
        action="store_true",
        help="Plot each 1D series in checkpoint train_history (one figure per metric).",
    )
    parser.add_argument(
        "--no-prior-classcond",
        action="store_true",
        help="Skip prior N(0,I) vs class-conditional latent sampling grid.",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Export t-SNE 2D/3D Plotly HTML (can be slow). Default: off.",
    )
    parser.add_argument("--tsne-max-samples", type=int, default=8000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoders / classwise stats.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ckpt_output_suffix(args.ckpt)

    ckpt_raw = load_checkpoint_dict(args.ckpt, device)
    if args.plot_train_history:
        paths = plot_train_history_from_checkpoint(ckpt_raw, out_dir, suffix)
        for p in paths:
            print(p.resolve())

    model, meta = load_model_from_checkpoint(ckpt_raw, device)
    ds = get_dataset(meta["dataset"], args.data_dir, train=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if not args.no_prior_classcond:
        mu_c, var_c = classwise_mu_mean_var(
            model,
            loader,
            device,
            args.num_classes,
            meta["latent_dim"],
        )
        cl = getattr(ds, "classes", None)
        if isinstance(cl, (list, tuple)) and len(cl) >= args.num_classes:
            class_labels = [str(cl[i]) for i in range(args.num_classes)]
        else:
            class_labels = [str(i) for i in range(args.num_classes)]
        prior_path = out_dir / f"prior_vs_classcond_4x4_{suffix}.png"
        save_prior_vs_cond_grid(
            model,
            mu_c,
            var_c,
            prior_path,
            device,
            class_labels=class_labels,
            nrow_prior=4,
            seed=args.seed,
        )
        print(prior_path.resolve())

    if args.tsne:
        loader_tsne = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        max_s = args.tsne_max_samples if args.tsne_max_samples > 0 else None
        mu, logvar, y = collect_mu_logvar_labels(model, loader_tsne, device, max_s)
        n = mu.shape[0]
        perp = min(args.perplexity, max(5, (n - 1) // 3))
        save_tsne_plotly(
            mu,
            logvar,
            y,
            out_dir / f"tsne_mu_2d_{suffix}.html",
            out_dir / f"tsne_mu_3d_{suffix}.html",
            random_state=args.seed,
            perplexity=float(perp),
        )

    save_latent_grid_png(
        model,
        out_dir / f"latent_sample_10x10_{suffix}.png",
        device,
        nrow=10,
        seed=args.seed,
    )
    save_recon_compare_png(
        model,
        ds,
        out_dir / f"recon_mu_vs_original_3x3_{suffix}.png",
        device,
        n_show=9,
        seed=args.seed + 1,
    )

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
