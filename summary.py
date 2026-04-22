"""
Summarize and compare multiple VAE checkpoints per dataset.

For each dataset (mnist / cifar10), expects the following checkpoints in cwd:
  - <ds>_baseline.pt
  - <ds>_globalK.pt
  - <ds>_localK.pt
  - <ds>_globalK_sparse.pt
  - <ds>_localK_sparse.pt

Outputs (into --out-dir/<dataset>/):
  - prior_grid_compare_4x4.png         (5 grids, arranged in 3 rows: baseline / non-sparse / sparse)
  - posterior_grid_compare_4x4.png     (same layout, z ~ q(z|x) sampling)
  - curves_*.png                       (training curves comparisons)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from ckpt import build_vae_from_checkpoint, load_checkpoint_dict


@dataclass(frozen=True)
class RunSpec:
    key: str
    title: str
    ckpt_path: Path


def _get_dataset(name: str, data_dir: str, train: bool) -> torch.utils.data.Dataset:
    tfm = transforms.ToTensor()
    if name == "mnist":
        return datasets.MNIST(data_dir, train=train, download=True, transform=tfm)
    if name == "cifar10":
        return datasets.CIFAR10(data_dir, train=train, download=True, transform=tfm)
    raise ValueError(f"unknown dataset: {name}")


def _grid_to_numpy(grid: torch.Tensor) -> np.ndarray:
    # grid: (C,H,W) in [0,1]
    arr = grid.detach().cpu().permute(1, 2, 0).numpy()
    return arr


def _draw_grid(ax: plt.Axes, grid: torch.Tensor, *, title: str) -> None:
    arr = _grid_to_numpy(grid)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    if arr.shape[2] == 1:
        ax.imshow(arr[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(arr, vmin=0, vmax=1)


def _sample_prior_grid(model, device: torch.device, n: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    z = torch.randn(n, model.latent_dim, device=device, generator=g)
    x = model.decode(z).detach().cpu()
    return make_grid(x, nrow=4, padding=2, pad_value=1.0)


@torch.inference_mode()
def _sample_posterior_grid(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    n: int = 16,
    seed: int = 0,
) -> torch.Tensor:
    # pick one batch deterministically, then sample z ~ q(z|x)
    # Use a CPU generator for maximum compatibility across PyTorch builds
    # (some ops like randperm require a CPU generator even when tensors are on CUDA).
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    x0, _ = next(iter(loader))
    if x0.size(0) < n:
        raise ValueError(f"need batch_size >= {n} for posterior grid, got {x0.size(0)}")
    idx = torch.randperm(x0.size(0), generator=g)[:n]
    x = x0[idx].to(device)
    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(mu.shape, device="cpu", dtype=mu.dtype, generator=g).to(device)
    z = mu + std * eps
    out = model.decode(z).detach().cpu()
    return make_grid(out, nrow=4, padding=2, pad_value=1.0)


def _make_5grid_3row_figure(
    grids: dict[str, torch.Tensor],
    titles: dict[str, str],
    *,
    out_path: Path,
    fig_title: str,
) -> None:
    # Layout: 3 rows × 2 cols with baseline spanning 2 cols on row 0.
    fig = plt.figure(figsize=(12.5, 16.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.0], hspace=0.18, wspace=0.06)

    ax_base = fig.add_subplot(gs[0, :])
    _draw_grid(ax_base, grids["baseline"], title=titles["baseline"])

    ax_g = fig.add_subplot(gs[1, 0])
    _draw_grid(ax_g, grids["globalK"], title=titles["globalK"])
    ax_l = fig.add_subplot(gs[1, 1])
    _draw_grid(ax_l, grids["localK"], title=titles["localK"])

    ax_gs = fig.add_subplot(gs[2, 0])
    _draw_grid(ax_gs, grids["globalK_sparse"], title=titles["globalK_sparse"])
    ax_ls = fig.add_subplot(gs[2, 1])
    _draw_grid(ax_ls, grids["localK_sparse"], title=titles["localK_sparse"])

    fig.suptitle(fig_title, fontsize=12, y=0.99)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def _series(ckpt: dict[str, Any], key: str) -> np.ndarray | None:
    th = ckpt.get("train_history")
    if not isinstance(th, dict):
        return None
    v = th.get(key)
    if not isinstance(v, list) or len(v) == 0:
        return None
    if not isinstance(v[0], (int, float, np.floating)):
        return None
    return np.asarray(v, dtype=np.float64)


def _plot_compare(
    runs: list[tuple[str, dict[str, Any]]],
    key: str,
    *,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for name, ckpt in runs:
        y = _series(ckpt, key)
        if y is None:
            continue
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, linewidth=1.2, label=name)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _dataset_runs(ds: str, root: Path) -> list[RunSpec]:
    return [
        RunSpec("baseline", "baseline", root / f"{ds}_baseline.pt"),
        RunSpec("globalK", "globalK", root / f"{ds}_globalK.pt"),
        RunSpec("localK", "localK", root / f"{ds}_localK.pt"),
        RunSpec("globalK_sparse", "globalK sparse", root / f"{ds}_globalK_sparse.pt"),
        RunSpec("localK_sparse", "localK sparse", root / f"{ds}_localK_sparse.pt"),
    ]


def _require_exists(p: Path) -> None:
    if not p.is_file():
        raise FileNotFoundError(str(p))


def summarize_dataset(ds: str, *, data_dir: str, device: torch.device, out_dir: Path, seed: int) -> None:
    out_ds = out_dir / ds
    out_ds.mkdir(parents=True, exist_ok=True)

    runspecs = _dataset_runs(ds, Path("."))
    for rs in runspecs:
        _require_exists(rs.ckpt_path)

    # Load ckpts and models
    ckpts: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}
    titles: dict[str, str] = {}
    for rs in runspecs:
        ckpt = load_checkpoint_dict(rs.ckpt_path, device)
        model, _meta = build_vae_from_checkpoint(ckpt, device)
        ckpts[rs.key] = ckpt
        models[rs.key] = model
        titles[rs.key] = rs.title

    # Posterior loader (single pass, deterministic ordering)
    ds_train = _get_dataset(ds, data_dir, train=True)
    post_loader = DataLoader(ds_train, batch_size=256, shuffle=False, num_workers=0)

    # 1) Prior grid compare
    prior_grids = {k: _sample_prior_grid(models[k], device, n=16, seed=seed) for k in titles.keys()}
    _make_5grid_3row_figure(
        prior_grids,
        titles,
        out_path=out_ds / "prior_grid_compare_4x4.png",
        fig_title=f"{ds}: prior samples (4×4 per checkpoint)",
    )

    # 2) Posterior grid compare
    post_grids = {
        k: _sample_posterior_grid(models[k], post_loader, device, n=16, seed=seed) for k in titles.keys()
    }
    _make_5grid_3row_figure(
        post_grids,
        titles,
        out_path=out_ds / "posterior_grid_compare_4x4.png",
        fig_title=f"{ds}: posterior samples z~q(z|x) (4×4 per checkpoint)",
    )

    # 3) Curves compare (multiple figures)
    runs = [(titles[k], ckpts[k]) for k in ["baseline", "globalK", "localK", "globalK_sparse", "localK_sparse"]]
    _plot_compare(
        runs,
        "monitor_mse",
        out_path=out_ds / "curves_mse_mon_compare.png",
        title=f"{ds}: mse(mon) comparison",
        ylabel="mse(mon)",
    )
    _plot_compare(
        runs,
        "global_k_diag_mean",
        out_path=out_ds / "curves_global_k_diag_mean_compare.png",
        title=f"{ds}: global_k_diag_mean comparison",
        ylabel="mean(diag(global_k))",
    )
    # localK-only curves (may be missing for baseline/globalK)
    _plot_compare(
        runs,
        "local_k",
        out_path=out_ds / "curves_local_k_compare.png",
        title=f"{ds}: local_k supervision loss comparison",
        ylabel="local_k",
    )
    _plot_compare(
        runs,
        "local_k_diag_mean",
        out_path=out_ds / "curves_local_k_diag_mean_compare.png",
        title=f"{ds}: local_k_diag_mean comparison",
        ylabel="mean(diag(local_k_pred))",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize and compare multiple VAE checkpoints.")
    p.add_argument("--datasets", type=str, default="mnist,cifar10", help="Comma-separated: mnist,cifar10")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./summary_out")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_list = [s.strip() for s in str(args.datasets).split(",") if s.strip()]
    for ds in ds_list:
        summarize_dataset(ds, data_dir=str(args.data_dir), device=device, out_dir=out_dir, seed=int(args.seed))
        print((out_dir / ds).resolve())


if __name__ == "__main__":
    main()

