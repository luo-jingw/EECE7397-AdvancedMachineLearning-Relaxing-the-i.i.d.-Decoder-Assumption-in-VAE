"""
Frozen checkpoint: residual covariance K_hat vs isotropic baseline, figures and metrics.
Training: use train.py (this script does not update weights).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ckpt import load_vae_from_path
from cov_plots import (
    save_K_diagonal_plot,
    save_K_row_slice_heatmaps,
    save_mean_residual_map,
    save_recon_grid_png,
    write_K_row_slice_interactive_html,
)
from residual_cov import (
    ChannelMode,
    IsoMetrics,
    estimate_residual_covariance,
    isotropic_structure_metrics,
    mean_residual_l2_norm,
    stack_recon_grid_tensors,
)


def _get_dataset(name: str, data_dir: str, train: bool) -> torch.utils.data.Dataset:
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


def _print_metrics(tag: str, m: IsoMetrics) -> None:
    print(
        f"{tag}  dim={m.dim}  sigma2_iso(mean diag)={m.sigma2_iso:.6e}  "
        f"rel_offdiag_energy={m.rel_offdiag_energy:.6f}  "
        f"diag_rel_std={m.diag_rel_std:.6f}  "
        f"||K-sigma2 I||_F={m.fro_vs_iso:.6e}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Residual covariance vs isotropic baseline (frozen VAE)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="./cov_out")
    p.add_argument("--split", type=str, choices=["train", "test"], default="train")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--channel-mode",
        type=str,
        choices=["independent", "joined"],
        default="independent",
        help="CIFAR-10: independent = separate K per channel (default); joined = full (C*H*W)^2 covariance. MNIST is single-channel (joined).",
    )
    p.add_argument("--max-batches", type=int, default=0, help="0 = use full split")
    p.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="In pass 2, print sub-sample metrics every N batches; 0 disables.",
    )
    p.add_argument("--grid-n", type=int, default=64, help="Number of samples in recon comparison grid image")
    p.add_argument("--grid-nrow", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-row-index", type=int, default=-1, help="Static PNG row index for K slice; -1 = middle")
    p.add_argument(
        "--only-k-row-slice-png",
        action="store_true",
        help="Only write K_row_slice_*.png (skip mean residual, recon grid, HTML, diagonal). "
        "Still runs covariance estimation over the split.",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.ckpt).stem

    model, meta, _ = load_vae_from_path(args.ckpt, device)
    ds = _get_dataset(meta["dataset"], args.data_dir, train=(args.split == "train"))
    max_b = args.max_batches if args.max_batches > 0 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    if meta["in_channels"] == 1:
        ch_mode: ChannelMode = "joined"
    else:
        ch_mode = args.channel_mode

    def log_fn(bi: int, n_seen: int, mlist: list[IsoMetrics], _ch: int | None) -> None:
        if bi < 0:
            print(f"[pass2 final] n={n_seen}")
        else:
            print(f"[pass2 batch {bi}] n={n_seen}")
        for i, m in enumerate(mlist):
            prefix = f"  ch{i}" if len(mlist) > 1 else "  K"
            _print_metrics(prefix, m)

    mean_r, n_img, K_or_list = estimate_residual_covariance(
        model,
        loader,
        device,
        channel_mode=ch_mode,
        max_batches=max_b,
        log_every=int(args.log_every),
        log_fn=log_fn if args.log_every > 0 else None,
    )

    print(f"samples={n_img}  ||r_bar||_2={mean_residual_l2_norm(mean_r):.6e}")
    only_row = bool(args.only_k_row_slice_png)

    if not only_row:
        save_mean_residual_map(mean_r, out_dir / f"mean_residual_{stem}.png")

        nrow = int(args.grid_nrow)
        n_show = int(args.grid_n)
        ox, rx = stack_recon_grid_tensors(model, ds, device, n_show=n_show, seed=int(args.seed))
        save_recon_grid_png(ox, rx, out_dir / f"recon_grid_{stem}.png", nrow=nrow)

    k_row = None if args.k_row_index < 0 else int(args.k_row_index)
    mr_np = None if only_row else mean_r.detach().cpu().numpy()

    if isinstance(K_or_list, list):
        for ch, Kc in enumerate(K_or_list):
            tag = f"ch{ch}"
            mc = isotropic_structure_metrics(Kc)
            _print_metrics(f"final {tag}", mc)
            save_K_row_slice_heatmaps(
                Kc,
                1,
                meta["img_h"],
                meta["img_w"],
                out_dir / f"K_row_slice_{tag}_{stem}.png",
                row_index=k_row,
            )
            if not only_row:
                assert mr_np is not None
                write_K_row_slice_interactive_html(
                    out_dir / f"K_row_slice_{tag}_{stem}.html",
                    Kc,
                    1,
                    meta["img_h"],
                    meta["img_w"],
                    picker_bg=mr_np[ch],
                    title=f"K row slice {tag} ({stem})",
                )
                save_K_diagonal_plot(Kc, out_dir / f"K_diagonal_{tag}_{stem}.png", title_suffix=f"({tag})")
    else:
        K = K_or_list
        mf = isotropic_structure_metrics(K)
        _print_metrics("final", mf)
        save_K_row_slice_heatmaps(
            K,
            meta["in_channels"],
            meta["img_h"],
            meta["img_w"],
            out_dir / f"K_row_slice_{stem}.png",
            row_index=k_row,
        )
        if not only_row:
            assert mr_np is not None
            write_K_row_slice_interactive_html(
                out_dir / f"K_row_slice_{stem}.html",
                K,
                meta["in_channels"],
                meta["img_h"],
                meta["img_w"],
                picker_bg=mr_np,
                title=f"K row slice ({stem})",
            )
            save_K_diagonal_plot(K, out_dir / f"K_diagonal_{stem}.png")

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
