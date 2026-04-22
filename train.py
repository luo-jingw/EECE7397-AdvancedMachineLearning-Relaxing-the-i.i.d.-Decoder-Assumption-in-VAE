from __future__ import annotations

import argparse
import os
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cov_l1_mask import build_cov_l1_masks
from global_k import (
    clone_global_k_to_cpu,
    ema_update_global_k,
    init_global_k,
    init_global_rrt_second_moment,
    recon_mahalanobis_loss,
)
from local_k import (
    LocalKHead,
    compute_outer_product_targets,
    local_k_supervision_loss,
)
from model import VAE


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def _global_k_checkpoint_mode(in_channels: int, cifar_independent: bool) -> str:
    if in_channels == 1:
        return "single"
    return "independent" if cifar_independent else "joined"


def build_train_checkpoint(
    model_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    input_dim: int,
    in_channels: int,
    img_h: int,
    img_w: int,
    loss_history: list[float],
    recon_history: list[float],
    kl_history: list[float],
    monitor_mse_history: list[float],
    global_k_diag_mean_history: list[float] | None = None,
    global_k_state: torch.Tensor | list[torch.Tensor] | None = None,
    global_k_mode: str | None = None,
    global_rrt_state: torch.Tensor | list[torch.Tensor] | None = None,
    local_k_state: dict[str, torch.Tensor] | None = None,
    local_k_loss_history: list[float] | None = None,
    local_k_diag_mean_history: list[float] | None = None,
) -> dict[str, Any]:
    th: dict[str, Any] = {
        "loss": loss_history,
        "recon": recon_history,
        "kl": kl_history,
        "monitor_mse": monitor_mse_history,
    }
    if global_k_diag_mean_history is not None:
        th["global_k_diag_mean"] = global_k_diag_mean_history
    if local_k_loss_history is not None:
        th["local_k"] = local_k_loss_history
    if local_k_diag_mean_history is not None:
        th["local_k_diag_mean"] = local_k_diag_mean_history
    ckpt: dict[str, Any] = {
        "model_state": model_state,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "in_channels": in_channels,
        "img_h": img_h,
        "img_w": img_w,
        "base_channels": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "seed": int(args.seed),
        "beta": float(args.beta),
        "grad_clip_norm": float(args.grad_clip_norm),
        "train_history": th,
    }
    if global_k_state is not None and global_k_mode is not None:
        ckpt["global_k"] = clone_global_k_to_cpu(global_k_state)
        ckpt["global_k_mode"] = global_k_mode
    if global_rrt_state is not None:
        ckpt["global_rrt"] = clone_global_k_to_cpu(global_rrt_state)
    if local_k_state is not None:
        ckpt["local_k_state"] = {k: v.detach().cpu().clone() for k, v in local_k_state.items()}
    ckpt["cov_sparse_l1_d"] = getattr(args, "cov_sparse_l1_d", None)
    if args.global_k:
        ckpt["global_k_warmup_epochs"] = int(args.global_k_warmup_epochs)
        ckpt["global_k_inv_trace_norm"] = bool(getattr(args, "global_k_inv_trace_norm", False))
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a convolutional VAE (MSE recon + KL)")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default=None, help="Default: ./vae_<dataset>.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training (python/torch).")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="KL weight.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="Global L2 grad clip before optimizer.step(); 0 disables.",
    )
    parser.add_argument(
        "--global-k",
        action="store_true",
        help="Use EMA full covariance K (detached): recon = 0.5 * r^T (K+ridge I)^{-1} r; optional K^{-1} trace norm (see --global-k-inv-trace-norm).",
    )
    parser.add_argument(
        "--global-k-ema",
        type=float,
        default=0.9999,
        help="EMA factor for K update: K <- ema*K + (1-ema)*batch_cov.",
    )
    parser.add_argument(
        "--global-k-ridge",
        type=float,
        default=1e-4,
        help="Ridge added as ridge*I to K before solve (loss only); not scaled by mean(diag(K)).",
    )
    parser.add_argument(
        "--global-k-inv-trace-norm",
        action="store_true",
        help="With --global-k: use tilde K^{-1} = (n/tr(K^{-1})) K^{-1} in recon (n=matrix dim). Default: off (raw K^{-1}).",
    )
    parser.add_argument(
        "--global-k-cifar-independent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CIFAR-10: separate (H*W)^2 K per channel (default true, same as cov_experiment).",
    )
    parser.add_argument(
        "--global-k-warmup-epochs",
        type=int,
        default=10,
        help="With --global-k: first N epochs skip global K EMA and local K head training; recon still uses initial K (identity).",
    )
    parser.add_argument(
        "--local-k",
        action="store_true",
        help="Train LocalKHead(z) with separate optimizer; recon loss still uses global K^{-1} only.",
    )
    parser.add_argument("--local-k-lr", type=float, default=1e-3, help="Adam lr for LocalKHead only.")
    parser.add_argument("--local-k-hidden", type=int, default=256, help="MLP hidden size for LocalKHead.")
    parser.add_argument(
        "--cov-sparse-l1-d",
        type=int,
        default=None,
        help="If set, keep only K[i,j] with L1 index distance <= d on the pixel grid (NCHW for joined). "
        "Applies to global K EMA, Mahalanobis recon, and local K I/O. Default: off.",
    )
    args = parser.parse_args()
    if args.local_k and not args.global_k:
        parser.error("--local-k requires --global-k (VAE recon uses normalized Mahalanobis with global Σ^{-1} only).")
    if args.local_k and args.cov_sparse_l1_d is None:
        parser.error("--local-k now requires --cov-sparse-l1-d (LocalKHead uses sparse mask parameterization only).")
    if args.cov_sparse_l1_d is not None and int(args.cov_sparse_l1_d) < 0:
        parser.error("--cov-sparse-l1-d must be >= 0")
    if args.global_k and int(args.global_k_warmup_epochs) < 0:
        parser.error("--global-k-warmup-epochs must be >= 0")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)
    save_path = args.save_path if args.save_path is not None else f"./vae_{args.dataset}.pt"

    os.makedirs(args.data_dir, exist_ok=True)
    if args.dataset == "mnist":
        train_ds = datasets.MNIST(
            args.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        in_channels, img_h, img_w = 1, 28, 28
        input_dim = 28 * 28
    else:
        train_ds = datasets.CIFAR10(
            args.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        in_channels, img_h, img_w = 3, 32, 32
        input_dim = 32 * 32 * 3
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = VAE(
        in_channels=in_channels,
        img_h=img_h,
        img_w=img_w,
        latent_dim=args.latent_dim,
        base_channels=args.hidden_dim,
    ).to(device)

    optimizer_vae = optim.Adam(model.parameters(), lr=args.lr)

    cifar_indep = bool(args.global_k_cifar_independent) if args.dataset == "cifar10" else False
    gk_mode = _global_k_checkpoint_mode(in_channels, cifar_indep)
    cov_l1_masks: torch.Tensor | list[torch.Tensor] | None = None
    if args.cov_sparse_l1_d is not None and (args.global_k or args.local_k):
        cov_l1_masks = build_cov_l1_masks(
            in_channels,
            img_h,
            img_w,
            cifar_channel_independent=cifar_indep,
            d_l1=int(args.cov_sparse_l1_d),
            device=device,
            dtype=torch.float32,
        )
    global_k: torch.Tensor | list[torch.Tensor] | None = None
    global_rrt: torch.Tensor | list[torch.Tensor] | None = None
    if args.global_k:
        global_k = init_global_k(
            in_channels,
            img_h,
            img_w,
            cifar_channel_independent=cifar_indep,
            device=device,
            dtype=torch.float32,
        )
        global_rrt = init_global_rrt_second_moment(
            in_channels,
            img_h,
            img_w,
            cifar_channel_independent=cifar_indep,
            device=device,
            dtype=torch.float32,
        )
        if global_k is not None and cov_l1_masks is not None:
            with torch.no_grad():
                if isinstance(global_k, list):
                    for ci in range(len(global_k)):
                        global_k[ci].mul_(cov_l1_masks[ci])
                else:
                    global_k.mul_(cov_l1_masks)

    local_k_head: LocalKHead | None = None
    optimizer_local: optim.Optimizer | None = None
    if args.local_k:
        local_k_head = LocalKHead(
            latent_dim=args.latent_dim,
            img_h=img_h,
            img_w=img_w,
            in_channels=in_channels,
            cifar_channel_independent=cifar_indep,
            hidden=int(args.local_k_hidden),
            cov_l1_masks=cov_l1_masks,
        ).to(device)
        optimizer_local = optim.Adam(local_k_head.parameters(), lr=float(args.local_k_lr))

    model.train()
    loss_history: list[float] = []
    recon_history: list[float] = []
    kl_history: list[float] = []
    monitor_mse_history: list[float] = []
    local_k_loss_history: list[float] = []
    global_k_diag_mean_history: list[float] = []
    local_k_diag_mean_history: list[float] = []

    def save_last_checkpoint() -> None:
        msd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        lk_sd = local_k_head.state_dict() if args.local_k and local_k_head is not None else None
        ckpt = build_train_checkpoint(
            msd,
            args,
            input_dim=input_dim,
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            loss_history=loss_history,
            recon_history=recon_history,
            kl_history=kl_history,
            monitor_mse_history=monitor_mse_history,
            global_k_diag_mean_history=global_k_diag_mean_history if args.global_k else None,
            global_k_state=global_k if args.global_k else None,
            global_k_mode=gk_mode if args.global_k else None,
            global_rrt_state=global_rrt if args.global_k else None,
            local_k_state=lk_sd,
            local_k_loss_history=local_k_loss_history if args.local_k else None,
            local_k_diag_mean_history=local_k_diag_mean_history if args.local_k else None,
        )
        torch.save(ckpt, save_path)

    if int(args.epochs) > 0:
        print(f"Each epoch end: overwrite {save_path} with last weights (full train_history in ckpt).")
    if args.global_k:
        print(
            f"global_k: mode={gk_mode}  ema={args.global_k_ema}  ridge={args.global_k_ridge}  "
            f"warmup_epochs={int(args.global_k_warmup_epochs)}  "
            f"inv_trace_norm={bool(args.global_k_inv_trace_norm)}  "
            "(recon uses detached K; checkpoint K is raw EMA)"
        )
    if args.local_k:
        print(
            f"local_k: lr={args.local_k_lr}  "
            "(supervision: MSE to single-sample outer product; VAE recon uses global Σ^{-1} only)"
        )
    if cov_l1_masks is not None:
        print(f"cov L1 sparsity: d={args.cov_sparse_l1_d} (masked off-diagonal entries beyond L1 ball)")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        recon_term_sum = 0.0
        kl_term_sum = 0.0
        mse_sum = 0.0
        local_k_sum = 0.0
        local_k_diag_sum = 0.0
        local_k_diag_n = 0
        n_batches = 0
        gcn = float(args.grad_clip_norm)

        for data, _ in train_loader:
            x = data.to(device)
            optimizer_vae.zero_grad(set_to_none=True)
            recon, mu, logvar, _ = model(x)
            r = x - recon
            if args.global_k:
                assert global_k is not None
                recon_term = recon_mahalanobis_loss(
                    r,
                    global_k,
                    float(args.global_k_ridge),
                    in_channels=in_channels,
                    cifar_channel_independent=cifar_indep,
                    cov_l1_mask=cov_l1_masks,
                    inv_trace_norm=bool(args.global_k_inv_trace_norm),
                )
            else:
                recon_term = (
                    0.5
                    * F.mse_loss(recon.view_as(x), x, reduction="none")
                    .view(x.size(0), -1)
                    .sum(dim=1)
                    .mean()
                )
            kl_term = kl_divergence(mu, logvar)
            loss = recon_term + float(args.beta) * kl_term
            with torch.no_grad():
                mse_sum += float(F.mse_loss(recon.detach(), x, reduction="mean").item())
            loss.backward()
            if gcn > 0:
                clip_grad_norm_(model.parameters(), gcn)
            optimizer_vae.step()
            after_global_k_warmup = epoch > int(args.global_k_warmup_epochs)
            if args.global_k and global_k is not None and after_global_k_warmup:
                with torch.no_grad():
                    ema_update_global_k(
                        global_k,
                        global_rrt,
                        r.detach(),
                        float(args.global_k_ema),
                        in_channels=in_channels,
                        img_h=img_h,
                        img_w=img_w,
                        cifar_channel_independent=cifar_indep,
                        cov_l1_mask=cov_l1_masks,
                    )

            if args.local_k and local_k_head is not None and optimizer_local is not None:
                optimizer_local.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if after_global_k_warmup:
                        tgt = compute_outer_product_targets(
                            model,
                            x,
                            mu,
                            logvar,
                            1,
                            in_channels=in_channels,
                            img_h=img_h,
                            img_w=img_w,
                            cifar_channel_independent=cifar_indep,
                            cov_l1_mask=cov_l1_masks,
                        )
                    else:
                        # Warmup: teach LocalKHead to output identity covariance.
                        b = int(x.size(0))
                        hw = img_h * img_w
                        if in_channels == 1:
                            eye = torch.eye(hw, device=device, dtype=torch.float32).expand(b, hw, hw)
                            if isinstance(cov_l1_masks, torch.Tensor):
                                eye = eye * cov_l1_masks.unsqueeze(0)
                            tgt = eye.to(dtype=x.dtype)
                        elif cifar_indep:
                            eye_hw = torch.eye(hw, device=device, dtype=torch.float32).expand(b, hw, hw).to(
                                dtype=x.dtype
                            )
                            if isinstance(cov_l1_masks, list):
                                tgt = [eye_hw * cov_l1_masks[c].unsqueeze(0) for c in range(in_channels)]
                            else:
                                tgt = [eye_hw for _ in range(in_channels)]
                        else:
                            d = in_channels * hw
                            eye = torch.eye(d, device=device, dtype=torch.float32).expand(b, d, d)
                            if isinstance(cov_l1_masks, torch.Tensor):
                                eye = eye * cov_l1_masks.unsqueeze(0)
                            tgt = eye.to(dtype=x.dtype)
                std = torch.exp(0.5 * logvar)
                z_loc = (mu.detach() + std.detach() * torch.randn_like(mu))
                k_pred = local_k_head(z_loc)
                with torch.no_grad():
                    if isinstance(k_pred, list):
                        # mean over batch and dims, then mean over channels
                        dm = 0.0
                        for kc in k_pred:
                            dm += float(kc.diagonal(dim1=-2, dim2=-1).mean().item())
                        dm /= float(len(k_pred))
                    else:
                        dm = float(k_pred.diagonal(dim1=-2, dim2=-1).mean().item())
                    local_k_diag_sum += dm
                    local_k_diag_n += 1
                loss_lk = local_k_supervision_loss(
                    k_pred,
                    tgt,
                    in_channels=in_channels,
                    cifar_channel_independent=cifar_indep,
                )
                loss_lk.backward()
                optimizer_local.step()
                local_k_sum += float(loss_lk.item())

            total_loss += loss.item()
            recon_term_sum += float(recon_term.item())
            kl_term_sum += float(kl_term.item())
            n_batches += 1

        loss_history.append(total_loss / n_batches)
        recon_history.append(recon_term_sum / n_batches)
        kl_history.append(kl_term_sum / n_batches)
        monitor_mse_history.append(mse_sum / n_batches)
        if args.local_k:
            local_k_loss_history.append(local_k_sum / max(n_batches, 1))
            local_k_diag_mean_history.append(local_k_diag_sum / float(max(local_k_diag_n, 1)))
        if args.global_k and global_k is not None:
            with torch.no_grad():
                if isinstance(global_k, list):
                    gdm = 0.0
                    for kc in global_k:
                        gdm += float(kc.diagonal().mean().item())
                    gdm /= float(len(global_k))
                else:
                    gdm = float(global_k.diagonal().mean().item())
            global_k_diag_mean_history.append(gdm)

        msg = (
            f"Epoch {epoch:3d}  avg_loss={total_loss / n_batches:.4f}  "
            f"avg_recon={recon_term_sum / n_batches:.4f}  avg_kl={kl_term_sum / n_batches:.4f}  "
            f"mse(mon)={mse_sum / n_batches:.6f}"
        )
        if args.local_k and n_batches > 0:
            msg += f"  local_k={local_k_sum / n_batches:.6f}"
            if local_k_diag_n > 0:
                msg += f"  local_k_diag_mean={local_k_diag_sum / float(local_k_diag_n):.6f}"
        if args.global_k and len(global_k_diag_mean_history) > 0:
            msg += f"  global_k_diag_mean={global_k_diag_mean_history[-1]:.6f}"
        print(msg)

        if n_batches > 0:
            save_last_checkpoint()

    if int(args.epochs) > 0:
        print(f"Last checkpoint (overwrite each epoch): {save_path}")
    else:
        print("No epochs scheduled; no checkpoint saved.")


if __name__ == "__main__":
    main()
