"""冻结 VAE 下：重建、残差 r=x-μ_θ(μ)、两遍法估计协方差 K̂ 及与 σ²I 的结构对比指标。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import VAE

ChannelMode = Literal["independent", "joined"]


@torch.inference_mode()
def recon_from_encoder_mean(model: VAE, x: torch.Tensor) -> torch.Tensor:
    """确定性重建：z 取编码均值 μ，recon = decode(μ)，与 r = x - μ_θ(μ) 一致。"""
    mu, _ = model.encode(x)
    return model.decode(mu)


@dataclass
class IsoMetrics:
    dim: int
    sigma2_iso: float
    fro_offdiag: float
    fro_full: float
    fro_vs_iso: float
    rel_offdiag_energy: float
    diag_mean: float
    diag_std: float
    diag_rel_std: float


def isotropic_structure_metrics(K: np.ndarray) -> IsoMetrics:
    """K 为对称协方差矩阵；σ² 取对角均值，报告非对角能量与对角均匀性。"""
    d = int(K.shape[0])
    diag = np.diag(K).astype(np.float64)
    sig2 = float(np.mean(diag))
    off = K.astype(np.float64) - np.diag(diag)
    fro_off = float(np.linalg.norm(off, ord="fro"))
    fro_full = float(np.linalg.norm(K, ord="fro")) + 1e-12
    eye = np.eye(d, dtype=np.float64) * sig2
    fro_vs_iso = float(np.linalg.norm(K.astype(np.float64) - eye, ord="fro"))
    dm = float(np.mean(diag))
    ds = float(np.std(diag))
    return IsoMetrics(
        dim=d,
        sigma2_iso=sig2,
        fro_offdiag=fro_off,
        fro_full=fro_full,
        fro_vs_iso=fro_vs_iso,
        rel_offdiag_energy=fro_off / fro_full,
        diag_mean=dm,
        diag_std=ds,
        diag_rel_std=ds / (abs(dm) + 1e-12),
    )


def _pass1_mean_residual(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[torch.Tensor, int]:
    c = model.in_channels
    h, w = model.img_h, model.img_w
    sum_r = torch.zeros(c, h, w, device=device)
    n_img = 0
    for bi, (x, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        recon = recon_from_encoder_mean(model, x)
        r = x - recon
        sum_r += r.sum(dim=0)
        n_img += x.size(0)
    if n_img == 0:
        raise RuntimeError("No samples in pass1.")
    mean_r = sum_r / float(n_img)
    return mean_r, n_img


def _pass2_covariance(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    mean_r: torch.Tensor,
    mode: ChannelMode,
    max_batches: int | None,
    log_every: int,
    log_fn: Callable[[int, int, list[IsoMetrics], int | None], None] | None,
) -> np.ndarray | list[np.ndarray]:
    c, h, w = model.in_channels, model.img_h, model.img_w
    hw = h * w
    mean_r = mean_r.to(device)

    if mode == "joined" or c == 1:
        d = c * hw
        S = torch.zeros(d, d, device=device, dtype=torch.float64)
        n_seen = 0
        for bi, (x, _) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = x.to(device)
            recon = recon_from_encoder_mean(model, x)
            r = (x - recon).reshape(x.size(0), -1).to(torch.float64)
            m = mean_r.reshape(1, -1).to(torch.float64)
            rc = r - m
            S += rc.T @ rc
            n_seen += x.size(0)
            if log_every > 0 and log_fn is not None and (bi + 1) % log_every == 0 and n_seen > 1:
                Kp = (S / float(n_seen - 1)).cpu().numpy()
                log_fn(bi + 1, n_seen, [isotropic_structure_metrics(Kp)], None)

        if n_seen < 2:
            raise RuntimeError("Need at least 2 samples for covariance.")
        K = (S / float(n_seen - 1)).cpu().numpy()
        if log_every > 0 and log_fn is not None:
            log_fn(-1, n_seen, [isotropic_structure_metrics(K)], None)
        return K

    # channel-independent (C>1): 单次遍历更新各通道 scatter 矩阵
    S_list = [torch.zeros(hw, hw, device=device, dtype=torch.float64) for _ in range(c)]
    m_ch_list = [mean_r[ch].reshape(1, hw).to(torch.float64) for ch in range(c)]
    n_seen = 0
    for bi, (x, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        recon = recon_from_encoder_mean(model, x)
        diff = x - recon
        for ch in range(c):
            r = diff[:, ch, :, :].reshape(x.size(0), hw).to(torch.float64)
            rc = r - m_ch_list[ch]
            S_list[ch] += rc.T @ rc
        n_seen += x.size(0)
        if log_every > 0 and log_fn is not None and (bi + 1) % log_every == 0 and n_seen > 1:
            mets = [
                isotropic_structure_metrics((S_list[ch] / float(n_seen - 1)).cpu().numpy())
                for ch in range(c)
            ]
            log_fn(bi + 1, n_seen, mets, None)

    if n_seen < 2:
        raise RuntimeError("Need at least 2 samples for covariance.")
    K_list = [(S_list[ch] / float(n_seen - 1)).cpu().numpy() for ch in range(c)]
    if log_every > 0 and log_fn is not None:
        log_fn(
            -1,
            n_seen,
            [isotropic_structure_metrics(K_list[ch]) for ch in range(c)],
            None,
        )
    return K_list


def estimate_residual_covariance(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    *,
    channel_mode: ChannelMode = "independent",
    max_batches: int | None = None,
    log_every: int = 0,
    log_fn: Callable[..., None] | None = None,
) -> tuple[torch.Tensor, int, np.ndarray | list[np.ndarray]]:
    """
    两遍扫描：先 r̄，再 K̂（以 r̄ 为中心）。
    channel_mode: cifar 默认 independent；joined 为全维 C·H·W 协方差。
    log_every>0 时在第二遍按 batch 用当前子样本打印指标（迭代动态）。
    """
    mean_r, n_total = _pass1_mean_residual(model, loader, device, max_batches)

    inner_log = log_fn if (log_every > 0 and log_fn is not None) else None

    K_or_list = _pass2_covariance(
        model,
        loader,
        device,
        mean_r,
        channel_mode if model.in_channels > 1 else "joined",
        max_batches,
        log_every,
        inner_log,
    )
    return mean_r.cpu(), n_total, K_or_list


def mean_residual_l2_norm(mean_r: torch.Tensor) -> float:
    """标量：‖r̄‖_2，用于快速检查系统偏差量级。"""
    return float(torch.linalg.norm(mean_r.reshape(-1)).item())


def stack_recon_grid_tensors(
    model: VAE,
    ds: torch.utils.data.Dataset,
    device: torch.device,
    n_show: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """取 n_show 张原图与 decode(μ) 重建，用于网格图。"""
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_show]
    xs = torch.stack([ds[int(i)][0] for i in idx])
    xs_d = xs.to(device)
    recon = recon_from_encoder_mean(model, xs_d).cpu()
    return xs.cpu(), recon
