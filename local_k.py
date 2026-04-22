"""Local covariance K(z): same shape as global K; sparsity-structured prediction when a mask is provided."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cov_l1_mask import apply_cov_l1_mask, row_sparse_indices_from_mask


def _sym(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (t + t.transpose(-1, -2))


class _SparseMaskCovBlock(nn.Module):
    """
    Predict only entries on a fixed sparsity mask support.

    Parameterization:
        Let idx be (D,M) column indices per row (with -1 padding).
        Network predicts values V(z) in R^{B x D x M} for those entries.
        Construct dense K as:
            K = I + sym(scatter(V, idx))
        and (optionally) apply the same mask again for safety.
    """

    def __init__(self, latent_dim: int, mask: torch.Tensor, hidden: int) -> None:
        super().__init__()
        if mask.dim() != 2 or mask.size(0) != mask.size(1):
            raise ValueError(f"mask must be square (D,D), got {tuple(mask.shape)}")
        self.d = int(mask.size(0))
        idx = row_sparse_indices_from_mask(mask)
        self.m = int(idx.size(1))
        self.register_buffer("mask", mask.to(dtype=torch.float32), persistent=False)
        self.register_buffer("idx", idx, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d * self.m),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = int(z.size(0))
        v = self.net(z).view(b, self.d, self.m)  # values for (i, idx[i,*])
        k = torch.zeros(b, self.d, self.d, device=z.device, dtype=z.dtype)

        # Scatter all valid entries in one shot.
        valid = self.idx >= 0  # (D,M)
        if valid.any():
            ip, mp = torch.nonzero(valid, as_tuple=True)  # (N,), (N,)
            jp = self.idx[ip, mp]  # (N,)
            n = int(ip.numel())
            b_idx = torch.arange(b, device=z.device).unsqueeze(1).expand(b, n)  # (B,N)
            i_idx = ip.unsqueeze(0).expand(b, n)
            j_idx = jp.unsqueeze(0).expand(b, n)
            k[b_idx, i_idx, j_idx] = v[:, ip, mp]

        # Symmetrize and add identity (init near I).
        k = 0.5 * (k + k.transpose(-1, -2))
        eye = torch.eye(self.d, device=z.device, dtype=z.dtype).expand(b, self.d, self.d)
        k = k + eye
        # Enforce exact support (keeps identity on diag too because mask has diag=1).
        k = k * self.mask.unsqueeze(0).to(dtype=k.dtype)
        return k


class LocalKHead(nn.Module):
    """
    Input: latent z (B, latent_dim) — use posterior mean μ or a sampled z in training.
    Output: same layout as global K — one (B, d, d), or list of (B, HW, HW) per channel.
    """

    def __init__(
        self,
        latent_dim: int,
        img_h: int,
        img_w: int,
        in_channels: int,
        *,
        cifar_channel_independent: bool,
        hidden: int,
        cov_l1_masks: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = in_channels
        self.cifar_channel_independent = cifar_channel_independent
        self.cov_l1_masks = cov_l1_masks
        hw = img_h * img_w

        if in_channels == 1:
            self.mode = "single"
            if not isinstance(cov_l1_masks, torch.Tensor):
                raise ValueError("LocalKHead requires cov_l1_masks (enable --cov-sparse-l1-d).")
            self._block = _SparseMaskCovBlock(latent_dim, cov_l1_masks, hidden)
        elif cifar_channel_independent:
            self.mode = "independent"
            if not isinstance(cov_l1_masks, list):
                raise ValueError("LocalKHead requires cov_l1_masks list (enable --cov-sparse-l1-d).")
            self._blocks = nn.ModuleList(
                [_SparseMaskCovBlock(latent_dim, cov_l1_masks[c], hidden) for c in range(in_channels)]
            )
        else:
            self.mode = "joined"
            if not isinstance(cov_l1_masks, torch.Tensor):
                raise ValueError("LocalKHead requires cov_l1_masks (enable --cov-sparse-l1-d).")
            self._block = _SparseMaskCovBlock(latent_dim, cov_l1_masks, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        if self.mode == "single" or self.mode == "joined":
            out = self._block(z)
            return apply_cov_l1_mask(out, self.cov_l1_masks, batched_k=True)
        outs = [self._blocks[c](z) for c in range(self.in_channels)]
        return apply_cov_l1_mask(outs, self.cov_l1_masks, batched_k=True)


@torch.no_grad()
def compute_outer_product_targets(
    model: nn.Module,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    k: int,
    *,
    in_channels: int,
    img_h: int,
    img_w: int,
    cifar_channel_independent: bool,
    cov_l1_mask: torch.Tensor | list[torch.Tensor] | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """
    Per-sample estimate of conditional residual *second moment*, no gradient:

        K(z) = E[r r^T | z]

    where r = x - decode(mu + std * eps).

    Notes:
      - We intentionally use a *single-sample* estimate (ignore `k`) to match the
        assumption E[r] ≈ 0 and avoid multi-sampling / centering.
    """
    b, _, h, w = x.shape
    hw = img_h * img_w
    std = torch.exp(0.5 * logvar)
    if in_channels == 1:
        eps = torch.randn_like(mu)
        z = mu + std * eps
        recon = model.decode(z)
        r = (x - recon).reshape(b, -1)
        out = _sym(torch.einsum("bi,bj->bij", r, r))
        return apply_cov_l1_mask(out, cov_l1_mask, batched_k=True)
    if cifar_channel_independent:
        eps = torch.randn_like(mu)
        z = mu + std * eps
        recon = model.decode(z)
        r = x - recon
        outs = []
        for c in range(in_channels):
            rc = r[:, c, :, :].reshape(b, -1)
            outs.append(_sym(torch.einsum("bi,bj->bij", rc, rc)))
        return apply_cov_l1_mask(outs, cov_l1_mask, batched_k=True)
    d = in_channels * hw
    eps = torch.randn_like(mu)
    z = mu + std * eps
    recon = model.decode(z)
    r = (x - recon).reshape(b, -1)
    out = _sym(torch.einsum("bi,bj->bij", r, r))
    return apply_cov_l1_mask(out, cov_l1_mask, batched_k=True)


def local_k_supervision_loss(
    k_pred: torch.Tensor | list[torch.Tensor],
    target: torch.Tensor | list[torch.Tensor],
    *,
    in_channels: int,
    cifar_channel_independent: bool,
) -> torch.Tensor:
    """Frobenius MSE between predicted K(z) and target second-moment matrices."""
    if in_channels == 1 or not cifar_channel_independent:
        assert isinstance(k_pred, torch.Tensor) and isinstance(target, torch.Tensor)
        return F.mse_loss(k_pred, target)
    assert isinstance(k_pred, list) and isinstance(target, list)
    s = F.mse_loss(k_pred[0], target[0])
    for c in range(1, in_channels):
        s = s + F.mse_loss(k_pred[c], target[c])
    return s / float(in_channels)
