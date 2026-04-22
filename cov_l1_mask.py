"""L1 (Manhattan) spatial sparsity masks for covariance matrices (same layout as global K)."""
from __future__ import annotations

import torch


def l1_mask_hw_grid(hw: int, img_w: int, d_l1: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mask (HW, HW): entry (i,j)=1 iff L1 between pixel i and j on H×W grid <= d_l1."""
    h = hw // img_w
    w = img_w
    idx = torch.arange(hw, device=device)
    ri = idx // w
    ci = idx % w
    l1 = (ri.unsqueeze(1) - ri.unsqueeze(0)).abs() + (ci.unsqueeze(1) - ci.unsqueeze(0)).abs()
    return (l1 <= int(d_l1)).to(dtype)


def l1_mask_nchw(
    c: int, img_h: int, img_w: int, d_l1: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Flatten order NCHW: index i -> (ch, row, col); mask (D,D) with D=C*H*W."""
    hw = img_h * img_w
    dtot = c * hw
    idx = torch.arange(dtot, device=device)
    ch = idx // hw
    rem = idx % hw
    ry = rem // img_w
    cx = rem % img_w
    l1 = (
        (ch.unsqueeze(1) - ch.unsqueeze(0)).abs()
        + (ry.unsqueeze(1) - ry.unsqueeze(0)).abs()
        + (cx.unsqueeze(1) - cx.unsqueeze(0)).abs()
    )
    return (l1 <= int(d_l1)).to(dtype)


def build_cov_l1_masks(
    in_channels: int,
    img_h: int,
    img_w: int,
    *,
    cifar_channel_independent: bool,
    d_l1: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | list[torch.Tensor]:
    """Same structure as global K: one (HW,HW) per mode, or list for independent, or (D,D) joined."""
    hw = img_h * img_w
    if in_channels == 1:
        return l1_mask_hw_grid(hw, img_w, d_l1, device, dtype)
    if cifar_channel_independent:
        m = l1_mask_hw_grid(hw, img_w, d_l1, device, dtype)
        return [m.clone() for _ in range(in_channels)]
    return l1_mask_nchw(in_channels, img_h, img_w, d_l1, device, dtype)


def apply_cov_l1_mask(
    k: torch.Tensor | list[torch.Tensor],
    mask: torch.Tensor | list[torch.Tensor] | None,
    *,
    batched_k: bool,
) -> torch.Tensor | list[torch.Tensor]:
    """Element-wise multiply; batched_k True if k is (B,d,d)."""
    if mask is None:
        return k
    if isinstance(k, list):
        assert isinstance(mask, list) and len(k) == len(mask)
        out = []
        for kc, mc in zip(k, mask):
            if batched_k:
                out.append(kc * mc.unsqueeze(0))
            else:
                out.append(kc * mc)
        return out
    assert isinstance(mask, torch.Tensor)
    if batched_k:
        return k * mask.unsqueeze(0)
    return k * mask


def row_sparse_indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Build per-row sparse column indices from a 0/1 mask (D,D).

    Returns:
        idx: LongTensor of shape (D, M) with -1 padding, where M = max nnz per row.
             For each row i, valid columns are idx[i, :nnz_i].
    """
    if mask.dim() != 2 or mask.size(0) != mask.size(1):
        raise ValueError(f"mask must be square (D,D), got {tuple(mask.shape)}")
    d = int(mask.size(0))
    # Count nonzeros per row (assume mask is 0/1).
    nnz = (mask != 0).sum(dim=1).to(torch.long)
    m = int(nnz.max().item()) if d > 0 else 0
    idx = torch.full((d, m), -1, device=mask.device, dtype=torch.long)
    for i in range(d):
        cols = torch.nonzero(mask[i] != 0, as_tuple=False).squeeze(1).to(torch.long)
        n = int(cols.numel())
        if n > 0:
            idx[i, :n] = cols
    return idx
