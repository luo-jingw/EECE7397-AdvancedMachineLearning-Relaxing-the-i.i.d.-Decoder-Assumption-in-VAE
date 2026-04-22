"""Online full covariance K with EMA; Mahalanobis recon loss (K detached, diag-mean norm only in loss)."""
from __future__ import annotations

import torch


def init_global_k(
    in_channels: int,
    img_h: int,
    img_w: int,
    *,
    cifar_channel_independent: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | list[torch.Tensor]:
    """Same layout as analysis K: MNIST / CIFAR-joined -> one (d,d); CIFAR independent -> C copies of (HW,HW)."""
    hw = img_h * img_w
    eye_hw = torch.eye(hw, device=device, dtype=dtype)
    if in_channels == 1:
        return eye_hw
    if cifar_channel_independent:
        return [eye_hw.clone() for _ in range(in_channels)]
    d = in_channels * hw
    return torch.eye(d, device=device, dtype=dtype)


def init_global_rrt_second_moment(
    in_channels: int,
    img_h: int,
    img_w: int,
    *,
    cifar_channel_independent: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | list[torch.Tensor]:
    """Init EMA estimate of E[r r^T]. With init K=I and mean=0, E[rr^T] starts as I."""
    return init_global_k(
        in_channels,
        img_h,
        img_w,
        cifar_channel_independent=cifar_channel_independent,
        device=device,
        dtype=dtype,
    )


def ema_update_global_k(
    global_k: torch.Tensor | list[torch.Tensor],
    global_rrt: torch.Tensor | list[torch.Tensor],
    r_det: torch.Tensor,
    ema: float,
    *,
    in_channels: int,
    img_h: int,
    img_w: int,
    cifar_channel_independent: bool,
    cov_l1_mask: torch.Tensor | list[torch.Tensor] | None = None,
) -> None:
    """
    Update global second moment K to estimate E[r r^T] over the full training distribution.

    Maintains EMA estimate:
      - global_rrt ≈ E[r r^T]
    and sets:
      - global_k := global_rrt
    """
    b = int(r_det.size(0))
    if b < 1:
        return
    one_m = 1.0 - float(ema)

    def _sym_inplace(K: torch.Tensor) -> None:
        K.copy_(0.5 * (K + K.T))

    def _apply_mask(mat: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
        if m is None:
            return mat
        return mat * m

    with torch.no_grad():
        if in_channels == 1:
            rd = r_det.reshape(b, -1)
            sb = (rd.T @ rd) / float(b)  # (D,D)  E_B[rr^T]
            assert isinstance(global_k, torch.Tensor)
            assert isinstance(global_rrt, torch.Tensor)
            mask = cov_l1_mask if isinstance(cov_l1_mask, torch.Tensor) else None

            global_rrt.mul_(ema).add_(sb, alpha=one_m)

            global_k.copy_(global_rrt)
            _sym_inplace(global_k)
            if mask is not None:
                global_k.copy_(_apply_mask(global_k, mask))
        elif cifar_channel_independent:
            assert isinstance(global_k, list)
            assert isinstance(global_rrt, list)
            for c in range(in_channels):
                rd = r_det[:, c, :, :].reshape(b, -1)
                sb = (rd.T @ rd) / float(b)
                mask = cov_l1_mask[c] if isinstance(cov_l1_mask, list) else None

                global_rrt[c].mul_(ema).add_(sb, alpha=one_m)

                global_k[c].copy_(global_rrt[c])
                _sym_inplace(global_k[c])
                if mask is not None:
                    global_k[c].copy_(_apply_mask(global_k[c], mask))
        else:
            rd = r_det.reshape(b, -1)
            sb = (rd.T @ rd) / float(b)
            assert isinstance(global_k, torch.Tensor)
            assert isinstance(global_rrt, torch.Tensor)
            mask = cov_l1_mask if isinstance(cov_l1_mask, torch.Tensor) else None

            global_rrt.mul_(ema).add_(sb, alpha=one_m)

            global_k.copy_(global_rrt)
            _sym_inplace(global_k)
            if mask is not None:
                global_k.copy_(_apply_mask(global_k, mask))


def _half_mahalanobis_mean(
    r_flat: torch.Tensor,
    sigma: torch.Tensor,
    ridge: float,
    cov_l1_mask: torch.Tensor | None = None,
    *,
    inv_trace_norm: bool = False,
) -> torch.Tensor:
    """
    Mean over batch of 0.5 * r^T Q r with Q = A^{-1} or optionally trace-normalized A^{-1}.

        A = sigma + ridge * I   (ridge is absolute, not scaled by mean(diag(sigma)))
        K^{-1} := A^{-1}

    If inv_trace_norm is True:
        \\tilde{K}^{-1} = (n / tr(K^{-1})) * K^{-1}   (n = matrix size D, so tr(\\tilde{K}^{-1}) = n).

    sigma is detached inside; gradients only through r_flat.
    If cov_l1_mask is set, element-wise multiply sigma (off-support zeros) before building A.
    """
    sig = sigma.detach()
    if cov_l1_mask is not None:
        sig = sig * cov_l1_mask
    d = sig.size(0)
    n = float(d)
    a = sig + float(ridge) * torch.eye(d, device=sig.device, dtype=sig.dtype)

    # Compute r^T A^{-1} r via solve (avoid forming inverse).
    sol = torch.linalg.solve(a, r_flat.T)  # (D,B)
    quad_raw = (r_flat * sol.T).sum(dim=1)  # (B,)

    if not inv_trace_norm:
        quad = quad_raw
    else:
        # \\tilde{K}^{-1} = (n / tr(K^{-1})) K^{-1}; exact trace O(D^2), Hutchinson for large D.
        if d <= 2048:
            a_inv = torch.linalg.inv(a)
            tr_ainv = a_inv.diagonal().sum().clamp(min=1e-8)
        else:
            num_probes = 32
            v = torch.empty((d, num_probes), device=a.device, dtype=a.dtype).bernoulli_(0.5).mul_(2).sub_(1)
            av = torch.linalg.solve(a, v)  # (D, num_probes)
            tr_ainv = (v * av).sum(dim=0).mean().clamp(min=1e-8)
        scale = n / tr_ainv
        quad = quad_raw * scale
    return 0.5 * quad.mean()


def recon_mahalanobis_loss(
    r: torch.Tensor,
    global_k: torch.Tensor | list[torch.Tensor],
    ridge: float,
    *,
    in_channels: int,
    cifar_channel_independent: bool,
    cov_l1_mask: torch.Tensor | list[torch.Tensor] | None = None,
    inv_trace_norm: bool = False,
) -> torch.Tensor:
    """r: x - recon, shape (B,C,H,W). Matches original MSE structure: one term per sample, then mean batch."""
    if in_channels == 1:
        rf = r.reshape(r.size(0), -1)
        assert isinstance(global_k, torch.Tensor)
        m = cov_l1_mask if isinstance(cov_l1_mask, torch.Tensor) else None
        return _half_mahalanobis_mean(
            rf, global_k, ridge, cov_l1_mask=m, inv_trace_norm=inv_trace_norm
        )
    if cifar_channel_independent:
        assert isinstance(global_k, list)
        total = torch.zeros((), device=r.device, dtype=r.dtype)
        for c in range(in_channels):
            mc = cov_l1_mask[c] if isinstance(cov_l1_mask, list) else None
            total = total + _half_mahalanobis_mean(
                r[:, c, :, :].reshape(r.size(0), -1),
                global_k[c],
                ridge,
                cov_l1_mask=mc,
                inv_trace_norm=inv_trace_norm,
            )
        return total
    rf = r.reshape(r.size(0), -1)
    assert isinstance(global_k, torch.Tensor)
    m = cov_l1_mask if isinstance(cov_l1_mask, torch.Tensor) else None
    return _half_mahalanobis_mean(rf, global_k, ridge, cov_l1_mask=m, inv_trace_norm=inv_trace_norm)


def clone_global_k_to_cpu(
    global_k: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(global_k, list):
        return [k.detach().cpu().clone() for k in global_k]
    return global_k.detach().cpu().clone()


def load_global_k_to_device(
    payload: torch.Tensor | list[torch.Tensor],
    *,
    in_channels: int,
    cifar_channel_independent: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | list[torch.Tensor]:
    if in_channels == 1:
        assert isinstance(payload, torch.Tensor)
        return payload.to(device=device, dtype=dtype)
    if cifar_channel_independent:
        assert isinstance(payload, list)
        return [t.to(device=device, dtype=dtype) for t in payload]
    assert isinstance(payload, torch.Tensor)
    return payload.to(device=device, dtype=dtype)
