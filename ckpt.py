"""从 train.py 保存的 checkpoint 构建 VAE（供 vis / 协方差实验等复用）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model import VAE, load_vae_state_dict


def load_checkpoint_dict(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def vae_meta_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": ckpt["dataset"],
        "in_channels": int(ckpt["in_channels"]),
        "img_h": int(ckpt["img_h"]),
        "img_w": int(ckpt["img_w"]),
        "latent_dim": int(ckpt["latent_dim"]),
        "base_channels": int(ckpt["base_channels"]),
    }


def build_vae_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> tuple[VAE, dict[str, Any]]:
    meta = vae_meta_from_checkpoint(ckpt)
    model = VAE(
        in_channels=meta["in_channels"],
        img_h=meta["img_h"],
        img_w=meta["img_w"],
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
    ).to(device)
    load_vae_state_dict(model, ckpt["model_state"])
    model.eval()
    return model, meta


def load_vae_from_path(ckpt_path: str | Path, device: torch.device) -> tuple[VAE, dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint_dict(ckpt_path, device)
    model, meta = build_vae_from_checkpoint(ckpt, device)
    return model, meta, ckpt
