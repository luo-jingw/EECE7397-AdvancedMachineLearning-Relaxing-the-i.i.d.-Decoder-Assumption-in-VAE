import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """4-layer Conv encoder + 4-layer decoder; MNIST 28x28 is padded to 32x32 internally.
    Decoder output: linear conv then sigmoid to [0, 1].
    """

    def __init__(
        self,
        in_channels: int,
        img_h: int,
        img_w: int,
        latent_dim: int,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if img_h != img_w:
            raise ValueError("img_h must equal img_w")
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.img_h = img_h
        self.img_w = img_w
        self.base_channels = base_channels
        self.dropout = dropout
        self._pad_to_32 = img_h == 28 and img_w == 28

        b = base_channels

        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, b, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(b, b * 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(b * 2, b * 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(b * 4, b * 4, 3, 1, 1),
            nn.GELU(),
        )
        self.enc_spatial = 8
        self.enc_flat_dim = (b * 4) * self.enc_spatial * self.enc_spatial
        self.enc_fc = nn.Linear(self.enc_flat_dim, latent_dim * 2)

        self.dec_fc = nn.Linear(latent_dim, self.enc_flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(b * 4, b * 2, 4, 2, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(b * 2, b, 4, 2, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(b, b, 3, 1, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(b, in_channels, 3, 1, 1),
        )

    def _maybe_pad_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad_to_32:
            return F.pad(x, (2, 2, 2, 2))
        return x

    def _maybe_crop_output(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad_to_32:
            return x[:, :, 2:30, 2:30]
        return x

    @staticmethod
    def _normalize_image_output(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._maybe_pad_input(x)
        h = self.enc_conv(x)
        h = h.flatten(1)
        stats = self.enc_fc(h)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        h = self.dec_fc(z).view(b, -1, self.enc_spatial, self.enc_spatial)
        x = self.dec_conv(h)
        x = self._maybe_crop_output(x)
        return self._normalize_image_output(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def load_vae_state_dict(model: VAE, state: dict[str, torch.Tensor]) -> None:
    """加载 VAE 权重；state 中多余键（旧实验）会被忽略。"""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError("Checkpoint missing required parameters: " + ", ".join(sorted(missing)))
