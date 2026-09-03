"""
VAE wrapper for Diffusers AutoencoderKL models.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn


@dataclass
class VAEConfig:
    """Configuration for a VAE type."""
    pretrained_path: str
    subfolder: Optional[str] = ""
    latent_channels: int = 16
    scaling_factor: Optional[float] = None  # None = read from vae.config
    shift_factor: Optional[float] = None    # None = read from vae.config
    downsample_factor: int = 8


# Pre-defined VAE configurations
VAE_CONFIGS: Dict[str, VAEConfig] = {
    "e2e-vavae": VAEConfig(
        pretrained_path="REPA-E/e2e-vavae-hf",
        latent_channels=32,
        downsample_factor=16,
    ),
    "e2e-invae": VAEConfig(
        pretrained_path="REPA-E/e2e-invae-hf",
        latent_channels=32,
        downsample_factor=16,
    ),
}


class VAE(nn.Module):
    """
    VAE wrapper for Diffusers AutoencoderKL that matches RAE interface.

    Supports Flux VAE, SD3.5 VAE etc. through config presets
    or any custom Diffusers-based VAE via pretrained_path.

    Example usage in config:
        stage_1:
          target: stage1.VAE
          params:
            vae_type: "flux"
            resolution: 256
            sample_mode: "mode"
    """

    def __init__(
        self,
        vae_type: str,
        resolution: int = 256,
        eps: float = 1e-5,
        sample_mode: Literal["sample", "mode"] = "mode",
    ):
        super().__init__()

        self.resolution = resolution
        self.eps = eps
        self.sample_mode = sample_mode

        config = VAE_CONFIGS[vae_type]
        self._pretrained_path = config.pretrained_path
        self._subfolder = config.subfolder
        self._latent_channels = config.latent_channels
        self._config_scaling_factor = config.scaling_factor
        self._config_shift_factor = config.shift_factor
        self._downsample_factor = config.downsample_factor

        self._load_vae()

        if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
            self.register_buffer('shift_factor', torch.tensor(self.vae.config.latents_mean).reshape(1, -1, 1, 1))
        elif self._config_shift_factor is not None:
            self.shift_factor = self._config_shift_factor
        else:
            self.shift_factor = getattr(self.vae.config, 'shift_factor', 0.0)

        if hasattr(self.vae.config, 'latents_std') and self.vae.config.latents_std is not None:
            self.register_buffer('scaling_factor', 1 / torch.tensor(self.vae.config.latents_std).reshape(1, -1, 1, 1))
        elif self._config_scaling_factor is not None:
            self.scaling_factor = self._config_scaling_factor
        else:
            self.scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)

    def _load_vae(self):
        """Load the VAE model from pretrained weights."""
        from diffusers import AutoencoderKL

        load_kwargs = {"subfolder": self._subfolder}
        self.vae = AutoencoderKL.from_pretrained(self._pretrained_path, **load_kwargs).eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    @property
    def latent_dim(self) -> int:
        """Return latent channels for compatibility with RAE interface."""
        return self._latent_channels

    @property
    def patch_size(self) -> int:
        """Return effective patch size (downsample factor) for compatibility."""
        return self._downsample_factor

    @property
    def hidden_size(self) -> int:
        """Alias for latent_dim for compatibility."""
        return self._latent_channels

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images for VAE.

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Images in [-1, 1] range, resized to target resolution
        """
        # Resize if needed
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False
            )

        # Convert from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        return x

    def _vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist

    def _vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z).sample

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latents.

        Args:
            x: Images in [0, 1] range, shape (B, 3, H, W)

        Returns:
            Latents in shape (B, C, H, W) where C=latent_channels, H=W=latent_size
        """
        x = self._preprocess(x)
        posterior = self._vae_encode(x)
        if self.sample_mode == "sample":
            z = posterior.sample()
        else:
            z = posterior.mode()

        z = (z - self.shift_factor) * self.scaling_factor

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            z: Latents in shape (B, C, H, W)

        Returns:
            Images in [0, 1] range, shape (B, 3, H, W)
        """
        z = z / self.scaling_factor + self.shift_factor
        x = self._vae_decode(z)
        x = ((x + 1.0) / 2.0).clamp(0, 1)  # Convert from [-1, 1] to [0, 1]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode (for reconstruction)."""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
