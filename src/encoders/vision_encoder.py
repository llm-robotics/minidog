"""Frozen vision encoders used as REPA alignment targets.

MiniDog uses DINOv2 (spec string ``dinov2-vit-<size>[flags]``, e.g.
``dinov2-vit-bnormworeg``). The base class keeps the interface generic so other
encoders can be added by subclassing ``VisionEncoder`` and registering the type
in ``ENCODER_REGISTRY``.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


class VisionEncoder(nn.Module):
    """Base class for frozen vision encoders."""

    def __init__(self, encoder_type: str, architecture: str, model_config: str,
                 device: torch.device, resolution: int = 256, accelerator=None):
        super().__init__()  # Initialize nn.Module
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim = None
        self.model = None
        self.patch_size = None  # Subclasses should set this

    def load_model(self):
        """Load and initialize the encoder model - subclasses should override"""
        raise NotImplementedError("Subclasses must implement load_model()")

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw images - subclasses should override
        Args:
            x: Raw images tensor (B, C, H, W) in range [0, 255]
        Returns:
            Preprocessed tensor ready for encoder
        """
        raise NotImplementedError("Subclasses must implement preprocess()")

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through encoder
        Args:
            x: Preprocessed images
        Returns:
            Dictionary with:
                - 'x_norm_clstoken': (B, D) CLS token or None if not available
                - 'x_norm_patchtokens': (B, T, D) patch tokens
        """
        # Default implementation - subclasses should override if needed
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        else:
            # Assume it's just patch tokens
            return {
                'x_norm_clstoken': None,
                'x_norm_patchtokens': out
            }

    def forward_features_multi(self, x: torch.Tensor, layer_indices: list) -> list:
        """Extract features from multiple layers.

        Default implementation for encoders that don't support intermediate layers.
        Runs forward_features once and returns the same result for each layer index.

        Args:
            x: Preprocessed input tensor
            layer_indices: List of layer indices (ignored for non-layer-supporting encoders)

        Returns:
            List of patch token tensors, one per layer index (all identical)
        """
        features = self.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        return [patch_tokens for _ in layer_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RAE-compatible forward pass returning only patch tokens.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch tokens (B, T, D)
        """
        x = self.preprocess(x)
        features = self.forward_features(x)
        return features['x_norm_patchtokens']

    @property
    def embed_dim(self) -> int:
        """Legacy property for compatibility"""
        return self._embed_dim

    @property
    def hidden_size(self) -> int:
        """RAE-compatible property (same as embed_dim)"""
        return self._embed_dim

    @property
    def supports_layer_idx(self) -> bool:
        """Whether this encoder supports extracting features from intermediate layers"""
        return False

    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, device):
        """Move model to device"""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class DINOv2Encoder(VisionEncoder):
    """DINOv2 encoder implementation.

    Supports optional flags in model_config: e.g., 'b[norm,woreg]'
      - Default (no flags): registers=True, norm_affine=False (matches legacy Dinov2withNorm)
      - [norm]: keep layernorm affine params
      - [woreg]: without register tokens
    """

    # Known flags that can appear in model_config after the base size letter
    _KNOWN_FLAGS = {'norm', 'woreg'}

    def _parse_config(self):
        """Parse model_config for base config and flags.

        Supports multiple formats:
            'b'              -> base='b', flags=set()
            'b[norm,woreg]'  -> base='b', flags={'norm','woreg'}  (bracket syntax)
            'bnormworeg'     -> base='b', flags={'norm','woreg'}  (concatenated suffix)
        """
        import re
        # Try bracket syntax first: e.g. 'b[norm,woreg]'
        match = re.match(r'^([a-z])(?:\[([^\]]+)\])?$', self.model_config)
        if match and match.group(2) is not None:
            base = match.group(1)
            flags = set(f.strip() for f in match.group(2).split(','))
            return base, flags

        # Try concatenated suffix syntax: e.g. 'bnorm', 'bworeg', 'bnormworeg'
        # First character is the size, rest is parsed for known flags
        cfg = self.model_config
        if len(cfg) >= 1 and cfg[0].isalpha():
            base = cfg[0]
            suffix = cfg[1:]
            if not suffix:
                return base, set()
            # Greedily match known flags from the suffix
            flags = set()
            remaining = suffix
            while remaining:
                matched = False
                for flag in self._KNOWN_FLAGS:
                    if remaining.startswith(flag):
                        flags.add(flag)
                        remaining = remaining[len(flag):]
                        matched = True
                        break
                if not matched:
                    # Unknown suffix — return raw config as base
                    return self.model_config, set()
            return base, flags

        return self.model_config, set()

    def load_model(self):
        import timm

        # Parse config and flags
        base_config, flags = self._parse_config()

        # Default: registers=True, norm_affine=False (legacy behavior)
        use_reg = 'woreg' not in flags
        use_norm_affine = 'norm' in flags

        # Load model from torch hub
        model_name = f'dinov2_vit{base_config}14{"_reg" if use_reg else ""}'

        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()

        # Resample position embeddings if needed
        patch_resolution = self.resolution // 16
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )

        # Set embed dim and patch size
        self._embed_dim = self.model.embed_dim
        self.patch_size = 14  # DINOv2 models use patch size 14

        # Remove layernorm affine params by default (matches legacy normalize=True)
        if not use_norm_affine:
            # Replace with LayerNorm without affine params
            self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)

        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, 224 * self.resolution // 256, mode='bicubic')
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINOv2 returns a dictionary with cls and patch tokens
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken': out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens')
        }


# Registry mapping encoder types to classes
ENCODER_REGISTRY = {
    'dinov2': DINOv2Encoder,
}


def create_encoder(encoder_string: str, device: torch.device,
                   resolution: int = 256, accelerator=None) -> VisionEncoder:
    """
    Factory function to create encoder from string specification

    Args:
        encoder_string: Format "encoder_type-architecture-model_config"
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training

    Returns:
        VisionEncoder instance
    """
    parts = encoder_string.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid encoder string format: {encoder_string}. "
                        f"Expected format: encoder_type-architecture-model_config")

    encoder_type, architecture, model_config = parts

    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available types: {list(ENCODER_REGISTRY.keys())}")

    encoder_class = ENCODER_REGISTRY[encoder_type]
    encoder = encoder_class(encoder_type, architecture, model_config,
                            device, resolution, accelerator)
    encoder.load_model()

    return encoder


@torch.no_grad()
def load_encoders(enc_type: str, device: torch.device, resolution: int = 256,
                  accelerator=None) -> List[VisionEncoder]:
    """
    Load multiple encoders from comma-separated string

    Args:
        enc_type: Comma-separated encoder specifications
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training

    Returns:
        List of VisionEncoder instances
    """
    enc_names = enc_type.split(',')
    encoders = []

    for enc_name in enc_names:
        # Parse encoder specification
        parts = enc_name.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid encoder format: {enc_name}")

        encoder = create_encoder(enc_name, device, resolution, accelerator)
        encoder.eval()
        encoders.append(encoder)

    return encoders
