# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the INP-Former implementation.

This module implements the core INP-Former model with encoder, INP Extractor,
bottleneck, and INP-Guided Decoder components.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import GaussianBlur2d
from anomalib.models.components.dinov2 import DinoV2Loader
from anomalib.models.image.inp_former.components import (
    Aggregation_Block,
    Mlp,
    Prototype_Block,
    SoftMiningLoss,
)

# Encoder architecture configurations for DINOv2 models
DINOV2_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
}

# Default fusion layer configurations
DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]

# Default values for inference processing
DEFAULT_RESIZE_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_MAX_RATIO = 0.01

# Transformer architecture constants
TRANSFORMER_CONFIG: dict[str, float | bool] = {
    "mlp_ratio": 4.0,
    "layer_norm_eps": 1e-8,
    "qkv_bias": True,
    "attn_drop": 0.0,
}


class INP_FormerModel(nn.Module):
    """INP-Former model for anomaly detection.

    This model implements the INP-Former architecture with the following components:
    1. Encoder: DINOv2 Vision Transformer for feature extraction
    2. INP Extractor: Aggregation block to extract intrinsic normal prototypes
    3. Bottleneck: MLP layer for feature compression
    4. INP-Guided Decoder: Prototype blocks to reconstruct normal features

    Args:
        encoder_name (str): Name of the DINOv2 encoder to use.
        inp_num (int): Number of intrinsic normal prototypes.
        bottleneck_dropout (float): Dropout rate for the bottleneck MLP.
        decoder_depth (int): Number of decoder layers.
        target_layers (list[int] | None): List of encoder layers to extract features from.
        fuse_layer_encoder (list[list[int]] | None): Layer groupings for encoder feature fusion.
        fuse_layer_decoder (list[list[int]] | None): Layer groupings for decoder feature fusion.
        remove_class_token (bool): Whether to remove class token from features.
        encoder_require_grad_layer (list[int]): List of encoder layers to keep trainable.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        inp_num: int = 6,
        bottleneck_dropout: float = 0.0,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        encoder_require_grad_layer: list[int] = [],
    ) -> None:
        super().__init__()

        # Set default values
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        if fuse_layer_encoder is None:
            fuse_layer_encoder = DEFAULT_FUSE_LAYERS
        if fuse_layer_decoder is None:
            fuse_layer_decoder = DEFAULT_FUSE_LAYERS

        self.encoder_name = encoder_name
        # Load DINOv2 encoder
        encoder = DinoV2Loader().load(encoder_name)

        # Extract architecture configuration
        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]
        target_layers = arch_config["target_layers"]

        # Initialize INP tokens
        prototype_token = []
        for _ in range(inp_num):
            token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            prototype_token.append(token)
        self.prototype_token = nn.ParameterList(prototype_token)

        # Build INP Extractor (Aggregation Block)
        aggregation = []
        aggregation_block = Aggregation_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=TRANSFORMER_CONFIG["mlp_ratio"],
            qkv_bias=TRANSFORMER_CONFIG["qkv_bias"],
            norm_layer=partial(nn.LayerNorm, eps=TRANSFORMER_CONFIG["layer_norm_eps"]),
            attn_drop=TRANSFORMER_CONFIG["attn_drop"],
        )
        aggregation.append(aggregation_block)
        aggregation = nn.ModuleList(aggregation)

        # Build Bottleneck
        bottleneck = []
        bottleneck_mlp = Mlp(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            drop=bottleneck_dropout,
        )
        bottleneck.append(bottleneck_mlp)
        bottleneck = nn.ModuleList(bottleneck)

        # Build INP-Guided Decoder
        decoder = []
        for _ in range(decoder_depth):
            decoder_block = Prototype_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=TRANSFORMER_CONFIG["mlp_ratio"],
                qkv_bias=TRANSFORMER_CONFIG["qkv_bias"],
                norm_layer=partial(nn.LayerNorm, eps=TRANSFORMER_CONFIG["layer_norm_eps"]),
                attn_drop=TRANSFORMER_CONFIG["attn_drop"],
            )
            decoder.append(decoder_block)
        decoder = nn.ModuleList(decoder)

        # Assign components
        self.encoder = encoder
        self.aggregation = aggregation
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        # Add num_register_tokens attribute if missing
        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

        # Initialize Gaussian blur for anomaly map smoothing
        self.gaussian_blur = GaussianBlur2d(
            sigma=DEFAULT_GAUSSIAN_SIGMA,
            channels=1,
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        )

        # Loss functions
        self.soft_mining_loss = SoftMiningLoss()

    def _get_architecture_config(self, encoder_name: str, target_layers: list[int] | None) -> dict:
        """Get architecture configuration based on encoder name.

        Args:
            encoder_name: Name of the encoder model
            target_layers: Override target layers if provided

        Returns:
            Dictionary containing embed_dim, num_heads, and target_layers
        """
        for arch_name, config in DINOV2_ARCHITECTURES.items():
            if arch_name in encoder_name:
                result = config.copy()
                if target_layers is not None:
                    result["target_layers"] = target_layers
                return result
        raise ValueError(f"Architecture not supported for encoder: {encoder_name}")

    def gather_loss(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Calculate INP Coherence Loss.

        Args:
            query: Input features
            keys: INP prototypes

        Returns:
            Calculated loss value
        """
        similarity = F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance = 1.0 - similarity
        min_distance, _ = torch.min(distance, dim=2)
        return min_distance.mean()

    def _fuse_feature(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature tensors by averaging.

        Args:
            feat_list: List of feature tensors to fuse

        Returns:
            Averaged feature tensor
        """
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _process_features_for_spatial_output(
        self,
        features: list[torch.Tensor],
        h_patches: int,
        w_patches: int,
    ) -> list[torch.Tensor]:
        """Process features for spatial output by removing tokens and reshaping.

        Args:
            features: List of feature tensors
            h_patches: Number of patches in height dimension
            w_patches: Number of patches in width dimension

        Returns:
            List of processed feature tensors with spatial dimensions
        """
        # Remove class token and register tokens if not already removed
        if not self.remove_class_token:
            features = [f[:, 1 + self.encoder.num_register_tokens :, :] for f in features]

        # Reshape to spatial dimensions
        batch_size = features[0].shape[0]
        return [
            f.permute(0, 2, 1).reshape([batch_size, -1, h_patches, w_patches]).contiguous()
            for f in features
        ]

    def calculate_anomaly_maps(
        self,
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Calculate anomaly maps by comparing encoder and decoder features.

        Args:
            source_feature_maps: List of encoder feature maps
            target_feature_maps: List of decoder feature maps
            out_size: Output size for anomaly maps

        Returns:
            Tuple containing combined anomaly map and list of individual maps
        """
        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)

        anomaly_map_list = []
        for i in range(len(target_feature_maps)):
            fs = source_feature_maps[i]
            ft = target_feature_maps[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            anomaly_map_list.append(a_map)
        anomaly_map = torch.cat(anomaly_map_list, dim=1).mean(dim=1, keepdim=True)
        return anomaly_map, anomaly_map_list

    def forward(self, x: torch.Tensor, global_step: int | None = None) -> torch.Tensor | InferenceBatch:
        """Forward pass of the INP-Former model.

        Args:
            x: Input batch of images
            global_step: Current training step

        Returns:
            During training: Total loss
            During inference: InferenceBatch with anomaly scores and maps
        """
        # Prepare tokens through encoder
        image_size = (x.shape[2], x.shape[3])
        h_patches = x.shape[2] // self.encoder.patch_size
        w_patches = x.shape[3] // self.encoder.patch_size
        x = self.encoder.prepare_tokens_with_masks(x)
        B, L, _ = x.shape

        # Extract encoder features from target layers
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)

        # Remove class token if specified
        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en_list]

        # Fuse encoder features
        x = self._fuse_feature(en_list)

        # Extract INPs through Aggregation
        agg_prototype = torch.cat([token for token in self.prototype_token], dim=1)
        agg_prototype = agg_prototype.repeat(B, 1, 1)
        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype, x)

        # Bottleneck processing
        for blk in self.bottleneck:
            x = blk(x)

        # INP-Guided Decoding
        de_list = []
        for blk in self.decoder:
            x = blk(x, agg_prototype)
            de_list.append(x)
        # Fuse features for spatial output
        decoder_depth = len(self.decoder)
        en = [self._fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([de_list[decoder_depth - 1 - idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        # Process features for spatial output
        en = self._process_features_for_spatial_output(en, h_patches, w_patches)
        de = self._process_features_for_spatial_output(de, h_patches, w_patches)

        if self.training:
            # Calculate INP Coherence Loss
            g_loss = self.gather_loss(x, agg_prototype)
            # Calculate Soft Mining Loss
            loss = self.soft_mining_loss(en, de)
            # Total loss: Soft Mining Loss + 0.2 * INP Coherence Loss
            total_loss = loss + 0.2 * g_loss
            return total_loss

        # Inference mode: generate anomaly maps and scores
        anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=image_size)
        anomaly_map_resized = anomaly_map.clone()

        # Resize and smooth anomaly map
        if DEFAULT_RESIZE_SIZE is not None:
            anomaly_map = F.interpolate(
                anomaly_map, size=DEFAULT_RESIZE_SIZE, mode="bilinear", align_corners=False
            )
        anomaly_map = self.gaussian_blur(anomaly_map)

        # Calculate anomaly score
        if DEFAULT_MAX_RATIO == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            anomaly_map_flat = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][
                :, : int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO)
            ]
            sp_score = sp_score.mean(dim=1)

        return InferenceBatch(pred_score=sp_score, anomaly_map=anomaly_map_resized)