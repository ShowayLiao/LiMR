# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss functions for INP-Former model.

This module implements the loss functions used in INP-Former:
1. INPCoherenceLoss: Ensures INPs faithfully represent normal features
2. SoftMiningLoss: Focuses training on difficult-to-reconstruct regions
"""

import torch
import torch.nn.functional as F


class INPCoherenceLoss(torch.nn.Module):
    """INP Coherence Loss to ensure INPs faithfully represent normal features.

    This loss calculates the mean minimum distance from each feature to the nearest INP,
    ensuring that INPs capture the normal patterns in the data.

    Args:
        reduction (str, optional): Reduction method for the loss. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Forward pass to calculate INP Coherence Loss.

        Args:
            query (torch.Tensor): Input features from the encoder.
            keys (torch.Tensor): INP prototypes.

        Returns:
            torch.Tensor: Calculated loss value.
        """
        # Calculate cosine similarity between each feature and all INPs
        similarity = F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        # Convert to distance and take minimum distance for each feature
        distance = 1.0 - similarity
        min_distance, _ = torch.min(distance, dim=2)
        # Calculate loss
        if self.reduction == "mean":
            loss = min_distance.mean()
        elif self.reduction == "sum":
            loss = min_distance.sum()
        else:
            loss = min_distance
        return loss


class SoftMiningLoss(torch.nn.Module):
    """Soft Mining Loss to focus on difficult-to-reconstruct regions.

    This loss adapts the gradient based on the reconstruction difficulty, giving more
    weight to regions that are harder to reconstruct.

    Args:
        gamma (float, optional): Exponent for the difficulty weight. Defaults to 3.0.
    """

    def __init__(self, gamma: float = 3.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, encoder_features: list[torch.Tensor], decoder_features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass to calculate Soft Mining Loss.

        Args:
            encoder_features (list[torch.Tensor]): List of encoder features.
            decoder_features (list[torch.Tensor]): List of decoder features.

        Returns:
            torch.Tensor: Calculated loss value.
        """
        total_loss = 0.0
        for item in range(len(encoder_features)):
            # Detach encoder features to prevent gradient flow
            en_ = encoder_features[item].detach()
            de_ = decoder_features[item]
            
            # Calculate point-wise cosine distance
            point_dist = 1.0 - F.cosine_similarity(en_, de_)
            # Calculate mean distance for normalization
            mean_dist = point_dist.mean()
            # Calculate difficulty weight factor
            factor = (point_dist / mean_dist) ** self.gamma
            
            # Apply factor to scale gradients without changing forward value
            # de_ * f + de_.detach() * (1-f) = de_ in forward, grad *= f in backward
            factor_d = factor.detach().unsqueeze(1)
            de_scaled = de_ * factor_d + de_.detach() * (1.0 - factor_d)
            
            # Calculate batch-wise cosine loss
            loss = 1.0 - F.cosine_similarity(
                en_.reshape(en_.shape[0], -1),
                de_scaled.reshape(de_scaled.shape[0], -1)
            )
            total_loss += loss.mean()
        
        # Average loss across all feature layers
        return total_loss / len(encoder_features)