# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Attention mechanisms for INP-Former model.

This module implements the two key attention mechanisms used in INP-Former:
1. Prototype_Attention: Used in the decoder to guide reconstruction using INPs
2. Aggregation_Attention: Used in the INP Extractor to aggregate features into INPs
"""

import torch
from torch import nn
import torch.nn.functional as F


class Prototype_Attention(nn.Module):
    """Prototype Attention mechanism for INP-Guided Decoder.

    This attention mechanism uses INPs as keys and values to guide the decoder's
    reconstruction process. It features learnable scaling factors and ReLU activation
    to suppress noise and focus on relevant normal patterns.

    Args:
        dim (int): Input feature dimension.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to use bias in QKV projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for attention scores. Defaults to None.
        attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout probability for output projection. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # Use learnable scaling factor instead of fixed scale
        self.learn_scale = nn.Parameter(nn.init.ones_(nn.Parameter(torch.zeros(num_heads, 1, 1))))
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prototype_token):
        """Forward pass through Prototype Attention.

        Args:
            x (torch.Tensor): Input features from decoder (queries).
            prototype_token (torch.Tensor): INP prototypes (keys and values).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output features and attention weights.
        """
        B, N, C = x.shape
        prototype_num = prototype_token.shape[1]

        # Project queries, keys, and values
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(prototype_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # L2 normalization for queries and keys
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Calculate attention scores with learnable scaling
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        # Use ReLU instead of Softmax to suppress weak correlations
        attn = F.relu(attn)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Aggregation_Attention(nn.Module):
    """Aggregation Attention mechanism for INP Extractor.

    This attention mechanism aggregates image features into INP prototypes using
    learnable tokens as queries and image features as keys and values.

    Args:
        dim (int): Input feature dimension.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to use bias in QKV projections. Defaults to False.
        qk_scale (float, optional): Scaling factor for attention scores. Defaults to None.
        attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout probability for output projection. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """Forward pass through Aggregation Attention.

        Args:
            x (torch.Tensor): Learnable tokens (queries).
            y (torch.Tensor): Image features (keys and values).

        Returns:
            torch.Tensor: Aggregated INP prototypes.
        """
        B, T, C = x.shape
        _, N, _ = y.shape

        # Project queries, keys, and values
        q = self.q(x).reshape(B, T, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Calculate attention scores with softmax normalization
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attnmap = attn.softmax(dim=-1)
        attn = self.attn_drop(attnmap)

        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x