import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence

from .linear_attention import LinearAttnFFN
from .BaseLayers import LayerNorm


# this class is modified from https://github.com/HowardLi0816/MobileViTv2_pytorch
# we integrate the masking unfolding and folding operations into the MobileViTv2 block
class MobileViTBlockv2(nn.Module):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.4,
        ffn_dropout: Optional[float] = 0.1,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        *args,
        **kwargs
    ) -> None:
        cnn_out_dim = attn_unit_dim

        conv_3x3_in = nn.Sequential(nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            dilation=dilation,
            groups=in_channels,
            # padding=(conv_ksize+(dilation-1)*(conv_ksize-1)-1)//2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=in_channels),
        nn.LeakyReLU(negative_slope=0.1))

        conv_1x1_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        super(MobileViTBlockv2, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
        )

        self.conv_proj = nn.Sequential(nn.Conv2d(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=in_channels))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize
        self.norm  = LayerNorm(normalized_shape=attn_unit_dim)

    def _build_attn_layer(
        self,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for block_idx in range(n_layers)
        ]
        # global_rep.append(
        #     LayerNorm(normalized_shape=d_model)
        # )

        return nn.Sequential(*global_rep), d_model



    def unfolding_pytorch(self, feature_map: Tensor, mask: Optional[Tensor] = None,idx_keep:Optional[Tensor] = None) -> Tuple[
        Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape

        # if mask is not None:
        #     pre_mask_channels = feature_map.shape[1]
        #     feature_map = torch.cat([feature_map, mask], dim=1)

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )


        if mask is not None:

            patches = patches.reshape(
                batch_size, in_channels, self.patch_h * self.patch_w, -1
            )

            # only input the visible patches to the transformer
            feature_patches = torch.gather(patches, 3,
                                           idx_keep.view([feature_map.shape[0], 1, 1, -1]).
                                           expand(batch_size, in_channels, self.patch_h * self.patch_w, -1))

        else:
            patches = patches.reshape(
                batch_size, in_channels , self.patch_h * self.patch_w, -1
            )
            feature_patches = patches
            mask_patches = None
            ids_restore = None

        return feature_patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int],ids_restore:Optional[Tensor]=None) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]=>[B, C*P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        if ids_restore is not None:
            mask_token = torch.zeros(
                (batch_size, in_dim * patch_size , ids_restore.shape[1] - n_patches),
                device=patches.device
            )
            patches = torch.cat([patches, mask_token], dim=2)
            patches = torch.gather(patches, 2, ids_restore.view([batch_size, 1, -1]).
                                   expand(batch_size, in_dim * patch_size, -1))

        # [B, C*P, N] --> [B, C*P, H, W]
        # onnx dosn't support fold, so we use reshape instead
        # feature_map = F.fold(
        #     patches,
        #     output_size=output_size,
        #     kernel_size=(self.patch_h, self.patch_w),
        #     stride=(self.patch_h, self.patch_w),
        # )


        # use reshape instead for onnx,ref:https://blog.csdn.net/xunan003/article/details/133752722
        # 计算平方根（转换为 Tensor）
        # 在tensorRT中，sqrt算子只能计算浮点数，而后面的reshape要求必须是整数，所以需要对结果进行转换
        sqrt_patch_size = torch.sqrt(torch.tensor(patch_size).to(torch.float32)).to(torch.int64)
        sqrt_n_patches = torch.sqrt(torch.tensor(n_patches).to(torch.float32)).to(torch.int64)

        feature_map = patches.reshape(
            batch_size, in_dim,
            sqrt_patch_size, sqrt_patch_size,
            sqrt_n_patches, sqrt_n_patches
        )
        feature_map = feature_map.permute(0,1,2,4,3,5)
        feature_map = feature_map.permute(0,1,3,2,5,4)
        feature_map = feature_map.reshape(batch_size, in_dim, output_size[0], output_size[1])

        return feature_map


    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)# 向上取整
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor,mask:Optional[Tensor] = None,idx_keep:Optional[Tensor] = None, ids_restore:Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        if mask is not None:
            x = x*mask
        fm_conv = self.local_rep(x)
        # B C H W

        # convert feature map to patches
        patches, output_size = self.unfolding_pytorch(fm_conv,mask,idx_keep)

        # learn global representations on all patches

        patches = self.global_rep(patches)# B C P N

        patches = self.norm(patches.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)



        # [B x C x P x N] --> [B x C x H x W]
        fm = self.folding_pytorch(patches=patches, output_size=output_size,ids_restore=ids_restore)

        output = self.conv_proj(fm)

        return output


    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], mask:Optional[Tensor] = None,idx_keep:Optional[Tensor] = None, ids_restore:Optional[Tensor] = None,*args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.forward_spatial(x,mask,idx_keep,ids_restore)

