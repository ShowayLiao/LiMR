# Copyright (c) 2022 Alpha-VL G.Peng et al.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# ConvMAE: https://github.com/Alpha-VL/ConvMAE
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn


from .utils import get_2d_sincos_pos_embed
from torch.nn import functional as F
from .mobileViTv2.mobilevit_v2 import MobileViTv2
from .FPN_new import FPNFactory


class MaskedAutoencoderMobileViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, patch_size=16,
                 embed_dim=(64,128,256,384,512), norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 cfg = None,scale_factors=(8.0, 4.0, 2.0), FPN_output_dim=(384, 256, 128),alpha=1.0):
        super().__init__()
        # --------------------------------------------------------------------------
        # encoder specifics
        embed_dim = [int(x * alpha) for x in embed_dim]

        self.encoder = MobileViTv2(width_multiplier=alpha,cfg=cfg)

        self.norm = norm_layer(embed_dim[3])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # FPN decoder specifics
        decoder_embed_dim = embed_dim[3]

        self.decoder_FPN_pos_embed = nn.Parameter(torch.zeros(1,  decoder_embed_dim, 14, 14),
                                                  requires_grad=False)  # fixed sin-cos embedding

        self.decoder = FPNFactory.build(cfg.TRAIN.LiMR.decoder, embed_dim[::-1], FPN_output_dim[::-1], 4)

        # --------------------------------------------------------------------------

        self.cfg = cfg

        self.patch_size = patch_size

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_FPN_pos_embed.shape[1],
                                                    14, cls_token=False)

        decoder_pos_embed = decoder_pos_embed.reshape(14, 14, -1)  # [14, 14, C]
        decoder_pos_embed = torch.from_numpy(decoder_pos_embed).float()
        decoder_pos_embed = decoder_pos_embed.permute(2, 0, 1)  # [C, 14, 14]
        decoder_pos_embed = decoder_pos_embed.unsqueeze(0)  # [1, C, 14, 14]

        self.decoder_FPN_pos_embed.data.copy_(decoder_pos_embed)



        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = 7*7

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def mask2ids(self, mask,path_size = 2):
        # mask：B,C,H,W->B,C,P,N
        mask = F.unfold(
            mask,
            kernel_size=(path_size, path_size),
            stride=(path_size, path_size),
        )
        mask = mask.reshape(mask.shape[0], 1, path_size*path_size, -1)

        # 获取被保留和遮蔽的索引
        idx_keep = mask[0, 0, 0, :].nonzero(as_tuple=True)[0]  # 保留位置的索引
        idx_mask = (1 - mask[0, 0, 0, :]).nonzero(as_tuple=True)[0]  # 遮蔽位置的索引

        # 构造乱序及恢复索引
        ids_shuffle = torch.cat([idx_keep, idx_mask], dim=0)
        ids_restore = torch.argsort(ids_shuffle)

        return idx_keep, ids_restore

    def mask_upsample(self, mask, H_low, H_high, batch_size):
        up_mask = (1-mask.reshape(-1, H_high, H_high).
                   unsqueeze(-1).repeat(1, 1, 1, H_low ** 2 // H_high ** 2).
                   reshape(-1,H_high,H_high,H_low // H_high,H_low // H_high).
                   permute(0, 1, 3, 2, 4).
                   reshape(batch_size, H_low, H_low).
                   unsqueeze(1))
        up_keep, up_restore = self.mask2ids(up_mask)

        return up_keep.unsqueeze(0).repeat(batch_size,1), up_mask, up_restore.unsqueeze(0).repeat(batch_size,1)

    def mask_everylayer(self,x,mask_ratio):
        H,W = x.shape[2],x.shape[3]
        _, mask, _ = self.random_masking(x, mask_ratio)
        mask_for_patch5 = 1-mask.reshape(-1,7,7).unsqueeze(1)# last layer
        ids_keep_5,ids_restore_5 = self.mask2ids(mask_for_patch5,path_size=1)
        ids_keep_1,mask_for_patch1,ids_restore_1 = self.mask_upsample(mask,H//2,H//32,x.shape[0])# layer1
        ids_keep_2,mask_for_patch2,ids_restore_2 = self.mask_upsample(mask,H//4,H//32,x.shape[0])# layer2
        ids_keep_3,mask_for_patch3,ids_restore_3 = self.mask_upsample(mask,H//8,H//32,x.shape[0])# layer3
        ids_keep_4,mask_for_patch4,ids_restore_4 = self.mask_upsample(mask,H//16,H//32,x.shape[0])# layer4

        masks = [mask_for_patch1, mask_for_patch2, mask_for_patch3, mask_for_patch4, mask_for_patch5]
        ids_keep_list = [ids_keep_1, ids_keep_2, ids_keep_3, ids_keep_4,ids_keep_5.unsqueeze(0).repeat(x.shape[0], 1)]
        ids_restore_list = [ids_restore_1, ids_restore_2, ids_restore_3,ids_restore_4,
                            ids_restore_5.unsqueeze(0).repeat(x.shape[0], 1)]

        return masks,ids_keep_list,ids_restore_list


    # encoder
    def forward_encoder(self, x, mask_ratio):

        if self.cfg.TEST.enable:# for test phase, mask does not work
            layers = self.encoder(x)
            return layers,None,None
        else:
            masks, ids_keep_list, ids_restore_list = self.mask_everylayer(x, mask_ratio)

            layers = self.encoder(x, masks,ids_keep_list,ids_restore_list)

            return layers, masks, ids_restore_list

    # FPN decoder
    def forward_decoder(self, x,mask=None):
        results = self.decoder(x,mask)
        results = results[1:]

        return {layer: feature for layer, feature in zip(self.cfg.TRAIN.LiMR.layers_to_extract_from, results[::-1])}



    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore  = self.forward_encoder(imgs, mask_ratio)
        multilayers = self.forward_decoder(latent,mask)  # [N, L, p*p*3]
        # loss = self.forward_loss(imgs, pred, mask)
        return multilayers



def LiMR_base(**kwargs):
    model = MaskedAutoencoderMobileViT(patch_size=[2,2, 2, 2],norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


