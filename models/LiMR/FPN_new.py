import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d as dcn_v2


# copied from detectron2
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation



    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



# copied from https://github.com/JustlfC03/MFDS-DETR
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 4, flag=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)


# copied from https://github.com/JustlfC03/MFDS-DETR
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, flag=True):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)


# copied from https://github.com/JustlfC03/MFDS-DETR
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.group_norm1 = nn.GroupNorm(8, out_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.group_norm2 = nn.GroupNorm(8, out_chan)
        nn.init.xavier_uniform_(self.conv_atten.weight)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        atten = self.sigmoid(self.group_norm1(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.group_norm2(self.conv(x))
        return feat


# copied from https://github.com/JustlfC03/MFDS-DETR
class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc)
        self.offset = nn.Conv2d(out_nc +in_nc, 2*3*3, kernel_size=1, stride=1, padding=0)
        self.group_norm1 = nn.GroupNorm(9, 2*3*3)
        # self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                         extra_offset_mask=True)
        self.dcpack_L2 = dcn_v2(in_nc, out_nc, 3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.offset.weight)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.group_norm1(self.offset(torch.cat([feat_arm, feat_up * 2], dim=1)))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2(feat_up, offset))  # [feat, offset]

        return feat_align + feat_arm

# modified from https://github.com/JustlfC03/MFDS-DETR
class cFeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(cFeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.group_norm1 = nn.GroupNorm(8, out_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.group_norm2 = nn.GroupNorm(8, out_chan)
        nn.init.xavier_uniform_(self.conv_atten.weight)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        atten = self.sigmoid(self.group_norm1(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.group_norm2(self.conv(x))
        return feat

# modified from https://github.com/JustlfC03/MFDS-DETR
class cFeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc):
        super(cFeatureAlign_V2, self).__init__()
        # self.lateral_conv = cFeatureSelectionModule(in_nc, out_nc)
        self.offset = nn.Conv2d(out_nc + in_nc, 2 * 3 * 3, kernel_size=1, stride=1, padding=0)
        self.group_norm1 = nn.GroupNorm(9, 2 * 3 * 3)
        # self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                         extra_offset_mask=True)
        self.dcpack_L2 = dcn_v2(in_nc, out_nc, 3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        # self.SA = SpatialAttention()
        # self.fusion = nn.Conv2d(out_nc*2, out_nc, kernel_size=1)

        nn.init.xavier_uniform_(self.offset.weight)
        # nn.init.xavier_uniform_(self.fusion.weight)

    def forward(self, feat_l, feat_s, main_path=None):
        # HW = feat_l.size()[2:]
        # if feat_l.size()[2:] != feat_s.size()[2:]:
        #     feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        # else:
        #     feat_up = feat_s
        # feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.group_norm1(
            self.offset(torch.cat([feat_l, feat_s * 2], dim=1)))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2(feat_s, offset))  # [feat, offset]

        # return self.fusion(torch.cat([feat_arm, self.SA(feat_align)], dim=1)) + feat_arm + feat_align
        return feat_align

# copy from detectron2
class Conv_LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class BaseFPN(nn.Module):

    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim


        self.align_modules = nn.ModuleList()
        self.channel_att = ChannelAttention
        self.spatial_att = SpatialAttention

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        if x.size()[2:] != y.size()[2:]:
            x = F.interpolate(x, size=(h, w), mode='bilinear')
        # return F.interpolate(x, size=(h, w), mode='bilinear') + y
        return x

    def _dynamic_weight_fusion(self, high, low, weight_conv):

        return weight_conv(high) * low

class FPN(BaseFPN):

    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__(in_channels, hidden_dim, num_feature_levels)


        self.bottomup_conv = nn.ModuleList()
        self.conv_proj = nn.ModuleList()
        for _ in range(num_feature_levels-1):
            self.bottomup_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels[_], in_channels[_+1],  kernel_size=2, stride=2),
                    Conv_LayerNorm(in_channels[_+1]),
                    nn.GELU(),
                )
            )

        for _ in range(num_feature_levels):

            self.conv_proj.append(
                nn.Sequential(
                    Conv2d(
                        in_channels[_],
                        hidden_dim[_],
                        kernel_size=1,
                        bias=False,
                        norm=Conv_LayerNorm(hidden_dim[_]),
                    ),
                    Conv2d(
                        hidden_dim[_],
                        hidden_dim[_],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm=Conv_LayerNorm(hidden_dim[_]),
                        groups=hidden_dim[_]
                    ),

                )
            )

    def forward(self, srcs,masks=None):
        feature_maps = srcs[::-1]
        up_results = [feature_maps[0]]
        results = [up_results[0]]

        for feature, upconv in zip(feature_maps[1:], self.bottomup_conv):
            up_feature = self._upsample_add(upconv(up_results[-1]),feature)
            up_results.append(up_feature)
            results.append(up_feature)

        return [self.conv_proj[i](f) for i, f in enumerate(results)]

class mFPN(BaseFPN):

    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__(in_channels, hidden_dim, num_feature_levels)


        self.bottomup_conv = nn.ModuleList()
        self.conv_proj = nn.ModuleList()
        for _ in range(num_feature_levels-1):
            self.bottomup_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels[_], in_channels[_+1],  kernel_size=2, stride=2),
                    Conv_LayerNorm(in_channels[_+1]),
                    nn.GELU(),
                )
            )

        for _ in range(num_feature_levels):

            self.conv_proj.append(
                nn.Sequential(
                    Conv2d(
                        in_channels[_],
                        hidden_dim[_],
                        kernel_size=1,
                        bias=False,
                        norm=Conv_LayerNorm(hidden_dim[_]),
                    ),
                    Conv2d(
                        hidden_dim[_],
                        hidden_dim[_],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm=Conv_LayerNorm(hidden_dim[_]),
                        groups=hidden_dim[_]
                    ),

                )
            )

        self.FSMs = nn.ModuleList([
            nn.Sequential(
                Conv2d(in_channels[_+1], in_channels[_+1], kernel_size=3, padding=1,bias=False, norm=Conv_LayerNorm(in_channels[_+1]), groups=in_channels[_+1]),
                nn.GELU(),
            ) for _ in range(num_feature_levels-1)
        ])

    def forward(self, srcs,masks=None):
        feature_maps = srcs[::-1]
        masks = masks[::-1]

        up_results = [feature_maps[0]]
        results = [up_results[0]]


        for feature, upconv,conv,mask in zip(feature_maps[1:], self.bottomup_conv,self.FSMs,masks[1:]):
            up_feature = self._upsample_add(upconv(up_results[-1]),feature)
            up_results.append(up_feature)
            arm_feature = conv(feature)

            results.append(arm_feature * mask + up_feature)


        return [self.conv_proj[i](f) for i, f in enumerate(results)]

# implement replied on http://arxiv.org/abs/1803.01534
class PAFPN(BaseFPN):

    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__(in_channels, hidden_dim, num_feature_levels)

        self.upbottom_conv = nn.ModuleList()
        self.bottomup_conv = nn.ModuleList()
        for _ in range(num_feature_levels - 1):
            self.upbottom_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1),
                    nn.GroupNorm(32, hidden_dim)
                )
            )
            self.bottomup_conv.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim)
                )
            )

    def forward(self, srcs):
        feature_maps = srcs[::-1]
        up_features = [feature_maps[0]]

        # 自顶向下路径
        for feature, conv in zip(feature_maps[1:], self.upbottom_conv):
            up_feature = self._upsample_add(up_features[0], conv(feature))
            up_features.insert(0, up_feature)

        # 自底向上路径
        bottom_features = [up_features[0]]
        for i in range(1, len(up_features)):
            down_feature = self.bottomup_conv[i-1](bottom_features[0])
            fused_feature = down_feature + up_features[i]
            bottom_features.insert(0, fused_feature)

        return [self.fpn_proj[i](f) for i, f in enumerate(bottom_features[::-1])]


# implement replied on http://arxiv.org/abs/2108.07058
class FaPN(BaseFPN):

    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__(in_channels, hidden_dim, num_feature_levels)


        for _ in range(num_feature_levels - 1):
            self.align_modules.append(
                FeatureAlign_V2(in_channels[_], in_channels[_+1])
            )

        self.bottomup_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[_], hidden_dim[_], 1),
                nn.GroupNorm(8, hidden_dim[_]),
                nn.ReLU()
            ) for _ in range(num_feature_levels)
        ])


    def forward(self, srcs):

        feature_maps = srcs[::-1]
        results = [feature_maps[0]]

        for feature, align in zip(feature_maps[1:], self.align_modules):
            aligned_feature = align(feature, results[0])
            results.insert(0, aligned_feature)

        return [conv(f) for conv, f in zip(self.bottomup_conv, results[::-1])]

# modified from https://github.com/JustlfC03/MFDS-DETR
class WBCFPN(BaseFPN):


    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__(in_channels, hidden_dim, num_feature_levels)

        for _ in range(num_feature_levels - 1):
            self.align_modules.append(
                cFeatureAlign_V2(in_channels[_+1], in_channels[_+1])
            )


        self.up_sample_conv = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels[_], out_channels=in_channels[_+1], kernel_size=3, stride=2,
                                  padding=1, output_padding=1)
            ) for _ in range(num_feature_levels-1)
        ])


        self.FSMs = nn.ModuleList([
            nn.Sequential(
                self.channel_att(in_channels[_]),
                nn.Conv2d(in_channels[_], in_channels[_], 1)
            ) for _ in range(num_feature_levels)
        ])

        self.SFFs = nn.ModuleList([
            self.channel_att(in_channels[_+1], ratio=4, flag=False)
            for _ in range(num_feature_levels - 1)
        ])


        self.lateral_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[_], hidden_dim[_], 3, padding=1),
                nn.GroupNorm(32, hidden_dim[_]),
                nn.ReLU()
            ) for _ in range(num_feature_levels)
        ])

    def forward(self, srcs,masks=None):
        feature_maps = srcs[::-1]
        up_sample_features = []
        up_bottom_features = []

        up_bottom_features.append(self.FSMs[0](feature_maps[0]))
        up_sample_features.append(up_bottom_features[0])
        for up_sample in self.up_sample_conv:
            up_sample_features.insert(0, up_sample(up_sample_features[0]))
        up_sample_features = up_sample_features[::-1]


        for i, (feature, FSM, SFF) in enumerate(zip(feature_maps[1:], self.FSMs[1:], self.SFFs)):

            down_feature = FSM(feature)
            _, _, h, w = feature.shape


            high_feature = up_sample_features[i+1]
            if high_feature.shape[-1] != w or high_feature.shape[-2] != h:
                high_feature = F.upsample(high_feature, size=(h, w), mode='bilinear')
            align_feature = self.align_modules[i](down_feature, high_feature)
            select_down_feature = SFF(align_feature)*down_feature

            fusion_feature = select_down_feature + high_feature
            up_bottom_features.append(fusion_feature)


        return [conv(f) for conv, f in zip(self.lateral_conv, up_bottom_features)]

class FPNFactory:
    @staticmethod
    def build(method, in_channels, hidden_dim, num_feature_levels):
        method_map = {
            'fpn': FPN,
            'pafpn': PAFPN,
            'fapn': FaPN,
            'wbcfpn': WBCFPN,
            'mfpn': mFPN,
        }
        assert method in method_map, f"Unsupported FPN type: {method}"
        return method_map[method](in_channels, hidden_dim, num_feature_levels)

