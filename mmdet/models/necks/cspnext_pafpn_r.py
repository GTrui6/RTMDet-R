# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..layers import CSPLayer,E_CSPLayer

class Prefusion(nn.Module):
    def __init__(self,in_channels: Sequence[int],
        out_channels: int,
        num_csp_blocks: int = 1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        conv_cfg: bool = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        ):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.bil = nn.Upsample()
        conv =  ConvModule
        self.upconv = conv(
                    512,
                    128,
                    3,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
        self.downconv = conv(
                    512,
                    512,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
        self.convfuse = conv(
                    896,
                    512,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
    
    def forward(self, inputs: Tuple[Tensor, ...]):
        outs = []
        x_0 = inputs[0]
        x_1 = inputs[1]
        x_2 = inputs[2]

        size = x_1.shape[2:]
        size_2 = x_0.shape[2:]
        x_0 = self.avg_pool(x_0, size)
        outs.append(x_0)
        outs.append(x_1)
        x_2 = F.interpolate(x_2, size, mode='bilinear', align_corners=False)
        outs.append(x_2)
        cat_out = self.convfuse(torch.cat(outs, 1))
        outs[1] = inputs[1]
        outs[2] = self.downconv(cat_out)+inputs[2]
        outs[0] = inputs[0] + F.interpolate(self.upconv(cat_out), size=size_2, mode='bilinear', align_corners=False)
        return tuple(outs)
    
@MODELS.register_module()
class CSPNeXtPAFPNBDD_R(BaseModule):

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        conv_cfg: bool = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                conv(
                    in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.prefusion = Prefusion(in_channels=in_channels,
        out_channels=out_channels,
        num_csp_blocks=num_csp_blocks,
        use_depthwise=use_depthwise,
        expand_ratio=expand_ratio,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.prefusion(inputs)
        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)
            if upsample_feat.shape[2] == 46:
                upsample_feat = F.interpolate(upsample_feat, (45,80), mode="bilinear", align_corners=False)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
