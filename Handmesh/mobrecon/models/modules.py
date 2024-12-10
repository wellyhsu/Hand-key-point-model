# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file modules.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief Modules composing MobRecon
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""

import torch.nn as nn
import torch
from conv.spiralconv import SpiralConv


# Basic modules

class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        """Reorg layer to re-organize spatial dim and channel dim
        """
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0,4))

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        """Init a depth-wise sparable convolution

        Args:
            channel_in (int): input channel size
            channel_out (_type_): output channel size
            stride (int, optional): conv stride. Defaults to 1.
            has_half_out (bool, optional): whether output intermediate result. Defaults to False.
            num3x3 (int, optional): amount of 3x3 conv layer. Defaults to 1.
        """
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


def Pool(x, trans, dim=1):
    """Upsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN'xD
    """
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0)//3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
    return out2


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, meshconv=SpiralConv):
        """Init a spiral conv block

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(SpiralDeblock, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out

# Advanced modules
class Reg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Reg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        # 計算每個網格層級的頂點數量
        self.num_vert = [u[0].size(0)//3 for u in self.up_transform] + [self.up_transform[-1][0].size(0)//6]
        self.uv_channel = uv_channel
        # self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, bn=False, relu=False)
        # 初始化spiral解碼層
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                # 第一層使用相同的輸出維度和輸入維度
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                # 其餘層將輸出維度與下一層的輸入維度連接
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        # 定義最後一層的head，用於輸出 3D 頂點
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        # 初始化一個可學習的upsample矩陣
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel])*0.01, requires_grad=True)


    def index(self, feat, uv):
        """
        根據 UV 坐標從特徵圖中取樣

        Args:
            feat (tensor): 輸入特徵圖
            uv (tensor): UV 坐標

        Returns:
            samples: UV 坐標對應的特徵值
        """
        # 增加一個維度，形狀為 [B, N, 1, 2]
        uv = uv.unsqueeze(2)  
        # 從特徵圖中取樣 [B, C, N, 1]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        # 去掉多餘的維度，形狀為 [B, C, N]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        """
        前向傳播
        Args:
            uv (tensor): UV 坐標
            x (tensor): 編碼後的特徵圖

        Returns:
            pred: 解碼後的 3D 頂點位置
        """
        # 將 UV 坐標值調整到 [-1, 1] 範圍
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        # x = self.de_layer_conv(x)
        # 通過 UV 坐標在特徵圖中取樣，並轉置維度
        x = self.index(x, uv).permute(0, 2, 1)
        # 通過上採樣矩陣對取樣後的特徵進行上採樣
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        # 遍歷sparial解碼層進行逐層解碼
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        # 通過head獲取最終的 3D 頂點預測結果
        pred = self.head(x)

        return pred
