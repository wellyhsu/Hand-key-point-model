# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file densestack.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief DenseStack
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append('/media/Pluto/Hao/HandMesh_origin/mobrecon')

import torchvision.models as models
import torch
import torch.nn as nn
from mobrecon.models.modules import conv_layer, mobile_unit, linear_layer, Reorg
import os
import numpy as np

class DenseBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in//4)
        self.conv2 = mobile_unit(channel_in*5//4, channel_in//4)
        self.conv3 = mobile_unit(channel_in*6//4, channel_in//4)
        self.conv4 = mobile_unit(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3),dim=1)
        out4 = self.conv4(comb3)
        comb4 = torch.cat((comb3, out4),dim=1)
        return comb4


class DenseBlock2(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in//2)
        self.conv2 = mobile_unit(channel_in*3//2, channel_in//2)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        return comb2


class DenseBlock3(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock3, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in)
        self.conv2 = mobile_unit(channel_in*2, channel_in)
        self.conv3 = mobile_unit(channel_in*3, channel_in)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2),dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3),dim=1)
        return comb3


class DenseBlock2_noExpand(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2_noExpand, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in*3//4)
        self.conv2 = mobile_unit(channel_in*7//4, channel_in//4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1),dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((out1, out2),dim=1)
        return comb2


class SenetBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel, size):
        super(SenetBlock, self).__init__()
        self.size = size
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel = channel
        self.fc1 = linear_layer(self.channel, min(self.channel//2, 256))
        self.fc2 = linear_layer(min(self.channel//2, 256), self.channel, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_out = x
        pool = self.globalAvgPool(x)
        pool = pool.view(pool.size(0), -1)
        fc1 = self.fc1(pool)
        out = self.fc2(fc1)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)

        return out * original_out


class DenseStack(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel):
        super(DenseStack, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel*2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel*2)
        self.senet2 = SenetBlock(input_channel*4,16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel*4)
        self.senet3 = SenetBlock(input_channel*8,8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel*8)
        self.dense5 = DenseBlock2_noExpand(input_channel*8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel*8, input_channel*4, num3x3=1), mobile_unit(input_channel*4, input_channel*4, num3x3=2))
        self.senet4 = SenetBlock(input_channel*4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink2 = nn.Sequential(mobile_unit(input_channel*4, input_channel*2, num3x3=1), mobile_unit(input_channel*2, input_channel*2, num3x3=2))
        self.senet5 = SenetBlock(input_channel*2,8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink3 = nn.Sequential(mobile_unit(input_channel*2, input_channel*2, num3x3=1), mobile_unit(input_channel*2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel,16)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.dense3(d2))
        u1 = self.upsample1(self.senet4(self.thrink1(d3)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.upsample3(self.senet6(self.thrink3(us2)))
        return u3


class DenseStack2(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel, final_upsample=True, ret_mid=False):
        super(DenseStack2, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel*2,32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel*2)
        self.senet2 = SenetBlock(input_channel*4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel*4)
        self.senet3 = SenetBlock(input_channel*8,8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel*8)
        self.dense5 = DenseBlock2_noExpand(input_channel*8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel*8, input_channel*4, num3x3=1), mobile_unit(input_channel*4, input_channel*4, num3x3=2))
        self.senet4 = SenetBlock(input_channel*4,4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink2 = nn.Sequential(mobile_unit(input_channel*4, input_channel*2, num3x3=1), mobile_unit(input_channel*2, input_channel*2, num3x3=2))
        self.senet5 = SenetBlock(input_channel*2,8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.thrink3 = nn.Sequential(mobile_unit(input_channel*2, input_channel*2, num3x3=1), mobile_unit(input_channel*2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel,16)
        self.final_upsample = final_upsample
        if self.final_upsample:
            self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ret_mid = ret_mid

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.senet3(self.dense3(d2)))
        d4 = self.dense5(self.dense4(d3))
        u1 = self.upsample1(self.senet4(self.thrink1(d4)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.senet6(self.thrink3(us2))
        if self.final_upsample:
            u3 = self.upsample3(u3)
        if self.ret_mid:
            return u3, u2, u1, d4
        else:
            return u3, d4


class DenseStack_Backnone(nn.Module):
    def __init__(self, input_channel=128, out_channel=24, latent_size=256, kpts_num=21, pretrain=True, control=None):
        """Init a DenseStack

        Args:
            input_channel (int, optional): the first-layer channel size. Defaults to 128.
            out_channel (int, optional): output channel size. Defaults to 24.
            latent_size (int, optional): middle-feature channel size. Defaults to 256.
            kpts_num (int, optional): amount of 2D landmark. Defaults to 21.
            pretrain (bool, optional): use pretrain weight or not. Defaults to True.
        """
        super(DenseStack_Backnone, self).__init__()
        self.control = control  # 儲存 control 參數
        
        # 初始化 DenseStack 的網路層
        self.pre_layer = nn.Sequential(
            # 初始卷積，將輸入影像特徵擴展
            conv_layer(3, input_channel // 2, 3, 2, 1),
            # 使用 MobileNet 單元進一步擴展
            mobile_unit(input_channel // 2, input_channel))
        # 壓縮通道維度
        self.thrink = conv_layer(input_channel * 4, input_channel)
        # 第一層 DenseStack
        self.dense_stack1 = DenseStack(input_channel, out_channel)
        # 重塑特徵圖
        self.stack1_remap = conv_layer(out_channel, out_channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 第二層 DenseStack 和投影層
        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, out_channel, final_upsample=False)
        self.mid_proj = conv_layer(1024, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)
        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(linear_layer(latent_size, 128, bn=False), linear_layer(128, 64, bn=False),
                                    linear_layer(64, 2, bn=False, relu=False))
        # 特徵重組操作
        self.reorg = Reorg()

        # 載入預訓練權重（如果需要）
        if pretrain:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            weight = torch.load(os.path.join(cur_dir, '../out/densestack.pth'))
            self.load_state_dict(weight, strict=False)
            print('Load pre-trained weight: densestack.pth')

        # 使用mobilenet_v2作為pretrained model
        mobile_netv2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.backbone2 = mobile_netv2.features

        # 使用mobilenet_v3_small作為pretrained model
        mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
        self.backbone3 = mobilenet_v3_small.features

        # mobilenet v2
        # 第一個分支
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1),  # (64, 512, 4, 4)
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),   # (64, 256, 4, 4)
        )
        
        # 第二個分支：將空間維度進一步減少，最終輸出 (64, 21, 2)
        self.conv_branch2 = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, stride=2, padding=1),  # (64, 512, 2, 2)
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),   # (64, 256, 2, 2)
            nn.Conv2d(256, 21, kernel_size=2, stride=1),               # (64, 21, 1, 1)
            nn.Flatten(),
            
        )
        self.fc = nn.Linear(21, 21 * 2)
        # 替换 MobileNet 中的第一个卷积层
        # mobile_net.features[0][0] = self.new_first_layer
        
        # mobilenet v3
        self.conv_branch3 = nn.Sequential(
            nn.Conv2d(576, 512, kernel_size=3, stride=1, padding=1),  # Change 576 to 256
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        
        # Adjusting the second convolutional branch input channels
        self.conv_branch4 = nn.Sequential(
            nn.Conv2d(576, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 21, kernel_size=1, stride=1),
        )
        self.fc = nn.Linear(4,2) # for mobileNet V3 small and Densestack
        # self.fc = nn.Linear(21,42) # for mobileNetV2



    def forward(self, x):
        #  Backbone is Densestack 
        if self.control == 'Densestack': 
            # 初始層提取特徵
            pre_out = self.pre_layer(x)
            # 特徵重組
            pre_out_reorg = self.reorg(pre_out)
            # 壓縮通道維度
            thrink = self.thrink(pre_out_reorg)
            # 第一層 DenseStack
            stack1_out = self.dense_stack1(thrink)
            # 特徵重塑
            stack1_out_remap = self.stack1_remap(stack1_out)
            # 合併特徵圖
            input2 = torch.cat((stack1_out_remap, thrink),dim=1)
            # 再次壓縮維度
            thrink2 = self.thrink2(input2)
            # 第二層 DenseStack
            stack2_out, stack2_mid = self.dense_stack2(thrink2)
            # 中間特徵投影
            latent = self.mid_proj(stack2_mid)
            # 2D 關鍵點預測
            uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))
            return latent, uv_reg


        # input shape: [1, 3, 128, 128]
        # latent shape:[1, 256, 4, 4]
        # uv_reg shape :[1, 21, 2] 

        # Backbone is mobilenet version 2
        # mobileNetV2_output = self.backbone2(x)
        # latent = self.conv_branch1[0](mobileNetV2_output)
        # latent = self.conv_branch1[1](latent)
        # uv_reg = self.conv_branch2(mobileNetV2_output)
        # uv_reg = self.fc(uv_reg)
        # uv_reg = uv_reg.view(x.size(0), 21, 2)

        # return latent, uv_reg

        # this is mobilenet version v3_small
        elif self.control == 'mobilenet_v3':
            # MobileNet V3 提取特徵
            mobileNetV3_output = self.backbone3(x)
            # 卷積分支處理
            latent = self.conv_branch3[0](mobileNetV3_output)
            latent = self.conv_branch3[1](latent)
            # 回歸分支處理
            uv_reg = self.conv_branch4(mobileNetV3_output)
            uv_reg = uv_reg.view(uv_reg.shape[0], uv_reg.shape[1], -1)
            # 線性層回歸
            uv_reg = self.fc(uv_reg)
            return latent, uv_reg

# 定義自訂的 block，接受 MobileNet 的輸出
class CustomConvBlock1(nn.Module):
    def __init__(self):
        super(CustomConvBlock1, self).__init__()
        
        # 第一個分支：保持 (64, 1280, 4, 4)，並將通道數減少到 (64, 256, 4, 4)
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1),  # (64, 512, 4, 4)
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),   # (64, 256, 4, 4)
        )
        
        # 第二個分支：將空間維度進一步減少，最終輸出 (64, 21, 2)
        self.conv_branch2 = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, stride=2, padding=1),  # (64, 512, 2, 2)
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),   # (64, 256, 2, 2)
            nn.Conv2d(256, 21, kernel_size=2, stride=1),               # (64, 21, 1, 1)
            nn.Flatten(),
            
        )
        self.fc = nn.Linear(21, 21 * 2)
        
# if __name__ == "__main__":
#     # x = torch.randn(64, 6, 4, 4)
    # model = DenseStack_Backnone()
#     # uv_reg, latent = model(x)
#     # print(f"this is the shape {uv_reg.shape}, {latent.shape} ,{x.shape}")
#         # 获取第一个卷积层
#     x = torch.randn(64, 6, 128, 128)
#     mobile_net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
#     mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
#     model3 = mobilenet_v3_small.features
#     model2 = mobile_net.features
#     first_conv_layer = model3[0][0]

# #     # 修改第一个卷积层以接受 6 个输入通道
#     new_first_layer = nn.Conv2d(6, first_conv_layer.out_channels, 
#                                 kernel_size=first_conv_layer.kernel_size, 
#                                 stride=first_conv_layer.stride, 
#                                 padding=first_conv_layer.padding, 
#                                 bias=first_conv_layer.bias is not None)

#     # 将新的卷积层的权重进行初始化（你可以使用已有的权重复制，或者随机初始化）
#     new_first_layer.weight.data[:, :3, :, :] = first_conv_layer.weight.data  # 复制前 3 个通道的权重
#     new_first_layer.weight.data[:, 3:, :, :] = first_conv_layer.weight.data.mean(dim=1, keepdim=True)  # 随机初始化其他通道的权重

#     # 替换 MobileNet 中的第一个卷积层
#     model3[0][0] = new_first_layer

#     y = model3(x)
#     print(x.shape)
#     print(y.shape)
    # 創建模型並運行示例
    # 模擬 MobileNet 輸出 (64, 1280, 4, 4)
    # x = torch.randn(64, 1280, 4, 4)

    # 創建模型並運行示例
    # model = CustomConvBlock()
    # conv_out1, conv_out2 = model(x)

    # print(conv_out1.shape)  # 應該輸出 (64, 256, 4, 4)
    # print(conv_out2.shape)  # 應該輸出 (64, 21, 2)