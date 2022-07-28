import torch
import torch.nn.functional as F
from torch import nn




class metalearner(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer1 = torch.nn.Linear(2, 2)
        self.layer2 = torch.nn.Linear(2, 2)

    def forward(self, grad_x):
        grad_x = F.relu(self.layer1(grad_x))
        grad_x = F.relu(self.layer2(grad_x))
        return grad_x


class ConvBlock(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out


class U_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.left_conv_1 = ConvBlock(in_channels=3, middle_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)

        # 定义右半部分网络
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2,
                                           output_padding=1)
        self.right_conv_2 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256)

        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2,
                                           output_padding=1)
        self.right_conv_3 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128)

        self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1,
                                           padding=1)
        self.right_conv_4 = ConvBlock(in_channels=128, middle_channels=64, out_channels=64)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 进行编码
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(feature_3_pool)
        feature_4_pool = self.pool_4(feature_4)

        feature_5 = self.left_conv_5(feature_4_pool)

        # 进行解码
        de_feature_1 = self.deconv_1(feature_5)
        # 特征拼接
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)

        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        out = self.right_conv_5(de_feature_4_conv)

        return out



# class BN_Conv2d(nn.Module):
#     """
#     BN_CONV, default activation is ReLU
#     """
#
#     def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
#                  dilation=1, groups=1, bias=False, activation=True) -> object:
#         super(BN_Conv2d, self).__init__()
#         layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                             padding=padding, dilation=dilation, groups=groups, bias=bias),
#                   nn.BatchNorm2d(out_channels)]
#         if activation:
#             layers.append(nn.ReLU(inplace=True))
#         self.seq = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.seq(x)
#
# class BasicBlock(nn.Module):
#     """
#     basic building block for ResNet-18, ResNet-34
#     """
#     message = "basic"
#
#     def __init__(self, in_channels, out_channels, strides, is_se=False):
#         super(BasicBlock, self).__init__()
#         self.is_se = is_se
#         self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
#         self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
#         if self.is_se:
#             self.se = SE(out_channels, 16)
#
#         # fit input with residual output
#         self.short_cut = nn.Sequential()
#         if strides is not 1:
#             self.short_cut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.is_se:
#             coefficient = self.se(out)
#             out = out * coefficient
#         out = out + self.short_cut(x)
#         return F.relu(out)
#
# class BottleNeck(nn.Module):
#     """
#     BottleNeck block for RestNet-50, ResNet-101, ResNet-152
#     """
#     message = "bottleneck"
#
#     def __init__(self, in_channels, out_channels, strides, is_se=False):
#         super(BottleNeck, self).__init__()
#         self.is_se = is_se
#         self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)  # same padding
#         self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
#         self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=False)
#         if self.is_se:
#             self.se = SE(out_channels * 4, 16)
#
#         # fit input with residual output
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels * 4)
#         )
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         if self.is_se:
#             coefficient = self.se(out)
#             out = out * coefficient
#         out = out + self.shortcut(x)
#         return F.relu(out)
#
#
# class ResNet(nn.Module):
#     """
#     building ResNet_34
#     """
#
#     def __init__(self, block: object, groups: object, num_classes=1000) -> object:
#         super(ResNet, self).__init__()
#         self.channels = 64  # out channels from the first convolutional layer
#         self.block = block
#
#         self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3, bias=False)
#         self.bn = nn.BatchNorm2d(self.channels)
#         self.pool1 = nn.MaxPool2d(3, 2, 1)
#         self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1, index=2)
#         self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
#         self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
#         self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)
#         self.pool2 = nn.AvgPool2d(7)
#         patches = 512 if self.block.message == "basic" else 512 * 4
#         self.fc = nn.Linear(patches, num_classes)  # for 224 * 224 input size
#
#     def _make_conv_x(self, channels, blocks, strides, index):
#         """
#         making convolutional group
#         :param channels: output channels of the conv-group
#         :param blocks: number of blocks in the conv-group
#         :param strides: strides
#         :return: conv-group
#         """
#         list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
#         conv_x = nn.Sequential()
#         for i in range(len(list_strides)):
#             layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
#             conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
#             self.channels = channels if self.block.message == "basic" else channels * 4
#         return conv_x
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = F.relu(self.bn(out))
#         out = self.pool1(out)
#         out = self.conv2_x(out)
#         out = self.conv3_x(out)
#         out = self.conv4_x(out)
#         out = self.conv5_x(out)
#         out = self.pool2(out)
#         out = out.view(out.size(0), -1)
#         out = F.softmax(self.fc(out))
#         return out
#
# def ResNet_18(num_classes=1000):
#     return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)
#
# def ResNet_34(num_classes=1000):
#     return ResNet(block=BasicBlock, groups=[3, 4, 6, 3], num_classes=num_classes)
#
# def ResNet_50(num_classes=1000):
#     return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes)
#
# def ResNet_101(num_classes=1000):
#     return ResNet(block=BottleNeck, groups=[3, 4, 23, 3], num_classes=num_classes)
#
# def ResNet_152(num_classes=1000):
#     return ResNet(block=BottleNeck, groups=[3, 8, 36, 3], num_classes=num_classes)