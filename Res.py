"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2021 0114, last modified in 2021 0114.
"""

import torch
from torch import nn
from utils_SimpleTool import *
import torch.optim as opt
import torch.nn.functional as func
# from  utils.SimpleTool import *
import warnings

warnings.filterwarnings("ignore")


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = func.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return func.relu(y + x)


class Residual_De(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False):
        super(Residual_De, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = func.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return func.relu(y + x)

def net():
    ret_net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    ret_net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    ret_net.add_module("resnet_block2", resnet_block(64, 128, 2))
    ret_net.add_module("resnet_block3", resnet_block(128, 256, 2))
    ret_net.add_module("resnet_block4", resnet_block(256, 512, 2))
    ret_net.add_module("resnet_block5", resnet_block(512, 256, 2, Mode=False))
    # ret_net.add_module("global_ave_pool", GlobalAvgPool2d())
    # ret_net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

    return ret_net


def resnet_block(in_channels, out_channels, num_residuals, first_block=False, last_block=False, Mode=True):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if Mode:
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        else:
            if i == num_residuals - 1 and not last_block:
                blk.append(Residual_De(in_channels, out_channels, use_1x1conv=True))
            else:
                blk.append(Residual_De(out_channels, out_channels))
    return nn.Sequential(*blk)


def test2():
    x = torch.rand((1, 1, 224, 224))
    for name, layer in net().named_children():
        x = layer(x)
        print(name, "output: shape:", x.shape)


if __name__ == '__main__':
    test2()

