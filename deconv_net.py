import numpy
import torch
from model import *
from utils import *
from SSIM import *
from Generate_kernel import *
from torch.autograd import Variable
from scipy.signal import convolve2d
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def img_deconv_net(img_tensor, X_tensor, K_tensor, X_Net, criterion_Loss, optimizer_X, BSR_param, Tensor, HR):
    X_tensor = X_tensor.type(Tensor)
    SIZE_X = X_tensor.shape[3]
    img_tensor = img_tensor.type(Tensor)
    # SIZE_I = img_tensor.shape[3]
    # U_C = img_tensor.shape[1]

    transforms_X = transforms.Resize(256)
    X_tensor_full = transforms_X(X_tensor)

    # K = K_tensor.cpu().numpy()
    # img_tensor_np = img_tensor.cpu().numpy()
    Udata = torch.zeros_like(X_tensor)
    transforms_X_1 = transforms.Resize(SIZE_X)
    # transforms_I = transforms.Resize(SIZE_I)

    for _ in range(800):

        X_Net.train()
        optimizer_X.zero_grad()
        X_1 = X_Net(X_tensor_full)
        # K = K_tensor.cpu().numpy()
        # Udata = torch.zeros_like(X_tensor)
        X_2 = transforms_X_1(X_1.detach())
        for c in range(U_C):
            Udata[0][c] = torch.Tensor(
                convolve2d(X_2[0][c], K_tensor, mode='same', boundary='fill', fillvalue=0)).type(Tensor) # 找下库
        LOSS = criterion_Loss(img_tensor, Udata)
        LOSS.requires_grad_(True)
        LOSS.backward()
        optimizer_X.step()
        X_tensor_full = X_1.detach()

    X_tensor = transforms_X_1(X_tensor)
    print("complete!")

    U_unpad = F.pad(X_tensor, mode='replicate', pad=(-(K_M // 2), -(K_M // 2), -(K_N // 2), -(K_N // 2)))
    SR = Tensor_To_Image(U_unpad)
    SR[SR > 1] = 1
    SR[SR < 0] = 0
    HR_M = HR.shape[0]
    HR_N = HR.shape[1]
    mpimg.imsave(
        BSR_param.resultsPath + 'SR/' + 'SR_image_sf_{}iters_{}.png'.format(BSR_param.scale, BSR_param.scale + 1), SR)

    SR_tensor = Image_To_Tensor(SR)

    SR_tensor_interpolate = F.interpolate(SR_tensor, size=[HR_M, HR_N], mode='bicubic', align_corners=False)

    SR_tensor_interpolate = Tensor_To_Image(SR_tensor_interpolate)

    evaluation_PSNR_SSIM_Image(SR_tensor_interpolate, HR, BSR_param.GT_name, BSR_param.scale, BSR_param.scale + 1)

    X_tensor = X_tensor.cpu().numpy()

    return torch.Tensor(X_tensor)
