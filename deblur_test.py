import numpy
import torch

from model import *
from utils import *
from SSIM import *
from Generate_kernel import *
from torch.autograd import Variable
import os
import cv2
import torch.nn.functional as F
from scipy.signal import convolve2d
from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime


def img_deblur(img_tensor, M_k, N_k, lambdas_, BSR_param, HR, kernel_GT, Tensor):
    #    for i in range(BSR_param.iterations):
    #     U_M = img_tensor.shape[2]
    #     U_N = img_tensor.shape[3]
    U_C = img_tensor.shape[1]

    # K_M = M_k
    # K_N = N_k

    K_GT_M = kernel_GT.shape[0]
    K_GT_N = kernel_GT.shape[1]

    HR_M = HR.shape[0]
    HR_N = HR.shape[1]

    # (U_C, int(U_N/2), int(U_M/2))
    # Y : low-resolution image(Tensor)
    # U : high-resolution image(Tensor)
    # K : blur kernel(Tensor)

    # divTV = TVLoss()

    # for c in range(U_C):
    # U_init = F.interpolate(img_tensor , size=[U_M, U_N], mode='bicubic', align_corners=False)
    # X_init_pad = F.pad(BSR_param.X, mode='replicate', pad=(K_M // 2,  K_M // 2, K_N // 2, K_N // 2)) # left right up down
    #
    # Y_pad = U_init_pad

    # NU = U_N + K_N - 1
    # MU = U_M + K_M - 1

    U_pad = BSR_param.X_pad_tensor
    K = BSR_param.K_tensor.type(Tensor)
    U_pad_np = U_pad.cpu().numpy()
    K_np = K.cpu().numpy()
    K_np_rot = torch.rot90(K, 2).cpu().numpy()
    gradUdata = (torch.zeros_like(U_pad)).type(Tensor)

    for i in range(BSR_param.iterations):

        # update sharp image Y
        for c in range(U_C):
            a = (torch.Tensor(convolve2d(U_pad_np[0][c], K_np, mode='valid', boundary='fill', fillvalue=0))) - \
                img_tensor[0][c]
            gradUdata[0][c] = torch.Tensor(
                convolve2d(a.cpu().numpy(), K_np_rot, mode='full', boundary='fill', fillvalue=0))
        unknown = gradTVcc(U_pad)
        gradU = (gradUdata - lambdas_ * gradTVcc(U_pad))

        sf = (5e-3 * torch.max(U_pad)) / max(1e-31, torch.max(abs(gradU)))

        U_pad = U_pad - sf * gradU

        # update blur kernel K
        gradk = (torch.zeros_like(K)).type(Tensor)

        for c in range(U_C):
            a = torch.Tensor(convolve2d(U_pad_np[0][c], K_np, mode='valid', boundary='fill', fillvalue=0)) - \
                img_tensor[0][c]
            b = (torch.rot90(U_pad[0][c], 2)).cpu().numpy()
            gradk = gradk + torch.Tensor(
                convolve2d(b, a.cpu().numpy(), mode='valid', boundary='fill', fillvalue=0)).type(Tensor)

        sh = (1e-3 * torch.max(K)) / max(1e-31, torch.max(abs(gradk)))
        K = K - sh * gradk

        # update blur kernel K

        K = torch.max(K, torch.zeros_like(K))
        K = K.div(torch.sum(K))

        if BSR_param.visual_flag == True and (i + 1) % 100 == 0:
            # U_unpad = F.pad(U_pad, mode='replicate', pad=(-(K_M // 2), -(K_M // 2), -(K_N // 2), -(K_N // 2)))
            SR = Tensor_To_Image(U_pad)
            SR[SR > 1] = 1
            SR[SR < 0] = 0

            fv = Tensor_To_Image(img_tensor)
            fv[fv > 1] = 1
            fv[fv < 0] = 0

            # mpimg.imsave(BSR_param.resultsPath + 'SR/' + 'SR_image_sf_{}iters_{}.png'.format(BSR_param.scale , i + 1), SR)
            mpimg.imsave(BSR_param.resultsPath + 'K/' + 'kernel_sf_{}iters_{}.png'.format(BSR_param.scale, i + 1), K)

            # plt.figure()
            # plt.subplot(2, 2, 3)
            # plt.imshow(kernel_GT)
            # plt.axis('off') # 不显示坐标轴
            #
            # plt.subplot(2, 2, 4)
            # plt.imshow(K)
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()

            K_tensor = K.clone().detach()
            K_tensor = K_tensor.unsqueeze(0).unsqueeze(0)

            K_tensor_interpolate = F.interpolate(K_tensor, size=[K_GT_M, K_GT_N], mode='bicubic', align_corners=False)

            K_np_interpolate = K_tensor_interpolate.squeeze(0).squeeze(0)
            K_np_interpolate = K_np_interpolate.numpy()

            evaluation_PSNR_Kernel(K_np_interpolate, kernel_GT, BSR_param.GT_name, BSR_param.scale, i + 1)

    return U_pad, K
