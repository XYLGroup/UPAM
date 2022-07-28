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
import scipy.io as scio

def img_deblur(img_tensor, M_k, N_k, lambdas_, BSR_param, HR, kernel_GT):

    U_C = img_tensor.shape[1]

    K_GT_M = kernel_GT.shape[0]
    K_GT_N = kernel_GT.shape[1]

    HR_M = HR.shape[0]
    HR_N = HR.shape[1]

    U_pad = BSR_param.X_pad_tensor
    U_pad = U_pad.cpu().detach()
    img_tensor = img_tensor.cpu().detach()
    K = BSR_param.K_tensor.detach()

    gradUdata = torch.zeros_like(U_pad)

    PSNR_kernel = []
    for i in range(BSR_param.iterations):



        # update sharp image Y
        for c in range(U_C):
            a = torch.Tensor(convolve2d(U_pad[0][c], K, mode='valid', boundary='fill', fillvalue=0)) - img_tensor[0][c]
            gradUdata[0][c] = torch.Tensor(convolve2d(a, torch.rot90(K, 2), mode='full', boundary='fill', fillvalue=0))

        gradU = (gradUdata - lambdas_ * gradTVcc(U_pad))

        sf = (5e-3 * torch.max(U_pad)) / max(1e-31, torch.max(abs(gradU)))

        U_pad = U_pad - sf * gradU



        # update blur kernel K
        gradk = torch.zeros_like(K)

        for c in range(U_C):
            a = torch.Tensor(convolve2d(U_pad[0][c], K, mode='valid', boundary='fill', fillvalue=0)) - img_tensor[0][c]
            gradk = gradk + convolve2d((torch.rot90(U_pad[0][c], 2)), a, mode='valid', boundary='fill', fillvalue=0)

        # gradk = (gradk - lambdas_ * gradTVcc(K))

        sh = (1e-3 * torch.max(K)) / max(1e-31, torch.max(abs(gradk)))
        K1 = K - sh * gradk

        K = torch.sign(K1) * torch.max((torch.abs(K1)-sh*BSR_param.mu),torch.zeros_like(K1))


        # update blur kernel K

        K = torch.max(K, torch.zeros_like(K))
        K = K.div(torch.sum(K))

        if BSR_param.visual_flag == True and (i + 1) % BSR_param.Visual_num == 0:
            # U_unpad = F.pad(U_pad, mode='replicate', pad=(-(K_M // 2), -(K_M // 2), -(K_N // 2), -(K_N // 2)))
            SR = Tensor_To_Image(U_pad)
            SR[SR > 1] = 1
            SR[SR < 0] = 0

            fv = Tensor_To_Image(img_tensor)
            fv[fv > 1] = 1
            fv[fv < 0] = 0

            mpimg.imsave(BSR_param.resultsPath + 'K/' + 'kernel_sf_{}_Outer_{}_iters_{}.png'.format(BSR_param.scale, BSR_param.outers + 1, i + 1), K)
            # scio.savemat(BSR_param.resultsPath + 'K/' + 'kernel_sf_{}iters_{}.mat'.format(BSR_param.scale, i + 1), {'Kernel': K})

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(kernel_GT)
            plt.axis('off') # 不显示坐标轴

            plt.subplot(1, 2, 2)
            plt.imshow(K)
            plt.axis('off')  # 不显示坐标轴
            plt.show()

            K_tensor = K.clone().detach()
            K_tensor = K_tensor.unsqueeze(0).unsqueeze(0)

            K_tensor_interpolate = F.interpolate(K_tensor, size=[K_GT_M, K_GT_N], mode='bicubic', align_corners=False)

            K_np_interpolate = K_tensor_interpolate.squeeze(0).squeeze(0)
            K_np_interpolate = K_np_interpolate.numpy()

            PSNR_kernel.append(evaluation_PSNR_Kernel(K_np_interpolate, kernel_GT, BSR_param.GT_name, BSR_param.scale, i + 1))

    return U_pad, K, max(PSNR_kernel)
