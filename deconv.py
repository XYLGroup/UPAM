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



def img_deconv(img_tensor, K_tensor,  lambdas_, BSR_param, HR, kernel_GT):



    U_M = img_tensor.shape[2]
    U_N = img_tensor.shape[3]
    U_C = img_tensor.shape[1]

    K_M = K_tensor.shape[0]
    K_N = K_tensor.shape[1]

    K_GT_M = kernel_GT.shape[0]
    K_GT_N = kernel_GT.shape[1]

    HR_M = HR.shape[0]
    HR_N = HR.shape[1]


    U_pad = BSR_param.X_pad_tensor
    K = K_tensor

    gradUdata = torch.zeros_like(U_pad)



    for i in range(BSR_param.iterations):
        # update sharp image Y
        for c in range(U_C):
            a = torch.Tensor(convolve2d(U_pad[0][c], K, mode='valid', boundary='fill', fillvalue=0)) - img_tensor[0][c]
            gradUdata[0][c] = torch.Tensor(convolve2d(a, torch.rot90(K,2), mode='full', boundary='fill', fillvalue=0))

        gradU = (gradUdata - lambdas_ * gradTVcc(U_pad))

        sf = (5e-3 * torch.max(U_pad))/max(1e-31, torch.max(abs(gradU)))

        U_pad = U_pad - sf * gradU


        if BSR_param.visual_flag == True and (i + 1)%100==0:

            U_unpad = F.pad(U_pad, mode='replicate', pad=(-(K_M // 2), -(K_M // 2), -(K_N // 2), -(K_N // 2)))
            SR = Tensor_To_Image(U_unpad)
            SR[SR > 1] = 1
            SR[SR < 0] = 0
            mpimg.imsave(BSR_param.resultsPath + 'SR/' + 'SR_image_sf_{}iters_{}.png'.format(BSR_param.scale, i + 1),SR)
            #mpimg.imsave(BSR_param.resultsPath + 'K/' + 'kernel_%s.png'%(i + 1), K)
            # plt.figure()
            # plt.subplot(2, 2, 1)
            # plt.imshow(HR)
            # plt.axis('off')
            #
            # plt.subplot(2, 2, 2)
            # plt.imshow(SR)
            # plt.axis('off')

            SR_tensor = Image_To_Tensor(SR)

            SR_tensor_interpolate = F.interpolate(SR_tensor, size=[HR_M, HR_N], mode='bicubic', align_corners=False)

            SR_tensor_interpolate = Tensor_To_Image(SR_tensor_interpolate)

            evaluation_PSNR_SSIM_Image(SR_tensor_interpolate, HR, BSR_param.GT_name, BSR_param.scale, i + 1)


    return SR_tensor
