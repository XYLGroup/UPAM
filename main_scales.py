
import matplotlib.image as mpimg
from utils import *
import numpy as np
import torch
import torch.nn.functional as F


# lambda_ = 3.5e-4


def Generate_scale(img, MK, NK , ctf_params):

    # Numscale = 15

    img_M = img.shape[0]
    img_N = img.shape[1]
    img_C = img.shape[2]

    img_tensor = Image_To_Tensor(img)

    smallestScale = 3

    scales = 0

    imgs = []
    Ms = []
    Ns = []
    MKs = []
    NKs = []
    lambdas = []

    imgs.append(img_tensor)
    Ms.append(img_M)
    Ns.append(img_N)
    MKs.append(MK)
    NKs.append(NK)

    lambdas.append(ctf_params.finalLambda)

    while (MKs[scales] > smallestScale and NKs[scales] > smallestScale and lambdas[scales] * ctf_params.lambdaMultiplier < ctf_params.maxLambda):

        scales = scales + 1
        imgs.append(1)
        Ms.append(1)
        Ns.append(1)
        MKs.append(1)
        NKs.append(1)
        lambdas.append(1)
        lambdas[scales] = lambdas[scales - 1] * ctf_params.lambdaMultiplier

        MKs[scales] = round(MKs[scales - 1] / ctf_params.kernelSizeMultiplier)
        NKs[scales] = round(NKs[scales - 1] / ctf_params.kernelSizeMultiplier)

        if ((MKs[scales] % 2) == 0):
            MKs[scales] = MKs[scales] - 1

        if ((NKs[scales] % 2) == 0):
            NKs[scales] = NKs[scales] - 1

        if MKs[scales] == MKs[scales - 1]:
            MKs[scales] = MKs[scales] - 2

        if NKs[scales] == NKs[scales - 1]:
            NKs[scales] = NKs[scales] - 2

        if MKs[scales] < smallestScale:
            MKs[scales] = smallestScale

        if NKs[scales] < smallestScale:
            NKs[scales] = smallestScale

        factorM = MKs[scales - 1]/MKs[scales]
        factorN = NKs[scales - 1]/NKs[scales]


        Ms[scales] = round(Ms[scales - 1] / factorM)
        Ns[scales] = round(Ns[scales - 1] / factorN)


        if ((Ms[scales] % 2) == 0):
            Ms[scales] = Ms[scales] - 1

        if ((Ns[scales] % 2) == 0):
            Ns[scales] = Ns[scales] - 1


        imgs[scales] = F.interpolate(img_tensor, size=[Ms[scales],Ns[scales]], mode='bicubic', align_corners=None)  # scale_factor=1

    return imgs, Ms, Ns, MKs, NKs, lambdas, scales






# Abspath = 'C:/Users/74503/Desktop/PAMcode/Learning-aided_PAM/'
# datapath = 'data/Set5/GTmod12/'
# kernelpath = 'data/Kernels/'
# #filename = 'butterfly.png'
# GT_name = 'butterfly.png'
#
# HR = mpimg.imread(Abspath + datapath + GT_name)
#
# MK = 49
# NK = 49
#

#
#
# [HRs, Ms, Ns, MKs, NKs, lambdas, scales] = Generate_scale(HR, MK, NK , ctf_params)

