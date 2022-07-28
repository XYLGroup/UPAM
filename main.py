import numpy
import torch
import os
import cv2
import datetime
import warnings

from model import *
from utils import *
from SSIM import *
from main_scales import *
from deblur import *
from deconv import *
from Generate_kernel import *
import errno
from torch.autograd import Variable
from scipy.io import loadmat

import torch.nn.functional as F
from scipy.signal import convolve2d
from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as scio
from model import U_Net as UNet
import os.path as osp
####### 初始参数设定 #######
warnings.filterwarnings("ignore")
U_iterations = 10
iterations = 400
Visual_num = 10
K_Visual_num = 400
learning_rate = 1e-2
interpolationMethod = 'bicubic'
TV_loss = TVLoss()
TV_rate = 10
mu = 5

####### 路径设置 #######
Abspath = r'E:\U-PAM/'
dataset_path = 'Data_pre/Dataset/Set5/'
HR_datapath = dataset_path + 'HR/'
kernel_size = 'K_19x19_S_1x5'
sf = 'sf_x4'
LR_datapath = dataset_path + 'LR/' + kernel_size + '/' + sf + '/'
Kernel_path = dataset_path + 'LR/' + kernel_size + '/kernel_'+ kernel_size +'.mat'
# kernel_name = 'kernel_K_39x39_S_15x39.mat'
visual_flag = True
U_Net_size = 512
Out_loops = 4

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

sys.stdout = Logger(osp.join('logs', '{}_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d'),kernel_size)))


PSNR_KERNEL_ALL = []
PSNR_IMG_ALL = []
SSIM_IMG_ALL = []
Bicubic_PSNR_ALL = []
Bicubic_SSIM_ALL = []

####### 迭代图像 #######
for filename in os.listdir(Abspath + HR_datapath):

    ####### 图像读取 #######
    HR = mpimg.imread(Abspath + HR_datapath + filename)[:,:,:3] # 高清
    LR = mpimg.imread(Abspath + LR_datapath + filename)[:,:,:3]  # 低清

    # kkk = mpimg.imread(r'C:/Users/xiaji/Desktop/U-PAM/UPAM/results/2022-05-30baby.png-7/K/kernel_sf_0iters_100.png')[:,:,:3]  # 低清

    ####### kernel读取 #######
    kernel_GT = scio.loadmat(Abspath + Kernel_path)['Kernel']
    K_M = kernel_GT.shape[0]
    K_N = kernel_GT.shape[1]

    ###### 参数设定 #######
    Nowtime = datetime.datetime.now().strftime('%Y-%m-%d') + filename
    resultsPath = 'results/' + kernel_size + sf + '/' + Nowtime + kernel_size + '/'

    BSR_param = params(iterations, K_M, K_N, learning_rate, Nowtime, Abspath, LR_datapath, kernelpath, filename, resultsPath,
                       visual_flag)
    BSR_param.iterations = iterations
    BSR_param.Visual_num = K_Visual_num

    Bicubic_SR = cv2.resize(LR, (HR.shape[1], HR.shape[0]), interpolation=cv2.INTER_CUBIC)
    Bicubic_SR[Bicubic_SR < 0] = 0
    Bicubic_SR[Bicubic_SR > 1] = 1
    print('Bicubic_SR:')
    Bicubic_PSNR, Bicubic_SSIM = evaluation_PSNR_SSIM_Image(Bicubic_SR, HR, BSR_param.GT_name, 'bicubic', 0,0)
    Bicubic_PSNR_ALL.append(Bicubic_PSNR)
    Bicubic_SSIM_ALL.append(Bicubic_SSIM)
    print('------------------------------------------------')

    # img = cv2.filter2D(HR, -1, kernel_GT)
    # print('LR_X1:PSNR')
    # evaluation_PSNR_SSIM_Image(img, HR, BSR_param.GT_name, 1, 1)
    # img_x4 = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4), interpolation=cv2.INTER_CUBIC)
    # print('LR_X4:PSNR')
    # img_x4_R = cv2.resize(img_x4, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # evaluation_PSNR_SSIM_Image(img_x4_R, HR, BSR_param.GT_name, 1, 1)

    ####### 初始数据保存 #######
    if os.path.exists(resultsPath):
        mpimg.imsave(resultsPath + 'LR_image.png', LR)
        mpimg.imsave(resultsPath + 'HR_image.png', HR)
        mpimg.imsave(resultsPath + 'Bicubic_image.png', Bicubic_SR)
        mpimg.imsave(resultsPath + 'kernel_GT.png', kernel_GT)
    else:
        os.makedirs(resultsPath)
        os.makedirs(resultsPath + 'SR')
        os.makedirs(resultsPath + 'K')
        mpimg.imsave(resultsPath + 'LR_image.png', LR)
        mpimg.imsave(resultsPath + 'HR_image.png', HR)
        mpimg.imsave(resultsPath + 'Bicubic_image.png', Bicubic_SR)
        mpimg.imsave(resultsPath + 'kernel_GT.png', kernel_GT)

    ####### 迭代参量初始化 #######
    img_tensor = Image_To_Tensor(LR)
    img_tensor_GT = img_tensor
    X_tensor = F.pad(img_tensor, mode='replicate', pad=(K_M // 2, K_M // 2, K_N // 2, K_N // 2))  # left right up down
    K_tensor = torch.ones(K_M, K_N) / (K_M * K_N)


    ####### 九转参量初始化 #######
    ctf_params.lambdaMultiplier = 1.9
    ctf_params.maxLambda = 1.1e-1
    ctf_params.finalLambda = 3.5e-4
    ctf_params.kernelSizeMultiplier = 1.1
    ctf_params.interpolationMethod = 'bicubic'

    ####### 九转 #######
    [img_tensors, Ms, Ns, MKs, NKs, lambdas, scales] = Generate_scale(LR, K_M, K_N, ctf_params)
    M_hr = Ms[scales]
    N_hr = Ns[scales]
    M_k = MKs[scales]
    N_k = NKs[scales]

    ####### Net初始化/cuda初始 #######
    # model = U_Net()
    model = UNet()
    # model = NestedUNet()

    optimizer_U = torch.optim.Adam(model.parameters(), lr=1e-4)
    MSEloss = torch.nn.MSELoss()
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        model = model.cuda()
        MSEloss.cuda()

    ####### 设定PSNR与SSIM #######
    PSNR_img = []
    SSIM_img = []
    PSNR_kernel = []


    ####### UPAM迭代 #######
    for scale in range(scales + 1):
        scale = scales - scale

        ####### 图像格式数据读取 #######
        M_X = Ms[scale]
        N_X = Ns[scale]
        M_k = MKs[scale]
        N_k = NKs[scale]
        img_tensor = img_tensors[scale]
        lambdas_ = lambdas[scale]

        ####### 图像、kernel大小调整 #######

        if scale == 8:
            X_tensor = F.pad(img_tensor, mode='replicate',
                             pad=(K_M // 2, K_M // 2, K_N // 2, K_N // 2))  # left right up down

            X_tensor = F.interpolate(X_tensor, size=[M_X + M_k - 1, N_X + N_k - 1], mode=interpolationMethod,
                                     align_corners=False)
        else:
            X_tensor = F.interpolate(X_tensor_Xupdate, size=[M_X + M_k - 1, N_X + N_k - 1], mode=interpolationMethod,
                                 align_corners=False)

        K_tensor = K_tensor.unsqueeze(0).unsqueeze(0)
        K_tensor = F.interpolate(K_tensor, size=[M_k, N_k], mode=interpolationMethod, align_corners=False)

        K_tensor = K_tensor.squeeze(0).squeeze(0)

        ####### kernel投影 #######
        K_np = K_tensor.numpy()
        K_np[K_np < 0] = 0
        K_np = K_np / K_np.sum()

        ####### 迭代中间参数设定 #######
        # BSR_param.X_pad_tensor = X_tensor
        BSR_param.K_tensor = torch.tensor(K_np)
        BSR_param.scale = scale
        BSR_param.mu = mu

        for j in range(Out_loops):
            ####### kernel迭代 #######

            X_tensor = F.interpolate(X_tensor, size=[M_X + M_k - 1, N_X + N_k - 1], mode=interpolationMethod,
                                     align_corners=False)

            BSR_param.X_pad_tensor = X_tensor
            BSR_param.outers = j

            [X_pad_tensor, K_tensor, PSNR_k] = img_deblur(img_tensor, M_k, N_k, lambdas_, BSR_param, HR, kernel_GT)

            K_flag = 1 / (M_k * N_k)

            # K_tensor[K_tensor<K_flag] = 0

            K_tensor = K_tensor / K_tensor.sum()

            PSNR_kernel.append(PSNR_k)



            ####### 图像迭代 #######
            for i in range(U_iterations):
                img_tensor_Xupdate = F.interpolate(img_tensor_GT.type(Tensor), size=[U_Net_size, U_Net_size], mode=interpolationMethod, align_corners=False).type(Tensor)
                optimizer_U.zero_grad()

                X_tensor_Xupdate = model(img_tensor_Xupdate)
                # X_tensor_Xupdate = model(X_tensor.type(Tensor))

                Conv = torch.nn.Conv2d(1, 1, [M_k, N_k], stride=1, padding=(M_k // 2, N_k // 2), dilation=1, groups=1,
                                       bias=False,
                                       padding_mode='zeros').type(Tensor)
                Conv.weight.data[0][0] = K_tensor.type(Tensor)
                loss = 0
                for c in range(3):
                    a = X_tensor_Xupdate[0][c]
                    b = F.interpolate(Conv(a.unsqueeze(0).unsqueeze(0)), size=[img_tensor_GT.shape[2], img_tensor_GT.shape[3]], mode=interpolationMethod,
                                     align_corners=False).type(Tensor)
                    loss = loss + MSEloss(b, img_tensor_GT[0][c].unsqueeze(0).unsqueeze(0).type(Tensor))

                T_loss = TV_rate * TV_loss(X_tensor_Xupdate)
                loss = loss + T_loss

                loss.backward()
                optimizer_U.step()

                ####### 图像输出 #######
                if BSR_param.visual_flag == True and (i + 1) % Visual_num == 0:
                    SR = Tensor_To_Image(X_tensor_Xupdate.detach())
                    SR[SR > 1] = 1
                    SR[SR < 0] = 0
                    SR_tensor = Image_To_Tensor(SR)

                    SR_tensor_interpolate = F.interpolate(SR_tensor, size=[HR.shape[0], HR.shape[1]], mode='bicubic',
                                                          align_corners=False)

                    SR_tensor_interpolate = Tensor_To_Image(SR_tensor_interpolate)

                    PSNR_i, SSIM_i = evaluation_PSNR_SSIM_Image(SR_tensor_interpolate, HR, BSR_param.GT_name, BSR_param.scale, j + 1 , i + 1)
                    PSNR_img.append(PSNR_i)
                    SSIM_img.append(SSIM_i)
                    SR_tensor_interpolate[SR_tensor_interpolate > 1] = 1
                    SR_tensor_interpolate[SR_tensor_interpolate < 0] = 0
                    mpimg.imsave(BSR_param.resultsPath + 'SR/' + 'SR_image_sf_{}_Outer_{}_iters_{}.png'.format(BSR_param.scale, j + 1 ,i + 1),
                                 SR_tensor_interpolate)

                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(HR)
                    plt.axis('off')  # 不显示坐标轴

                    plt.subplot(1, 2, 2)
                    plt.imshow(SR_tensor_interpolate)
                    plt.axis('off')  # 不显示坐标轴
                    plt.show()

                X_tensor = X_tensor_Xupdate


    print("Average Kernel PSNR:", max(PSNR_kernel))
    print("Average Image PSNR:", max(PSNR_img))
    print("Average Image SSIM:", max(SSIM_img))

    PSNR_KERNEL_ALL.append(max(PSNR_kernel))
    PSNR_IMG_ALL.append(max(PSNR_img))
    SSIM_IMG_ALL.append(max(SSIM_img))


print("Complete ALL:" , dataset_path)
print("Average Kernel PSNR:", np.mean(np.array(PSNR_KERNEL_ALL)))
print("Average Image PSNR:", np.mean(np.array(PSNR_IMG_ALL)))
print("Average Image SSIM:", np.mean(np.array(SSIM_IMG_ALL)))

print("Average Bicubic PSNR:",np.mean(np.array(Bicubic_PSNR_ALL)))
print("Average Bicubic SSIM:",np.mean(np.array(Bicubic_SSIM_ALL)))