import math
import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import scipy.io as sio
from scipy.interpolate import interp2d
from PIL import Image


def evaluation_PSNR_SSIM_Image(sr, hr, filename, scale, outers, iters ):

    im_psnr = calculate_psnr(hr * 255, sr * 255)
    im_ssim = calculate_ssim(hr * 255, sr * 255)

    print('{}_Outer_{}_iter_{}_sf_{} (1 images), Average Imgae PSNR/SSIM: {:.2f}/{:.4f}'.format(filename, outers, iters, scale, im_psnr, im_ssim))
    return im_psnr, im_ssim


def evaluation_PSNR_Kernel(kernel, kernel_gt, filename, scale, iters ):

    kernel_psnr = calculate_psnr(kernel_gt, kernel, is_kernel=True)
    print('{}_iter_{}_sf_{} (1 images), Average Kernel PSNR: {:.2f}'.format(filename, iters, scale, kernel_psnr))
    return kernel_psnr


def evaluation_Single_Image(sr, kernel, hr, kernel_gt, filename, scale, iters ):

    kernel_psnr = calculate_psnr(kernel_gt, kernel, is_kernel=True)
    # hr = rgb2ycbcr(hr / 255., only_y=True)
    # sr = rgb2ycbcr(sr / 255., only_y=True)
    # crop_border = sf
    # cropped_hr = hr[crop_border:-crop_border, crop_border:-crop_border]
    # cropped_sr = sr[crop_border:-crop_border, crop_border:-crop_border]
    im_psnr = calculate_psnr(hr * 255, sr * 255)
    im_ssim = calculate_ssim(hr * 255, sr * 255)

    print('{}_iter_{}_sf_{} (1 images), Average Imgae PSNR/SSIM: {:.2f}/{:.4f}, Average Kernel PSNR: {:.2f}'.format(filename, iters, scale, im_psnr, im_ssim, kernel_psnr))




def evaluation_dataset(input_dir, conf, used_iter=''):
    ''' Evaluate the model with kernel and image PSNR'''
    print('Calculating PSNR...')
    filesource = os.listdir(os.path.abspath(input_dir))
    filesource.sort()

    im_psnr = 0
    im_ssim = 0
    kernel_psnr = 0
    for filename in filesource:
        # load gt kernel
        if conf.real:
            kernel_gt = np.ones([min(conf.sf * 4 + 3, 21), min(conf.sf * 4 + 3, 21)])
        else:
            path = os.path.join(input_dir, filename).replace('lr_x', 'gt_k_x').replace('.png', '.mat')
            kernel_gt = sio.loadmat(path)['Kernel']

        # load estimated kernel
        path = os.path.join(conf.output_dir_path, filename).replace('.png', '.mat')
        kernel = sio.loadmat(path)['Kernel']

        # calculate psnr
        kernel_psnr += calculate_psnr(kernel_gt, kernel, is_kernel=True)

        # load HR
        path = os.path.join(input_dir.replace(input_dir.split('/')[-1], 'HR'), filename)
        hr = read_image(path)
        hr = modcrop(hr, conf.sf)

        # load SR
        path = os.path.join(conf.output_dir_path, filename)
        sr = read_image(path)

        # calculate psnr
        hr = rgb2ycbcr(hr / 255., only_y=True)
        sr = rgb2ycbcr(sr / 255., only_y=True)
        crop_border = conf.sf
        cropped_hr = hr[crop_border:-crop_border, crop_border:-crop_border]
        cropped_sr = sr[crop_border:-crop_border, crop_border:-crop_border]
        im_psnr += calculate_psnr(cropped_hr * 255, cropped_sr * 255)
        im_ssim += calculate_ssim(cropped_hr * 255, cropped_sr * 255)

        # psnr, ssim = comp_upto_shift(hr * 255, sr*255, maxshift=1, border=conf.sf, min_interval=0.25)
        # im_psnr += psnr
        # im_ssim += ssim


    print('{}_iter{} ({} images), Average Imgae PSNR/SSIM: {:.2f}/{:.4f}, Average Kernel PSNR: {:.2f}'.format(conf.output_dir_path,
                                                                                                  used_iter,
                                                                                                  len(filesource),
                                                                                                  im_psnr / len(
                                                                                                      filesource),
                                                                                                  im_ssim / len(
                                                                                                      filesource),
                                                                                                  kernel_psnr / len(
                                                                                                      filesource)))


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def comp_upto_shift(img1, img2, maxshift=5, border=0, min_interval=0.25):
    '''
   compute sum of square differences between two images, after
   finding the best shift between them. need to account for shift
   because the kernel reconstruction is shift invariant- a small
   shift of the image and kernel will not effect the likelihood score.
   Args:
        I1/img1: estimated image
        I2/img2: reference
        ychannel: use ychannel for evaluation, faster and better
        maxshift: assumed maxshift
        boarder: shave boarder to calculate PSNR and SSIM
    '''

    I1 = img1
    I2 = img2

    I2 = I2[border:-border,border:-border]
    I1 = I1[border-maxshift:-border+maxshift,border-maxshift:-border+maxshift]
    N1, N2 = I2.shape[:2]

    gx, gy = np.arange(-maxshift, N2+maxshift, 1.0), np.arange(-maxshift, N1+maxshift, 1.0)

    shifts = np.linspace(-maxshift, maxshift, int(2*maxshift/min_interval+1))
    gx0, gy0 = np.arange(0, N2, 1.0), np.arange(0, N1, 1.0)

    ssdem=np.zeros([len(shifts),len(shifts)])
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            gxn = gx0+shifts[i]
            gvn = gy0+shifts[j]
            if I1.ndim == 2:
                tI1 = interp2d(gx, gy, I1)(gxn, gvn)
            elif I1.ndim == 3:
                tI1 = np.zeros(I2.shape)
                for k in range(I1.shape[-1]):
                    tI1[:,:,k] = interp2d(gx, gy, I1[:,:,k])(gxn, gvn)
            ssdem[i,j]=np.sum((tI1[border:-border, border:-border]-I2[border:-border, border:-border])**2)

    # util.surf(ssdem)
    idxs = np.unravel_index(np.argmin(ssdem), ssdem.shape)
    # print('shifted pixel is {}x{}'.format(shifts[idxs[0]], shifts[idxs[1]]))

    gxn = gx0+shifts[idxs[0]]
    gvn = gy0+shifts[idxs[1]]
    if I1.ndim == 2:
        tI1 = interp2d(gx, gy, I1)(gxn, gvn)
    elif I1.ndim == 3:
        tI1 = np.zeros(I2.shape)
        for k in range(I1.shape[-1]):
            tI1[:,:,k] = interp2d(gx, gy, I1[:,:,k])(gxn, gvn)

    psnr = calculate_psnr(tI1, I2)
    ssim = calculate_ssim(tI1, I2)

    return psnr, ssim

def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)



def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def calculate_psnr(img1, img2, is_kernel=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse)) if is_kernel else 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()