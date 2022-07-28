import math
import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import scipy.io as sio
from scipy.interpolate import interp2d


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def forw(para_v, x):
    return (para_v[0]*x + para_v[1])*(para_v[0]*x + para_v[1])



def gradTVcc(f, epsilon = 1e-3):  # f (1,3,544,762)

    batch_size = f.size()[0]
    c_f = f.size()[1]
    h_f = f.size()[2]
    w_f = f.size()[3]



    fxforw_1 = (f[:, :, 1:h_f , :] - f[:, :, :h_f - 1, :])
    fxforw_0 = torch.zeros(batch_size,c_f,1,w_f)
    fxforw = torch.cat((fxforw_1,fxforw_0),2)


    fyforw_1 = (f[:, :, : , 1:w_f] - f[:, :, :, :w_f - 1])
    fyforw_0 = torch.zeros(batch_size,c_f,h_f,1)
    fyforw = torch.cat((fyforw_1,fyforw_0),3)


    fxback = torch.cat((fxforw_0, fxforw_1), 2)
    fyback = torch.cat((fyforw_0, fyforw_1), 3)


    fxmixd_1 = (f[:, :, 1:h_f , :w_f - 1] - f[:, :, :h_f - 1, :w_f - 1])
    fxmixd_10 = fxmixd_1[:,:,:,0]
    fxmixd_10.resize_(fxmixd_1.size()[0],fxmixd_1.size()[1],fxmixd_1.size()[2],1)
    fxmixd_11 = torch.cat((fxmixd_10, fxmixd_1),3)
    fxmixd = torch.cat((fxmixd_11, fxforw_0),2)


    fymixd_1 = (f[:, :, :h_f - 1, 1:w_f] - f[:, :, :h_f - 1, :w_f - 1])
    fymixd_10 = fymixd_1[:, :, 0, :]
    fymixd_10.resize_(fymixd_1.size()[0], fymixd_1.size()[1], 1, fymixd_1.size()[3])
    fymixd_11 = torch.cat((fymixd_10, fymixd_1),2)
    fymixd = torch.cat((fymixd_11, fyforw_0),3)



    # divTV = torch.zeros_like(f)
    epsilon_m = epsilon * torch.ones_like(f)

    divTV =   (fxforw.add(fyforw)).div(torch.max(epsilon_m, torch.sqrt(fxforw.mul(fxforw).add(fyforw.mul(fyforw)))))  \
             - fxback.div(torch.max(epsilon_m, torch.sqrt(fxback.mul(fxback).add(fymixd.mul(fymixd))))) \
             - fyback.div(torch.max(epsilon_m, torch.sqrt(fxmixd.mul(fxmixd).add(fyback.mul(fyback)))))


    return divTV


def Image_To_Tensor(img):

    Y = img.transpose(2, 0, 1)
    Y = torch.tensor(Y)
    Y = Y.type(torch.float32)
    Y = Y.unsqueeze(0)

    return Y


def Tensor_To_Image(Y):

    Y = Y.squeeze(0)
    Y = Y.cpu().numpy()
    img = Y.transpose(1, 2, 0)

    return img


class params():
    def __init__(self, iterations, K_M, K_N, learning_rate, Nowtime, Abspath, datapath, kernelpath, GT_name, resultsPath, visual_flag):
        self.iterations = iterations
        self.K_M = K_M
        self.K_N = K_N
        self.learning_rate = learning_rate
        self.Nowtime = Nowtime
        self.Abspath = Abspath
        self.datapath = datapath
        self.kernelpath = kernelpath
        self.GT_name = GT_name
        self.resultsPath = resultsPath
        self.visual_flag = visual_flag


class ctf_params():

    def __init__(self, lambdaMultiplier):
        self.lambdaMultiplier = lambdaMultiplier

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
