
import random
from torch.nn import init
import math
from torch.nn.modules.module import Module
import numpy as np
import tsaug
import torch
import time
from torch.nn.functional import interpolate

def totensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor).cuda()
    
class Jittering():
    def __init__(self, sigma=0.3) -> None:
        self.sigma = sigma
    def __call__(self,x):
        ret =  x + torch.normal(mean=0., std=self.sigma, size=x.shape).cuda()
        if torch.isnan(ret).any():
            ret = torch.nan_to_num(ret)
        return ret
    
    
class Scaling():
    def __init__(self, sigma=0.5) -> None:
        self.sigma = sigma
    def __call__(self, x):
        factor = torch.normal(mean=1., std=self.sigma, size=(x.shape[0], x.shape[2])).cuda()
        ret = torch.multiply(x, torch.unsqueeze(factor, 1))
        if torch.isnan(ret).any():
            ret = torch.nan_to_num(ret)

        return ret


class WindowSlicing():
    def __init__(self, reduce_ratio=0.5,diff_len=True) -> None:
        self.reduce_ratio = reduce_ratio
        self.diff_len = diff_len
    def __call__(self,x):

        x = torch.transpose(x,2,1)

        target_len = np.ceil(self.reduce_ratio * x.shape[2]).astype(int)
        if target_len >= x.shape[2]:
            return x
        if self.diff_len:
            starts = np.random.randint(low=0, high=x.shape[2] - target_len, size=(x.shape[0])).astype(int)
            ends = (target_len + starts).astype(int)
            croped_x =  torch.stack([x[i, :, starts[i]: ends[i]] for i in range(x.shape[0])], 0)

        else:
            start = np.random.randint(low=0, high=x.shape[2] - target_len)
            end = target_len + start
            croped_x = x[:, :, start:end]

        ret = interpolate(croped_x, x.shape[2], mode='linear', align_corners=False)
        ret = torch.transpose(ret, 2, 1)
        if torch.isnan(ret).any():
            ret = torch.nan_to_num(ret)

        return ret


class TimeWarping():
    def __init__(self, n_speed_change=100, max_speed_ratio=10) -> None:
        self.transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        x_tran =  self.transform.augment(x)
        ret = totensor(x_tran.astype(np.float32))
        if torch.isnan(ret).any():
            ret = torch.nan_to_num(ret)
        return ret


class WindowWarping():
    def __init__(self, window_ratio=0.3, scales=[0.5, 2.]) -> None:
        self.window_ratio = window_ratio
        self.scales = scales

    def __call__(self,x_torch):

        B, T, D = x_torch.size()
        x = torch.transpose(x_torch, 2, 1)
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        warp_scales = np.random.choice(self.scales, B)
        warp_size = np.ceil(self.window_ratio * T).astype(int)
        window_steps = np.arange(warp_size)

        if(1 >= T - warp_size - 1):
            return x_torch

        window_starts = np.random.randint(low=1, high=T-warp_size-1, size=(B)).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        rets = []

        for i  in range(x.shape[0]):
            window_seg = torch.unsqueeze(x[i, :, window_starts[i]: window_ends[i]], 0)
            window_seg_inter = interpolate(window_seg, int(warp_size * warp_scales[i]), mode='linear', align_corners=False)[0]
            start_seg = x[i, :, :window_starts[i]]
            end_seg = x[i, :, window_ends[i]:]
            ret_i = torch.cat([start_seg, window_seg_inter, end_seg], -1)
            ret_i_inter = interpolate(torch.unsqueeze(ret_i, 0), T, mode='linear', align_corners=False)
            rets.append(ret_i_inter)

        ret = torch.cat(rets, 0)
        ret = torch.transpose(ret, 2, 1)
        if torch.isnan(ret).any():
            ret = torch.nan_to_num(ret)

        return ret
    

def aug(x):   #  B  T  C
    all_augs = [Jittering(), Scaling(), WindowSlicing(), TimeWarping(), WindowWarping()]
    
    xs1_list = []
    
    for aug in all_augs:
        xs1_list.append(aug(x).cuda())
 
    return xs1_list

