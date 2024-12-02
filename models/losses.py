import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .encoder import ProjectionHead
import os


def ProtoLoss(z1, z2, tao, proj_head_pro,proj_head_aug):

    G = z1.shape[0]
    B = z1.shape[1]
    T= z1.shape[2]
    C = z1.shape[3]    #  G x B  x T  x C

    z1 = torch.reshape(z1, (z1.shape[0]*z1.shape[1]* z1.shape[2], z1.shape[3]))   #  G*B*T  x C
    z2 = torch.reshape(z2, (z2.shape[0]*z2.shape[1]*z2.shape[2], z2.shape[3]))

    z1_pro = proj_head_pro(z1)   #  G*B*T  x D
    z2_pro = proj_head_pro(z2) 

    D = z1_pro.shape[1]

    z1_pro = torch.reshape(z1_pro, (G, B, T, z1_pro.shape[1]))  #  G x B x T x D
    z2_pro = torch.reshape(z2_pro, (G, B, T, z2_pro.shape[1]))  #  G x B x T x D

    centroids1 = z1_pro.mean(dim=0)  #  B  x T  x C
    centroids2 = z2_pro.mean(dim=0)  
    loss_inter = contrastive_loss(centroids1, centroids2)



    z1 = proj_head_pro(z1)   #  G*B*T  x D
    z2 = proj_head_pro(z2) 

    z1 = torch.reshape(z1, (G,B,T,z1.shape[1]))  #  G x B x T x D
    z2 = torch.reshape(z2, (G,B,T,z2.shape[1]))  #  G x B x T x D


    z1 = z1.permute(0, 1, 3, 2)    #  G x B x C x T
    z2 = z2.permute(0, 1, 3, 2)

    z1 = torch.reshape(z1, (z1.shape[0]*z1.shape[1]* z1.shape[2], z1.shape[3]))   #  G*B x C x T
    z2 = torch.reshape(z2, (z2.shape[0]*z2.shape[1]*z2.shape[2], z2.shape[3]))   

    z1 = F.max_pool1d(   
                z1,
                kernel_size = z1.size(1), 
            )   #  G*B x C x 1
    z2 = F.max_pool1d(   
                z2,
                kernel_size = z2.size(1),
            )
    
    z1 = torch.reshape(z1, (G, B, D))  #  G x B x C 
    z2 = torch.reshape(z2, (G, B, D))   
    
    weight = torch.eye(G, device=z1.device)  

    mat0 = torch.tril(weight, diagonal=-1)[:, :-1]   
    mat0 += torch.triu(weight, diagonal=1)[:, 1:]
    labels_L = torch.cat([mat0, weight], dim=1)
    labels_R = torch.cat([weight, mat0], dim=1)

    if G == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2G x B x C
    z = z.transpose(0, 1)  # B x 2G x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2G x 2G

    sim = sim / tao.unsqueeze(0).expand(B, -1, -1)

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2G x (2G-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(G, device=z1.device)

    # loss = (logits[:, i, G + i - 1].mean() + logits[:, G + i, i].mean()) / 2

    loss_intra = torch.sum(logits[:, i] * labels_L)
    loss_intra += torch.sum(logits[:, G + i] * labels_R)
    loss_intra /= (4 * G * B)

    del logits, labels_L, labels_R, sim, weight, z, centroids1, centroids2, z1_pro, z2_pro

    return loss_inter, loss_intra



def contrastive_loss(z1, z2):         # B x T x C
    loss1 = torch.tensor(0., device=z1.device)
    d1 = 0

    while z1.size(1) > 1:
        loss1 +=  instance_contrastive_loss(z1, z2)               # B x T x C
        d1 += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)    # B x T x C
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    
    if z1.size(1) == 1:
        loss1 +=  instance_contrastive_loss(z1, z2)
        d1 += 1

    return loss1 / d1

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)

    if B == 1:
        return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    del z, sim, logits, i
    return loss
    
def mixup(a, b, alpha=0.1):
    """
    根据给定的公式实现 MixUp
    参数:
    - a: Tensor,形状为 (B, D)
    - b: Tensor,形状为 (B, D)
    - alpha: Beta 分布的参数
    
    返回:
    - Tensor,形状为 (B, D)
    """
    # 计算 θ = cos^(-1)(a · b)，其中 · 表示点积
    cos_theta = torch.sum(a * b, dim=1) / (torch.norm(a, dim=1) * torch.norm(b, dim=1))
    theta = torch.acos(cos_theta)

    # 从 Beta 分布中采样 λ
    lambda_val = torch.distributions.Beta(alpha, alpha).sample([a.size(0)]).to(a.device)  # 形状为 (B,)

    # 计算 MixUp 的权重
    sin_theta = torch.sin(theta).view(-1, 1)  
    sin_lambda_theta = torch.sin(lambda_val * theta).view(-1, 1) 
    sin_1_minus_lambda_theta = torch.sin((1 - lambda_val) * theta).view(-1, 1)  

    # 计算混合后的向量
    mixed = (a * sin_lambda_theta + b * sin_1_minus_lambda_theta) / sin_theta

    return mixed
