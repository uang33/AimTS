import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
import torch.nn.functional as F

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
    return devices if len(devices) > 1 else devices[0]



def adjust_learning_rate(optimizer, epoch, lr):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def convert_coeff(x, eps=1e-6):
    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    phase = torch.atan2(x.imag, x.real + eps)
    return amp, phase


def dis(tensor1, tensor2,method='euclidean'):
    T=tensor1.shape[0]
    if method=='euclidean':   
        distances = torch.norm(tensor1 - tensor2, dim=2)
        distances =distances/T   
        mean_distance = torch.mean(distances)
    
    else:
        return
    
    return mean_distance.item()



def tao(tensor_list1,tensor_list2,method='+-'):
    G= len(tensor_list1)

    distance_matrix = torch.zeros((2*G,2*G), dtype=torch.float32)

    for i in range(G):
        sum_i=0
        sum_iG=0
        for j in range(G):
            distance_matrix[i,j]=dis(tensor_list1[i].transpose(1,2),tensor_list1[j].transpose(1,2))  
            distance_matrix[i+G,j]=dis(tensor_list2[i].transpose(1,2),tensor_list1[j].transpose(1,2))   
            distance_matrix[i,j+G]=dis(tensor_list1[i].transpose(1,2),tensor_list2[j].transpose(1,2))   
            distance_matrix[i+G,j+G]=dis(tensor_list2[i].transpose(1,2),tensor_list2[j].transpose(1,2))   
            sum_i=sum_i+distance_matrix[i,j]+distance_matrix[i,j+G]
            sum_iG=sum_iG+distance_matrix[i+G,j]+distance_matrix[i+G,j+G]
        
        distance_matrix[i,i] =sum_i/(2*G-1)
        distance_matrix[i+G,i+G] = sum_iG/(2*G-1)

  
    for i in range(2):
        for j in range(2):
            sub_matrix = distance_matrix[i*G:(i+1)*G, j*G:(j+1)*G].clone()
            mask = torch.eye(G, dtype=torch.bool)
            sub_matrix[mask] = -float('inf')
            distance_matrix[i*G:(i+1)*G, j*G:(j+1)*G] = sub_matrix

    matrix=F.softmax(distance_matrix, dim=1)+1

    a, b = torch.min(matrix).item(), torch.max(matrix).item()
    c, d = 1, 1.01
    tao_matrix = c + (matrix - a) * (d - c) / (b - a)
  
    return torch.tensor(tao_matrix)
