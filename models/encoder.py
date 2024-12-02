import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
import torchvision.models as models

class Mutil_ProjectionHeads(nn.Module):
    def __init__(self, input_dims, num_heads=5, output_dims=128, hidden_dims=256):
        super( Mutil_ProjectionHeads, self).__init__()
        self.projection_heads = nn.ModuleList(
            [ProjectionHead(input_dims, output_dims, hidden_dims) for _ in range(num_heads)]
        )
    
    def forward(self, x, head_index):
        assert 0 <= head_index < len(self.projection_heads), "Invalid head_index"
        return self.projection_heads[head_index](x)
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.repr_dropout(self.proj_head(x))
        return x

# ts encoder
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims 
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
       
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, ts):  # x: B x T x input_dims  T：len
        B = ts.shape[0]
        
        x = ts.transpose(1, 2)   # B x C x T
        x = channel_independence(x)  # B*C x T
        x = torch.unsqueeze(x, -1)  # B*C x T x 1
        
        nan_mask = ~x.isnan() 
        x[~nan_mask] = 0
        x = x.float()

        x = self.input_fc(x)    
        # conv encoder                          # Dilated Convolutions
        x = x.transpose(1, 2)    
        x = self.repr_dropout(self.feature_extractor(x))   
        x = x.transpose(1, 2)   
        
        x = torch.reshape(x, (B, int(x.shape[0]/B), x.shape[1], x.shape[2]))   # B x C x T x C'

        return x

# image encoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        Encoder = models.resnet18(pretrained=True)
        Encoder.fc = nn.Identity()
        self.encoder = Encoder

    def forward(self, x):
        return self.encoder(x)


def channel_independence(x):
    '''
    Reshape the input tensor to have a flattened channel dimension.

    Args:
        x (torch.Tensor): Input tensor of shape bs x c x patch_num x patch_len or bs x c x T

    Returns:
        torch.Tensor: Reshaped tensor of shape bs*c x patch_num x patch_len for 4-dimensional input,
                      or bs*c x T for 3-dimensional input.
    '''

    if x.ndim == 4:
        return torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
    if x.ndim == 3:
        return torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2]))
        