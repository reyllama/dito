import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms as T
from transformers import AutoFeatureExtractor

class CustomConv2D(nn.Module):
    def __init__(self, in_channels):
        super(CustomConv2D, self).__init__()
        kernel = torch.tensor(
                [[-0.125, -0.125, -0.125],
                [-0.125,  1,  -0.125],
                [-0.125, -0.125, -0.125]], dtype=torch.float32)
        self.in_channels = in_channels
        self.weight = nn.Parameter(create_custom_conv_kernel(kernel, in_channels), requires_grad=False)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.in_channels)

def create_custom_conv_kernel(kernel, in_channels):
    kernel = kernel.expand(in_channels, 1, 3, 3)
    return kernel

def get_feature_pca(features, n_components=3, return_type='pt'):
    '''
    features: torch.Tensor of shape (B, C, H, W)
    '''
    B, C, H, W = features.shape
    features = rearrange(features, 'b c h w -> (b h w) c')
    features = features.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    pca_features = pca_features.reshape(B, -1, 3)  # B x (H * W) x 3
    if return_type == 'np':
        pca_features_np = pca_features.reshape(B, H, W, 3)
        return pca_features_np
    elif return_type == 'pt':
        pca_features_pt = rearrange(torch.tensor(pca_features), 'b (h w) c -> b c h w', h=H, w=W) # TODO : dtype=torch.float32
        return pca_features_pt
    else:
        raise ValueError("return_type must be 'np' or 'pt'")


def high_pass_filter(channel, r=30):
    dft = torch.fft.fft2(channel)
    dft_shifted = torch.fft.fftshift(dft)
    
    _, _, rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.ones((1, rows, cols), dtype=np.float32)
    mask[:, crow - r:crow + r, ccol - r:ccol + r] = 0
    mask_tensor = torch.tensor(mask)
    
    fshift = dft_shifted * mask_tensor
    f_ishift = torch.fft.ifftshift(fshift)
    channel_back = torch.fft.ifft2(f_ishift)
    channel_back = torch.abs(channel_back)
    
    return channel_back