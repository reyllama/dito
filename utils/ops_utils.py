import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
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

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * 2
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * 2

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * torch.sqrt(torch.tensor(2 * torch.pi))) * \
                      torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.conv = F.conv2d

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups, padding='same')

def create_custom_conv_kernel(kernel, in_channels):
    kernel = kernel.expand(in_channels, 1, 3, 3)
    return kernel

def apply_smoothing(feature_map, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian smoothing to a PyTorch tensor feature map.
    
    Parameters:
        feature_map (torch.Tensor): Tensor of shape (batch, channel, height, width).
        kernel_size (int): Size of the Gaussian kernel. Default is 5.
        sigma (float): Standard deviation of the Gaussian kernel. Default is 1.0.
    
    Returns:
        torch.Tensor: Smoothed feature map.
    """
    _, channels, _, _ = feature_map.shape
    smoothing = GaussianSmoothing(channels, kernel_size, sigma).to(feature_map.device, feature_map.dtype)
    return smoothing(feature_map)

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