import torch
import torch.nn as nn
import torch.nn.functional as F

def create_custom_conv_kernel(kernel, in_channels):
    kernel = kernel.expand(in_channels, 1, 3, 3)
    return kernel

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