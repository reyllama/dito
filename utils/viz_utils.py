import os
import numpy as np
from PIL import Image
from math import sqrt
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms as T
from transformers import AutoFeatureExtractor

from utils.ops_utils import *

def visualize_and_save_features_pca(features, t, save_dir):
    '''
    features: torch.Tensor of shape (B, C, H, W)
    t: int, time step
    save_dir: str, path to save the visualization
    '''
    # B, C, H, W = features.shape
    # features = rearrange(features, 'b c h w -> (b h w) c')
    # features = features.cpu().numpy()
    # pca = PCA(n_components=3)
    # pca_features = pca.fit_transform(features)
    # pca_features = pca_features.reshape(B, -1, 3)  # B x (H * W) x 3
    pca_features = get_feature_pca(features, n_components=3, return_type='np')


    for i in range(B):
        pca_img = pca_features[i]  # (H * W) x 3
        # h = w = int(sqrt(pca_img.shape[0]))
        # assert H == h and W == w, "Input features shape does not match the PCA features shape"
        # pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(save_dir, f"time_{t}.png"))

def visualize_and_save_as_image(features, t, save_dir, H=512, W=512, cmap=None, interpolation='nearest'):

    if isinstance(features, torch.Tensor):
        features = features.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(H/100, W/100), dpi=100)
    plt.imshow(features, cmap=cmap, interpolation=interpolation)
    plt.axis('off')
    plt.title('')
    
    # Save the heatmap as an image
    save_path = os.path.join(save_dir, f"time_{t}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    heatmap_image = Image.open(save_path)
    heatmap_image = heatmap_image.resize((512, 512), Image.LANCZOS)
    heatmap_image.save(save_path)  # Save the resized image

def visualize_and_save_pca_highpass(features, t, save_dir, threshold=0.1):
    '''
    features: torch.Tensor of shape (B, C, H, W)
    t: int, time step
    save_dir: str, path to save the visualization
    '''
    pca_features = get_feature_pca(features, n_components=3, return_type='pt')
    
    r_channel = high_pass_filter(pca_features[:, 0:1, :, :])
    g_channel = high_pass_filter(pca_features[:, 1:2, :, :])
    b_channel = high_pass_filter(pca_features[:, 2:3, :, :])

    # assuming that batch_size = 1
    image_back = torch.cat((r_channel, g_channel, b_channel), dim=1).squeeze().permute(1, 2, 0).numpy()

    visualize_and_save_as_image(image_back, t, save_dir)



# deprecated --
def visualize_and_save_heatmap(features, t, save_dir):
    '''
    heatmap: torch.Tensor of shape (B, 1, H, W)
    t: int, time step
    save_dir: str, path to save the visualization
    '''
    features = features.squeeze().detach().cpu().numpy()
    
    # Create a heatmap using matplotlib
    plt.figure(figsize=(5.12, 5.12), dpi=100)
    plt.imshow(features, cmap='coolwarm', interpolation='nearest')
    plt.axis('off')
    plt.title('')
    
    # Save the heatmap as an image
    save_path = os.path.join(save_dir, f"time_{t}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    heatmap_image = Image.open(save_path)
    heatmap_image = heatmap_image.resize((512, 512), Image.LANCZOS)
    heatmap_image.save(save_path)  # Save the resized image