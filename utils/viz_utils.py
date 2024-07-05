import os
import numpy as np
from PIL import Image
from math import sqrt
from einops import rearrange
from sklearn.decomposition import PCA
from torchvision import transforms as T
from transformers import AutoFeatureExtractor

def visualize_and_save_features_pca(features, t, save_dir):
    '''
    features: torch.Tensor of shape (B, C, H, W)
    t: int, time step
    save_dir: str, path to save the visualization
    '''
    B, C, H, W = features.shape
    features = rearrange(features, 'b c h w -> (b h w) c')
    features = features.cpu().numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    pca_features = pca_features.reshape(B, -1, 3)  # B x (H * W) x 3


    for i in range(B):
        pca_img = pca_features[i]  # (H * W) x 3
        h = w = int(sqrt(pca_img.shape[0]))
        assert H == h and W == w, "Input features shape does not match the PCA features shape"
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(save_dir, f"time_{t}.png"))