import os
import sys
import argparse
import torch
import numpy as np
from src.sd3_pipeline import StableDiffusion3Pipeline

from utils.viz_utils import *
from utils.ops_utils import *
from utils.tome.patch import apply_patch

def main(args):

    generator = torch.manual_seed(args.seed)

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    ratio = 0.1

    for ratio in np.arange(0, 0.5, 0.1):

        if args.tome:
            pipe = apply_patch(pipe, ratio=ratio)

        output = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            return_dict=True,
            do_apply_smoothing=args.apply_smoothing,
            generator=generator,
        )

        image = output['images'][0]
        latents = output['intermediate_latents']

        output_path = os.path.join(args.output_path, f"ratio_{int(100*ratio)}")

        os.makedirs(output_path, exist_ok=True)

        image.save(os.path.join(output_path, f"sample.png"))

    # sys.exit(0)

        args_dict = vars(args)
        with open(os.path.join(output_path, "args.txt"), "w") as f:
            for k, v in args_dict.items():
                f.write(f"{k}: {v}\n")

        initial_latent = latents['1000.00']
        if args.spatial_divergence:
            divergence_conv = CustomConv2D(initial_latent.shape[1]).to("cuda").half()
            initial_divergence = divergence_conv(initial_latent)
            initial_divergence /= (torch.norm(initial_divergence, p=2, dim=[2,3], keepdim=True) + 1e-8)
            
        for k, v in latents.items():
            if args.spatial_divergence:
                os.makedirs(os.path.join(output_path, "latent_divergence"), exist_ok=True)
                # compute difference with the 8 neighboring pixels (=latent features, to be exact)
                v_d = divergence_conv(v)
                v_d /= (torch.norm(v_d, p=2, dim=[2,3], keepdim=True) + 1e-8)
                divergence_diff = minmax_normalize(torch.mean(v_d - initial_divergence, dim=1, keepdim=True))
                # make binary mask with threshold value
                # threshold = 0.5
                # divergence_diff = (divergence_diff > threshold).float()
                # v_l2 = torch.norm(v, p=2, dim=1, keepdim=True)
                # visualize_and_save_heatmap(v_l2, k, os.path.join(args.output_path, "heatmap_latents_div"))
                visualize_and_save_as_image(divergence_diff, k, os.path.join(output_path, "heatmap_latents_div"), H=args.height, W=args.width, cmap='cividis')
            
            if args.pca_highpass:
                os.makedirs(os.path.join(output_path, "latent_pca_highpass"), exist_ok=True)
                visualize_and_save_pca_highpass(v, k, os.path.join(output_path, "pca_highpass"), n_components=args.n_components)
            
            if args.pca_latents:
                os.makedirs(os.path.join(output_path, "latent_pca"), exist_ok=True)
                visualize_and_save_features_pca(v, k, os.path.join(output_path, "pca_latents"))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # generation
    parser.add_argument("--prompt", '-p', type=str, help="Prompt to run")
    parser.add_argument("--negative_prompt", '-np', type=str, default="", help="Negative Prompt to run")
    parser.add_argument("--num_inference_steps", '-n', type=int, default=28, help="Number of inference steps")
    parser.add_argument("--guidance_scale", '-g', type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--height", '-H', type=int, default=1024, help="Height of the output image")
    parser.add_argument("--width", '-W', type=int, default=1024, help="Width of the output image")
    parser.add_argument("--output_path", '-o', type=str, default="outputs", help="Output path")
    parser.add_argument("--seed", type=int, default=2149442, help="Random seed for torch")

    # visualization
    parser.add_argument("--spatial_divergence", action="store_true", help="Use spatial divergence")
    parser.add_argument("--pca_highpass", action="store_true", help="Use highpass pca features")
    parser.add_argument("--pca_latents", action="store_true", help="Use pca features")
    parser.add_argument("--n_components", type=int, default=3, help="Number of components for PCA")
    parser.add_argument("--apply_smoothing", action="store_true", help="Apply Gaussian smoothing to the initial latent feature")
   
    # tokenization
    parser.add_argument("--tome", action="store_true", help="Use tome for tokenization")

    # utils
    parser.add_argument("--show_timesteps", action="store_true", help="Show timesteps for sampling")

    args = parser.parse_args()

    main(args)