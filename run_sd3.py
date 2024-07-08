import os
import argparse
import torch
from src.sd3_pipeline import StableDiffusion3Pipeline

from utils.viz_utils import *
from utils.ops_utils import *

def main(args):

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    output = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        return_dict=True
    )

    image = output['images'][0]
    latents = output['intermediate_latents']

    os.makedirs(args.output_path, exist_ok=True)

    args_dict = vars(args)
    with open(os.path.join(args.output_path, "args.txt"), "w") as f:
        for k, v in args_dict.items():
            f.write(f"{k}: {v}\n")

    for k, v in latents.items():
        if args.spatial_divergence:
            os.makedirs(os.path.join(args.output_path, "heatmap_latents_div"), exist_ok=True)
            # compute difference with the 8 neighboring pixels (=latent features, to be exact)
            divergence_conv = CustomConv2D(v.shape[1]).to("cuda").half()
            v = divergence_conv(v)
            v_l2 = torch.norm(v, p=2, dim=1, keepdim=True)
            visualize_and_save_heatmap(v_l2, k, os.path.join(args.output_path, "heatmap_latents_div"))
        elif args.pca_highpass:
            os.makedirs(os.path.join(args.output_path, "pca_highpass"), exist_ok=True)
            visualize_and_save_pca_highpass(v, k, os.path.join(args.output_path, "pca_highpass"))
        else:
            os.makedirs(os.path.join(args.output_path, "pca_latents"), exist_ok=True)
            visualize_and_save_features_pca(v, k, os.path.join(args.output_path, "pca_latents"))

    image.save(os.path.join(args.output_path, "sample.png"))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--prompt", '-p', type=str, help="Prompt to run")
    parser.add_argument("--negative_prompt", '-np', type=str, default="", help="Negative Prompt to run")
    parser.add_argument("--num_inference_steps", '-n', type=int, default=28, help="Number of inference steps")
    parser.add_argument("--guidance_scale", '-g', type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--height", '-H', type=int, default=1024, help="Height of the output image")
    parser.add_argument("--width", '-W', type=int, default=1024, help="Width of the output image")
    parser.add_argument("--output_path", '-o', type=str, default="outputs", help="Output path")

    # experiments
    parser.add_argument("--spatial_divergence", action="store_true", help="Use spatial divergence")
    parser.add_argument("--pca_highpass", action="store_true", help="Use highpass pca features")
   
    args = parser.parse_args()

    main(args)