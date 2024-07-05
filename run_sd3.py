import os
import argparse
import torch
from utils.viz_utils import visualize_and_save_features_pca
from src.sd3_pipeline import StableDiffusion3Pipeline

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

    os.makedirs(os.path.join(args.output_path, "pca_latents"), exist_ok=True)

    args_dict = vars(args)
    with open(os.path.join(args.output_path, "args.txt"), "w") as f:
        for k, v in args_dict.items():
            f.write(f"{k}: {v}\n")

    for k, v in latents.items():
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
    
    args = parser.parse_args()

    main(args)