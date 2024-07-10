### TomeSD for SD3

```python
from patch import apply_patch


pipe = StableDiffusion3Pipeline.from_pretraine(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

pipe = apply_patch(pipe)
```