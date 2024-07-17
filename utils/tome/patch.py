import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from . import merge
from .utils import isinstance_str, init_generator
from diffusers.models.attention import _chunked_feed_forward

def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    """
    x: torch.Tensor of shape (2*batch_size, 4096, 1536)
    """
    original_h, original_w = tome_info["size"] # 64, 64 (n_patches_h, n_patches_w)
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, 
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_m, u_a, u_m  # Okay this is probably not very good

def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock

def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeJointTransformerBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            temb=None
        ):
            # print("Hidden States: ", hidden_states.size()) # torch.Size([2, 4096, 1536])
            # print("Encoder Hidden States: ", encoder_hidden_states.size()) # torch.Size([2, 333, 1536])
            # print("Token Embeddings: ", temb.size()) # torch.Size([2, 1536])
            m_a, m_m, u_a, u_m = compute_merge(
                hidden_states,
                self._tome_info,
            )
            
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb
            )
            
            # ToMe m_c
            # print("Before: ", norm_hidden_states.size())
            norm_hidden_states = m_a(norm_hidden_states)
            # print("After: ", norm_hidden_states.size())
            
            # last layer only
            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(
                    encoder_hidden_states,
                    temb,
                )
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb,
                )
            

            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
            )
            

            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + u_a(attn_output)
            
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            
            norm_hidden_states = m_m(norm_hidden_states)
            if self._chunk_size is not None:
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            
            hidden_states = hidden_states + u_m(ff_output)
            
            # applies only to JointTransformer last layer (which is NOT part of transformer_blocks)
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output
                
                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                if self._chunk_size is not None:
                    context_ff_output = _chunked_feed_forward(
                        self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                    )
                else:
                    context_ff_output = self.ff_context(norm_encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            return encoder_hidden_states, hidden_states

    return ToMeJointTransformerBlock

def hook_tome_model(model: torch.nn.Module, patch_size: int = 2):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, input):
        # register weight and height
        # module._tome_info["size"] = (input[0].shape[2], input[0].shape[3])        
        module._tome_info["size"] = (input[0].shape[2] // patch_size, input[0].shape[3] // patch_size)

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))

def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_mlp: bool = False):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    print(f"[*] Applying ToMe to {'Diffusers' if is_diffusers else 'Stable Diffusion'} ...")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # model -> model.transformers
        diffusion_model = model.transformer if hasattr(model, "transformer") else model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_mlp": merge_mlp
        }
    }
    
    # change _tome_info
    hook_tome_model(diffusion_model, patch_size=2) # TODO hard-coded for SD3

    for block in diffusion_model.transformer_blocks:
        # If for some reason this has a different name, create an issue and I'll fix it
        if not isinstance_str(block, "JointTransformerBlock"):
            continue
        make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
        block.__class__ = make_tome_block_fn(block.__class__)
        block._tome_info = diffusion_model._tome_info

        # Something introduced in SD 2.0 (LDM only)
        # if not hasattr(module, "disable_self_attn") and not is_diffusers:
        #     module.disable_self_attn = False

        # # Something needed for older versions of diffusers
        # if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
        #     module.use_ada_layer_norm = False
        #     module.use_ada_layer_norm_zero = False

    print("[*] ToMe Applied!")

    return model

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model
