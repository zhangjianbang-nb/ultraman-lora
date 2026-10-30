"""Helpers to attach / load the peft-format LoRA used by train_lora.py."""
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def attach(unet, rank=32, alpha=32):
    cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                     target_modules=TARGETS, lora_dropout=0.0)
    return get_peft_model(unet, cfg)


def upcast_vae(pipe):
    """Run the VAE in fp32 (fp16 overflows) while accepting fp16 latents.

    diffusers' decode() does not cast the latents, so a plain `.to(float32)` on the VAE
    blows up with "Input type (Half) and bias type (float) should be the same".
    """
    pipe.vae.to(dtype=torch.float32)
    orig = pipe.vae.decode

    def decode(z, *a, **k):
        if z.dtype != torch.float32:
            z = z.to(torch.float32)
        return orig(z, *a, **k)

    pipe.vae.decode = decode
    return pipe


def set_scale(unet, scale):
    """Multiply every LoRA layer's scaling factor (0 = base model, 1 = as trained)."""
    n = 0
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict):
            for k in sc:
                sc[k] = sc[k] * scale
            n += 1
    return n


def load_into(unet, path, rank=32, alpha=32, scale=1.0):
    """Attach LoRA to a plain UNet and load trained weights from `path`."""
    unet = attach(unet, rank, alpha)
    sd = load_file(path)
    # peft saves adapter-name-agnostic keys (".lora_A.weight") but a freshly attached
    # adapter registers them as ".lora_A.default.weight". Without this rename
    # set_peft_model_state_dict silently loads nothing (and the model stays at init).
    sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
            .replace(".lora_B.weight", ".lora_B.default.weight"): v for k, v in sd.items()}
    missing, unexpected = set_peft_model_state_dict(unet, sd)
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        raise RuntimeError("%d LoRA tensors missing from %s, e.g. %s"
                           % (len(lora_missing), path, lora_missing[0]))
    if scale != 1.0:
        set_scale(unet, scale)
    return unet, missing, unexpected
