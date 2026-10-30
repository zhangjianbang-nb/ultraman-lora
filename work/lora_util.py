"""Helpers to attach / load the peft-format LoRA used by train_lora.py."""
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def attach(unet, rank=32, alpha=32):
    cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                     target_modules=TARGETS, lora_dropout=0.0)
    return get_peft_model(unet, cfg)


def load_into(unet, path, rank=32, alpha=32):
    """Attach LoRA to a plain UNet and load trained weights from `path`."""
    unet = attach(unet, rank, alpha)
    sd = load_file(path)
    missing, unexpected = set_peft_model_state_dict(unet, sd)
    return unet, missing, unexpected
