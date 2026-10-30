"""Export a trained checkpoint to a compact fp16 LoRA for distribution.

Training saves fp32 (~186 MB per checkpoint). Casting to fp16 halves that, which keeps the
file under GitHub's 100 MB per-file limit.

    python3 export_lora.py --ckpt out/lora_step12000.safetensors --out ultraman-lora-r32.safetensors
"""
import os, argparse
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
args = ap.parse_args()

DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
sd = load_file(args.ckpt)
out = {k: v.to(DT).contiguous() for k, v in sd.items()}
save_file(out, args.out, metadata={"format": "pt", "dtype": args.dtype,
                                   "targets": "to_q,to_k,to_v,to_out.0"})
size = os.path.getsize(args.out) / 1e6
print("wrote %s  %d tensors  %.1f MB  dtype=%s" % (args.out, len(out), size, args.dtype))
