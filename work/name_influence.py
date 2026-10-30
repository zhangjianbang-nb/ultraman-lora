"""Does the name actually change the picture? A metric that avoids DINOv2 entirely.

For one identity i, render twice with the same prompt at different seeds -> that gives
the "seed noise" (how much the image moves for no reason). Then render the other
identities at seed A -> that gives the "name effect" (how much the image moves when only
the name changes). If the LoRA binds names, the name effect must beat the seed noise.

    ratio = mean_cross_name_diff / mean_cross_seed_diff
    base model: ratio ~ 1.0   (it ignores the name, so it is all seed noise)
    working LoRA: ratio > 1.5
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_shim  # noqa: F401
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from lora_util import load_into, upcast_vae

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/whu/ultraman/models/sdxl-base-1.0"
ap = argparse.ArgumentParser()
ap.add_argument("--lora", default=None)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lora-scale", type=float, default=1.0)
ap.add_argument("--ids", default="tiga,zero,gomora,zetton,ultraman,gaia,seven,tyrant")
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--cfg", type=float, default=6.0)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--tag", default=None)
args = ap.parse_args()

meta = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
ids = args.ids.split(",")
tag = args.tag or ("lora" if args.lora else "base")
OUT = os.path.join(ROOT, "data", "nameinf", tag)
os.makedirs(OUT, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                variant="fp16", use_safetensors=True).to("cuda")
upcast_vae(pipe)
pipe.set_progress_bar_config(disable=True)
if args.lora:
    path = args.lora if os.path.isabs(args.lora) else os.path.join(ROOT, args.lora)
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16,
                                               variant="fp16", use_safetensors=True)
    unet, _, _ = load_into(unet, path, args.rank, args.rank, args.lora_scale)
    pipe.unet = unet.to("cuda")


def prompt(i):
    return "%s, tokusatsu, full body, cinematic lighting, best quality" % meta[i]["en"]


def gen(p, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return pipe(prompt=p, num_inference_steps=args.steps, guidance_scale=args.cfg,
                width=args.size, height=args.size, generator=g).images[0]


def arr(im, s=256):
    return np.asarray(im.convert("RGB").resize((s, s), Image.BILINEAR), np.float32) / 255.0


SEED_A, SEED_B = 1000, 2000
A = {i: gen(prompt(i), SEED_A) for i in ids}
B = {i: gen(prompt(i), SEED_B) for i in ids}
for i in ids:
    A[i].save(os.path.join(OUT, "%s_s%d.png" % (i, SEED_A)))
    B[i].save(os.path.join(OUT, "%s_s%d.png" % (i, SEED_B)))

seed_d, name_d = [], []
for i in ids:
    seed_d.append(float(np.abs(arr(A[i]) - arr(B[i])).mean()))
    for j in ids:
        if i != j:
            name_d.append(float(np.abs(arr(A[i]) - arr(A[j])).mean()))
print("%s: cross-seed diff %.4f | cross-name diff %.4f | ratio %.2f"
      % (tag, np.mean(seed_d), np.mean(name_d), np.mean(name_d) / max(1e-9, np.mean(seed_d))))
json.dump({"tag": tag, "seed_diff": round(float(np.mean(seed_d)), 4),
           "name_diff": round(float(np.mean(name_d)), 4),
           "ratio": round(float(np.mean(name_d) / max(1e-9, np.mean(seed_d))), 3)},
          open(os.path.join(OUT, "report.json"), "w"), indent=2)
