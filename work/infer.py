"""Generate images with SDXL base + the trained Ultraman LoRA.

    CUDA_VISIBLE_DEVICES=1 python3 infer.py --lora out/lora_step12000.safetensors \
        --prompt "生迪迦" --out diga.png

Chinese prompts are rewritten into the English tag space the LoRA was trained on
(see names.py); pass --raw to send the prompt through untouched.

    python3 infer.py --lora ... --grid --outdir grids      # one image per identity
    python3 infer.py --lora ... --prompt 迪迦 --n 4        # four candidates
"""
import os, json, argparse
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import names

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/ultraman/models/sdxl-base-1.0"))
ap.add_argument("--lora", default=None)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--prompt", default=None)
ap.add_argument("--raw", action="store_true", help="do not rewrite Chinese names")
ap.add_argument("--out", default="out.png")
ap.add_argument("--outdir", default="grids")
ap.add_argument("--grid", action="store_true")
ap.add_argument("--ids", default=None, help="comma separated identity ids for grid mode")
ap.add_argument("--n", type=int, default=1, help="images per prompt")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=6.5)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--size", type=int, default=1024)
args = ap.parse_args()

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from lora_util import load_into

pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
pipe.vae.to(dtype=torch.float32)  # SDXL VAE overflows in fp16
if args.lora:
    unet = UNet2DConditionModel.from_pretrained(
        args.model, subfolder="unet", torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True)
    unet, missing, unexpected = load_into(unet, args.lora, args.rank, args.rank)
    pipe.unet = unet.to("cuda")
    print("lora loaded: missing=%d unexpected=%d" % (len(missing), len(unexpected)))


def render(prompt, out_path, seed=None):
    g = torch.Generator(device="cuda").manual_seed(args.seed if seed is None else seed)
    im = pipe(prompt=prompt, num_inference_steps=args.steps,
              guidance_scale=args.cfg, width=args.size, height=args.size,
              generator=g).images[0]
    im.save(out_path)
    return im


if args.grid:
    ids = [x.strip() for x in args.ids.split(",")] if args.ids else list(names.META)
    os.makedirs(args.outdir, exist_ok=True)
    for i in ids:
        prompt = names._caption(i) + ", " + names.SUFFIX
        p = os.path.join(args.outdir, "%s.png" % i)
        render(prompt, p)
        print("  %-12s -> %s" % (i, p))
else:
    if not args.prompt:
        raise SystemExit("need --prompt or --grid")
    prompt = args.prompt if args.raw else names.normalize(args.prompt)
    print("prompt:", prompt)
    base, ext = os.path.splitext(args.out)
    for k in range(args.n):
        out = args.out if args.n == 1 else "%s_%d%s" % (base, k + 1, ext)
        render(prompt, out, seed=args.seed + k)
        print("wrote", out)
