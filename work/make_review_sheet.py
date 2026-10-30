"""Render one name-only image per identity with the LoRA and tile them into one sheet.

    CUDA_VISIBLE_DEVICES=1 python3 make_review_sheet.py --lora out/lora_step14000.safetensors
"""
import os, sys, json, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_shim  # noqa: F401
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from lora_util import load_into, upcast_vae

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/whu/ultraman/models/sdxl-base-1.0"

ap = argparse.ArgumentParser()
ap.add_argument("--lora", default="out/lora_step14000.safetensors")
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lora-scale", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=26)
ap.add_argument("--cfg", type=float, default=6.0)
ap.add_argument("--outdir", default="data/review")
ap.add_argument("--sheet", default="review_sheet.jpg")
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--tile", type=int, default=248)
ap.add_argument("--ids", default="")
args = ap.parse_args()

meta = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
ids = [i for i in meta if meta[i]["kind"] in ("ultra", "kaiju")]
if args.ids:
    ids = args.ids.split(",")
os.makedirs(os.path.join(ROOT, args.outdir), exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                variant="fp16", use_safetensors=True)
pipe = pipe.to("cuda")
upcast_vae(pipe)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16,
                                          variant="fp16", use_safetensors=True)
lora_path = args.lora if os.path.isabs(args.lora) else os.path.join(ROOT, args.lora)
unet, _, _ = load_into(unet, lora_path, args.rank, args.rank, args.lora_scale)
pipe.unet = unet.to("cuda")
pipe.set_progress_bar_config(disable=True)

def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

rows = (len(ids) + args.cols - 1) // args.cols
W = args.cols * args.tile
H = rows * (args.tile + 22)
sheet = Image.new("RGB", (W, H), (18, 18, 20))
d = ImageDraw.Draw(sheet)
f = font(17)

for n, i in enumerate(ids):
    m = meta[i]
    prompt = "%s, tokusatsu, full body, cinematic lighting, best quality" % m["en"]
    seed = int(hashlib.md5(i.encode()).hexdigest()[:8], 16) % 100000
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
               width=1024, height=1024, generator=g).images[0]
    img.save(os.path.join(ROOT, args.outdir, "%s.png" % i))
    t = img.resize((args.tile, args.tile), Image.LANCZOS)
    x = (n % args.cols) * args.tile
    y = (n // args.cols) * (args.tile + 22)
    sheet.paste(t, (x, y))
    d.text((x + 6, y + args.tile + 2), "%s | %s" % (i, m["zh"]), font=f, fill=(230, 230, 230))
    print("%-10s seed=%-6d -> %s.png" % (i, seed, i), flush=True)

out = os.path.join(ROOT, args.sheet)
sheet.save(out, quality=88)
print("sheet ->", out, sheet.size)
