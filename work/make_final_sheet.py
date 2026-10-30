"""One acceptance sheet: base model vs LoRA at two strengths, same seeds.

Row 1 base, row 2 LoRA@0.7, row 3 LoRA@1.0, columns = key characters.
    CUDA_VISIBLE_DEVICES=1 python3 make_final_sheet.py --lora out_c2/lora_step5000.safetensors
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
ap.add_argument("--lora", default="out_c2/lora_step5000.safetensors")
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--scales", default="0.7,1.0")
ap.add_argument("--ids", default="tiga,zero,gomora,zetton,ultraman,gaia")
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--cfg", type=float, default=6.0)
ap.add_argument("--tile", type=int, default=238)
ap.add_argument("--outdir", default="data/final")
ap.add_argument("--sheet", default="final_sheet.jpg")
args = ap.parse_args()

meta = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
ids = args.ids.split(",")
scales = [float(x) for x in args.scales.split(",")]
os.makedirs(os.path.join(ROOT, args.outdir), exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                variant="fp16", use_safetensors=True).to("cuda")
upcast_vae(pipe)
pipe.set_progress_bar_config(disable=True)
lora_path = args.lora if os.path.isabs(args.lora) else os.path.join(ROOT, args.lora)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16,
                                            variant="fp16", use_safetensors=True)
unet, _, _ = load_into(unet, lora_path, args.rank, args.rank, 1.0)
unet = unet.to("cuda")
base_unet = unet


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def set_scale(u, s):
    for m in u.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict):
            for k in sc:
                sc[k] = s


rows = ["base"] + ["lora@%.1f" % s for s in scales]
W = len(ids) * args.tile
H = len(rows) * (args.tile + 20)
sheet = Image.new("RGB", (W, H), (16, 16, 18))
d = ImageDraw.Draw(sheet)
f = font(16)

for r, label in enumerate(rows):
    if label == "base":
        pipe.unet = base_unet
        set_scale(base_unet, 0.0)
    else:
        pipe.unet = base_unet
        set_scale(base_unet, float(label.split("@")[1]))
    for c, i in enumerate(ids):
        m = meta[i]
        prompt = "%s, tokusatsu, full body, cinematic lighting, best quality" % m["en"]
        seed = 1234 + c
        g = torch.Generator(device="cpu").manual_seed(seed)
        img = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
                   width=1024, height=1024, generator=g).images[0]
        tag = "%s_%s_s%d" % (i, label.replace("@", "").replace(".", ""), seed)
        img.save(os.path.join(ROOT, args.outdir, tag + ".png"))
        t = img.resize((args.tile, args.tile), Image.LANCZOS)
        x, y = c * args.tile, r * (args.tile + 20)
        sheet.paste(t, (x, y))
        d.text((x + 5, y + args.tile + 1), "%s | %s | %s" % (i, m["zh"], label),
               font=f, fill=(228, 228, 228))
        print("%-9s %-9s -> %s.png" % (i, label, tag), flush=True)

out = os.path.join(ROOT, args.sheet)
sheet.save(out, quality=90)
print("sheet ->", out, sheet.size)
