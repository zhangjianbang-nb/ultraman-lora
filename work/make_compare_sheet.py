"""Reference photo vs base render vs LoRA render, side by side per character.

Row 1 = the most on-model REAL photo of that identity (nearest to its own centroid),
row 2 = base model, row 3 = LoRA. Same seed for both renders, so the only variable
is the LoRA. This is the sheet a human (or a judging agent) can actually grade.

    CUDA_VISIBLE_DEVICES=1 python3 make_compare_sheet.py --lora out_c3/lora_step12000.safetensors
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_shim  # noqa: F401
import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import Dinov2Model
from lora_util import load_into, upcast_vae
from names import NEGATIVE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/whu/ultraman/models/sdxl-base-1.0"
CLEAN = os.path.join(ROOT, "data", "clean3")
ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--scales", default="0.9")
ap.add_argument("--ids", default="tiga,zero,gomora,zetton,ultraman,gaia")
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--cfg", type=float, default=6.0)
ap.add_argument("--tile", type=int, default=252)
ap.add_argument("--outdir", default="data/compare")
ap.add_argument("--sheet", default="compare_sheet.jpg")
args = ap.parse_args()

meta = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
ids = args.ids.split(",")
scales = [float(x) for x in args.scales.split(",")]
os.makedirs(os.path.join(ROOT, args.outdir), exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
dino = Dinov2Model.from_pretrained("facebook/dinov2-base").eval().to("cuda")


def emb(im):
    a = np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), np.float32) / 255.0
    x = torch.from_numpy((a - MEAN) / STD).permute(2, 0, 1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        r = dino(pixel_values=x).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(r, dim=1)[0]


refs = {}
for i in ids:
    d = os.path.join(CLEAN, i)
    files = [f for f in sorted(os.listdir(d)) if f.endswith(".jpg")]
    ims = [Image.open(os.path.join(d, f)).convert("RGB") for f in files]
    E = torch.stack([emb(m) for m in ims])
    c = torch.nn.functional.normalize(E.mean(0, keepdim=True), dim=1)
    refs[i] = ims[int((E @ c.T).squeeze(1).argmax())]
    print("ref %-9s <- %s" % (i, files[int((E @ c.T).squeeze(1).argmax())]), flush=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                variant="fp16", use_safetensors=True).to("cuda")
upcast_vae(pipe)
pipe.set_progress_bar_config(disable=True)
path = args.lora if os.path.isabs(args.lora) else os.path.join(ROOT, args.lora)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16,
                                           variant="fp16", use_safetensors=True)
unet, _, _ = load_into(unet, path, args.rank, args.rank, 1.0)
unet = unet.to("cuda")
pipe.unet = unet


def set_scale(u, s):
    for m in u.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict):
            for k in sc:
                sc[k] = s


def font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


rows = [("reference photo", None), ("base", "name")]
for s in scales:
    rows.append(("lora@%.1f name" % s, "name"))
    rows.append(("lora@%.1f desc" % s, "desc"))
W = len(ids) * args.tile
H = len(rows) * (args.tile + 20)
sheet = Image.new("RGB", (W, H), (16, 16, 18))
d = ImageDraw.Draw(sheet)
f = font(16)

for r, (label, style) in enumerate(rows):
    for c, i in enumerate(ids):
        if style is None:
            img = refs[i]
            tag = "%s_ref" % i
        else:
            set_scale(unet, 0.0 if label == "base" else float(label.split("@")[1].split(" ")[0]))
            if style == "name":
                prompt = "%s, tokusatsu, full body, cinematic lighting, best quality" % meta[i]["en"]
            else:
                prompt = "%s, %s, single character, tokusatsu, full body, cinematic lighting, best quality" % (
                    meta[i]["en"], meta[i]["desc"])
            g = torch.Generator(device="cpu").manual_seed(1234 + c)
            img = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                       num_inference_steps=args.steps, guidance_scale=args.cfg,
                       width=1024, height=1024, generator=g).images[0]
            tag = "%s_%s" % (i, label.replace("@", "").replace(".", "").replace(" ", "_"))
            print("  %-9s %-14s" % (i, label), flush=True)
        img.save(os.path.join(ROOT, args.outdir, tag + ".png"))
        t = img.resize((args.tile, args.tile), Image.LANCZOS)
        x, y = c * args.tile, r * (args.tile + 20)
        sheet.paste(t, (x, y))
        d.text((x + 5, y + args.tile + 1), "%s | %s | %s" % (i, meta[i]["zh"], label),
               font=f, fill=(228, 228, 228))

out = os.path.join(ROOT, args.sheet)
sheet.save(out, quality=90)
print("sheet ->", out, sheet.size)
