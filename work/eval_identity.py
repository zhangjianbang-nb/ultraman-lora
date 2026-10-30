"""Identity-binding evaluation: generate images per identity, then check whether a DINOv2
nearest-centroid classifier assigns each generated image to the CORRECT identity.

Reference centroids come from the cleaned training images (data/clean/<id>).
Reported metric = top-1 identity accuracy, i.e. "生迪迦出来的是迪迦，不是别的奥特曼".

    python3 eval_identity.py --lora ~/ultraman/out/lora_step3000.safetensors --n 2
    python3 eval_identity.py --n 2 --tag base          # baseline, no LoRA

Writes data/eval/<tag>/<id>_<k>.png and data/eval/<tag>/report.json
"""
import os, json, argparse, sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_shim  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/ultraman/models/sdxl-base-1.0"))
ap.add_argument("--lora", default=None)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--n", type=int, default=2, help="images per identity")
ap.add_argument("--tag", default=None)
ap.add_argument("--ids", default=None)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=7.0)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--style", default="full", choices=["full", "zh", "zh_only"])
args = ap.parse_args()

meta = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
CLEAN = os.path.join(ROOT, "data", "clean")
ids = [i for i in meta if os.path.isdir(os.path.join(CLEAN, i))]
if args.ids:
    ids = [x.strip() for x in args.ids.split(",")]
tag = args.tag or ("lora" if args.lora else "base")
OUT = os.path.join(ROOT, "data", "eval", tag)
os.makedirs(OUT, exist_ok=True)

# ---------- reference centroids ----------
from transformers import Dinov2Model
dino = Dinov2Model.from_pretrained("facebook/dinov2-base").eval()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
dino.to(DEV)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _prep(im, size=224):
    im = im.convert("RGB").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a).permute(2, 0, 1)


def embed(images, bs=16):
    out = []
    for i in range(0, len(images), bs):
        x = torch.stack([_prep(im) for im in images[i:i + bs]]).to(DEV)
        with torch.no_grad():
            r = dino(pixel_values=x).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(r, dim=1).cpu())
    return torch.cat(out)


cent = {}
for i in ids:
    paths = sorted(os.path.join(CLEAN, i, f) for f in os.listdir(os.path.join(CLEAN, i))
                   if f.endswith(".jpg"))
    ims = [Image.open(p) for p in paths]
    e = embed(ims)
    cent[i] = torch.nn.functional.normalize(e.mean(0, keepdim=True), dim=1)
cent_mat = torch.cat([cent[i] for i in ids])
print("reference centroids built for %d identities" % len(ids))

# ---------- generate ----------
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
# SDXL's VAE overflows in fp16; decode in fp32 so no image comes out black/NaN
pipe.vae.to(dtype=torch.float32)
if args.lora:
    from diffusers import UNet2DConditionModel
    from lora_util import load_into
    unet = UNet2DConditionModel.from_pretrained(
        args.model, subfolder="unet", torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True)
    unet, missing, unexpected = load_into(unet, args.lora, args.rank, args.rank)
    pipe.unet = unet.to("cuda")
    print("lora loaded (missing %d / unexpected %d)" % (len(missing), len(unexpected)))


def prompt_for(i):
    m = meta[i]
    if args.style == "zh":
        return "%s, %s, %s" % (m["en"], m["zh"], m["desc"])
    if args.style == "zh_only":
        return m["zh"]
    # "full" == exactly what names.normalize() produces at inference time
    return "%s, %s, full body, tokusatsu, cinematic lighting, best quality" % (
        m["en"], m["desc"])


rows = []
correct = 0
total = 0
per_id = {}
for idx, i in enumerate(ids):
    prompt = prompt_for(i)
    gens = []
    for k in range(args.n):
        g = torch.Generator(device="cuda").manual_seed(args.seed + idx * 17 + k)
        im = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
                  width=args.size, height=args.size, generator=g).images[0]
        p = os.path.join(OUT, "%s_%d.png" % (i, k))
        im.save(p)
        gens.append(im)
    e = embed(gens)
    sims = e @ cent_mat.T
    best = sims.argmax(dim=1).tolist()
    bestsim = sims.max(dim=1).values.tolist()
    hits = [ids[b] == i for b in best]
    per_id[i] = {"prompt": prompt, "pred": [ids[b] for b in best],
                 "sim": [round(s, 3) for s in bestsim],
                 "hit": sum(hits), "n": len(hits)}
    correct += sum(hits)
    total += len(hits)
    print("  %-12s pred=%-30s sim=%s  %d/%d" %
          (i, ",".join(ids[b] for b in best), [round(s, 2) for s in bestsim], sum(hits), len(hits)))

acc = correct / max(1, total)
print("\nTOP-1 IDENTITY ACCURACY (%s): %.1f%%  (%d/%d)" % (tag, acc * 100, correct, total))
json.dump({"tag": tag, "style": args.style, "n": args.n, "accuracy": round(acc, 4),
           "correct": correct, "total": total, "per_id": per_id},
          open(os.path.join(OUT, "report.json"), "w"), ensure_ascii=False, indent=1)
