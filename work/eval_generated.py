"""Identity metric that stays inside the generated-image domain.

Comparing renders against real-photo centroids fails for two reasons measured in
metric_audit.py: the metric cannot tell the 34 designs apart even for real photos
(49.8% ceiling), and renders live in a different visual domain than suit photos.
So instead: generate K images per identity with the SAME prompt template and build
centroids from the renders themselves, then leave-one-out top-1.

Base model has no name->identity binding, so it should score near chance (1/34).
A LoRA that binds names should cluster same-name renders and separate different ones.
"""
import os, sys, json, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_shim  # noqa: F401
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import Dinov2Model
from lora_util import load_into, upcast_vae

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/whu/ultraman/models/sdxl-base-1.0"
ap = argparse.ArgumentParser()
ap.add_argument("--lora", default=None)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lora-scale", type=float, default=1.0)
ap.add_argument("--n", type=int, default=3, help="renders per identity")
ap.add_argument("--ids", default=None)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--cfg", type=float, default=6.0)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=4242)
ap.add_argument("--tag", default=None)
args = ap.parse_args()

meta = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
ids = [i for i in meta if meta[i]["kind"] in ("ultra", "kaiju")]
if args.ids:
    ids = args.ids.split(",")
tag = args.tag or ("lora" if args.lora else "base")
OUT = os.path.join(ROOT, "data", "evalgen", tag)
os.makedirs(OUT, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
dino = Dinov2Model.from_pretrained("facebook/dinov2-base").eval().to("cuda")


def prep(im):
    a = np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), np.float32) / 255.0
    return torch.from_numpy((a - MEAN) / STD).permute(2, 0, 1)


def embed(imgs, bs=16):
    o = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([prep(m) for m in imgs[i:i + bs]]).to("cuda")
        with torch.no_grad():
            r = dino(pixel_values=x).last_hidden_state[:, 0]
        o.append(torch.nn.functional.normalize(r, dim=1).cpu())
    return torch.cat(o)


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
    print("lora %s scale %.2f" % (args.lora, args.lora_scale), flush=True)

imgs, owner = [], []
for i in ids:
    m = meta[i]
    prompt = "%s, tokusatsu, full body, cinematic lighting, best quality" % m["en"]
    for k in range(args.n):
        g = torch.Generator(device="cpu").manual_seed(args.seed + k)
        im = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
                  width=args.size, height=args.size, generator=g).images[0]
        im.save(os.path.join(OUT, "%s_%d.png" % (i, k)))
        imgs.append(im)
        owner.append(i)
    print("  rendered %s x%d" % (i, args.n), flush=True)

E = embed(imgs)
n_id = len(ids)
correct = 0
sims_all = E @ E.T
intra, inter = [], []
for a in range(len(imgs)):
    sims = sims_all[a].clone()
    sims[a] = -9
    best = int(sims.argmax())
    correct += (owner[best] == owner[a])
    for b in range(a + 1, len(imgs)):
        (intra if owner[a] == owner[b] else inter).append(float(sims_all[a, b]))

acc = correct / len(imgs)
chance = 1.0 / n_id
print("\n%s: leave-one-out top-1 on generated renders = %.1f%% (%d/%d), chance %.1f%%"
      % (tag, 100 * acc, correct, len(imgs), 100 * chance))
print("intra-identity sim %.3f | inter-identity sim %.3f | gap %+.3f"
      % (np.mean(intra), np.mean(inter), np.mean(intra) - np.mean(inter)))
json.dump({"tag": tag, "accuracy": round(acc, 4), "chance": round(chance, 4),
           "n_imgs": len(imgs), "n_ids": n_id,
           "intra": round(float(np.mean(intra)), 4),
           "inter": round(float(np.mean(inter)), 4),
           "gap": round(float(np.mean(intra) - np.mean(inter)), 4)},
          open(os.path.join(OUT, "report.json"), "w"), indent=2)
