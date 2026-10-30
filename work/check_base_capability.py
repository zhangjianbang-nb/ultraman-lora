"""Base-capability regression check: did the LoRA damage SDXL's general ability?

For a fixed set of prompts that have nothing to do with tokusatsu, render one image with
the base model and one with the LoRA (same seed), then measure how far the LoRA output
drifted from the base output in DINOv2 feature space. A LoRA that merely adds a new
concept keeps these two close; one that wrecked the UNet makes them diverge.

    CUDA_VISIBLE_DEVICES=1 python3 check_base_capability.py --lora out/lora_step12000.safetensors

Pass condition: mean drift similarity >= 0.75 and no degenerate (flat/black) images.
"""
import os, json, argparse
import torch
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/ultraman/models/sdxl-base-1.0"))
ap.add_argument("--lora", default=None)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--cfg", type=float, default=6.5)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--outdir", default="data/capcheck")
ap.add_argument("--min-drift-sim", type=float, default=0.75)
args = ap.parse_args()

PROMPTS = [
    "a photo of a tabby cat sitting on a wooden floor",
    "a red sports car parked on a city street at night",
    "a bowl of steaming ramen with an egg on top",
    "a snowy mountain landscape under a blue sky",
    "portrait of an old man with a grey beard, studio light",
    "a futuristic robot standing in a laboratory",
    "a quiet beach at sunset with palm trees",
    "a plate of sushi on a black table, top view",
]

os.makedirs(args.outdir, exist_ok=True)

from transformers import Dinov2Model
dino = Dinov2Model.from_pretrained("facebook/dinov2-base").eval().to("cuda")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def embed(im):
    a = np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    x = torch.from_numpy((a - MEAN) / STD).permute(2, 0, 1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        r = dino(pixel_values=x).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(r, dim=1)[0]


def degenerate(im):
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    return float(a.std()) < 3.0 or bool(np.isnan(a).any())


def make_pipe(lora=None):
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
    from lora_util import load_into
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    if lora:
        unet = UNet2DConditionModel.from_pretrained(
            args.model, subfolder="unet", torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True)
        unet, missing, unexpected = load_into(unet, lora, args.rank, args.rank)
        pipe.unet = unet.to("cuda")
        print("lora loaded (missing %d / unexpected %d)" % (len(missing), len(unexpected)))
    return pipe


def render(pipe, prompt, seed, path):
    g = torch.Generator(device="cuda").manual_seed(seed)
    im = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
              width=args.size, height=args.size, generator=g).images[0]
    im.save(path)
    return im


rows = []
print("== base model ==")
base_pipe = make_pipe(None)
base_ims = []
for k, p in enumerate(PROMPTS):
    im = render(base_pipe, p, args.seed + k, os.path.join(args.outdir, "base_%d.png" % k))
    base_ims.append(im)
    print("  %d %s  degenerate=%s" % (k, p[:40], degenerate(im)))
del base_pipe
torch.cuda.empty_cache()

print("== with LoRA ==")
lora_pipe = make_pipe(args.lora)
sims = []
bad = 0
for k, p in enumerate(PROMPTS):
    im = render(lora_pipe, p, args.seed + k, os.path.join(args.outdir, "lora_%d.png" % k))
    d = degenerate(im)
    bad += int(d)
    s = float(embed(im) @ embed(base_ims[k]))
    sims.append(s)
    rows.append({"prompt": p, "drift_sim": round(s, 3), "degenerate": d})
    print("  %d %s  drift_sim=%.3f degenerate=%s" % (k, p[:40], s, d))

mean = float(np.mean(sims))
ok = mean >= args.min_drift_sim and bad == 0
print("\nmean drift similarity: %.3f (min %.2f), degenerate images: %d" %
      (mean, args.min_drift_sim, bad))
print("RESULT:", "PASS" if ok else "FAIL")
json.dump({"lora": args.lora, "mean_drift_sim": round(mean, 4), "degenerate": bad,
           "pass": ok, "rows": rows},
          open(os.path.join(args.outdir, "report.json"), "w"), ensure_ascii=False, indent=1)
