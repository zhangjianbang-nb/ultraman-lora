"""SDXL LoRA training for Ultraman identity binding.

Two phases:
  A) precompute latents (VAE) + text embeddings (both CLIP text encoders) -> cache.pt
  B) train LoRA on the UNet only (text encoders frozen), gradient checkpointing, bf16

    python3 train_lora.py --manifest data/manifest.json --cache data/cache.pt \
        --steps 3000 --rank 32 --lr 1e-4 --out ~/ultraman/out

Phase A is skipped automatically if the cache already exists and matches the manifest.
"""
import compat_shim  # noqa: F401  (must run before diffusers imports)
import os, json, math, time, argparse, hashlib, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/ultraman/models/sdxl-base-1.0"))
ap.add_argument("--manifest", default="data/manifest.json")
ap.add_argument("--cache", default="data/cache.pt")
ap.add_argument("--out", default=os.path.expanduser("~/ultraman/out"))
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--rebuild-cache", action="store_true")
ap.add_argument("--caption-dropout", type=float, default=0.4,
                help="probability of training on the SHORT caption (name only) instead of "
                     "the full descriptive one, so the name alone becomes a trigger")
ap.add_argument("--kinds", default="ultra,kaiju",
                help="comma separated kinds to train on; empty = all")
args = ap.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
DEV = "cuda"
DT = torch.bfloat16


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(args.out, "train.log"), "a") as f:
        f.write(line + "\n")


manifest = json.load(open(args.manifest))
if args.kinds:
    keep = set(args.kinds.split(","))
    manifest = [m for m in manifest if m["kind"] in keep]
def short_caption(m):
    """Name-only caption: this is what makes a bare "tiga ultraman" prompt work."""
    return "%s, tokusatsu, full body" % m["caption"].split(",")[0]


log("manifest: %d images, %d identities, caption-dropout %.2f" %
    (len(manifest), len({m["identity"] for m in manifest}), args.caption_dropout))

# ---------- Phase A: precompute ----------
if os.path.exists(args.cache) and not args.rebuild_cache:
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    log("cache loaded: %d entries" % len(cache))
else:
    from diffusers import StableDiffusionXLPipeline
    log("loading pipeline for encoding...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    # SDXL's VAE overflows in fp16 (every latent comes out NaN); encode in fp32.
    vae = pipe.vae.to(DEV, dtype=torch.float32)
    te1 = pipe.text_encoder.to(DEV)
    te2 = pipe.text_encoder_2.to(DEV)
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    scaling = vae.config.scaling_factor

    cache = []
    t0 = time.time()
    for i, m in enumerate(manifest):
        im = Image.open(m["path"]).convert("RGB")
        # center-crop square, then resize to target resolution
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((args.res, args.res), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEV, torch.float32)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample().float() * scaling
            if torch.isnan(lat).any():
                raise RuntimeError("NaN latent for %s -- VAE encoding is broken" % m["path"])
            # encode with both CLIP text encoders exactly like the SDXL pipeline does
            # (hidden_states[-2] + text_embeds); spelled out so it works on any diffusers
            t1 = tok1(m["caption"], padding="max_length", max_length=tok1.model_max_length,
                      truncation=True, return_tensors="pt").to(DEV)
            t2 = tok2(m["caption"], padding="max_length", max_length=tok2.model_max_length,
                      truncation=True, return_tensors="pt").to(DEV)
            e1 = te1(t1.input_ids, output_hidden_states=True).hidden_states[-2]
            o2 = te2(t2.input_ids, output_hidden_states=True)
            emb = torch.cat([e1, o2.hidden_states[-2]], dim=-1).to(torch.float16)
            pooled = o2.text_embeds.to(torch.float16)
            short = short_caption(m)
            s1 = tok1(short, padding="max_length", max_length=tok1.model_max_length,
                      truncation=True, return_tensors="pt").to(DEV)
            s2 = tok2(short, padding="max_length", max_length=tok2.model_max_length,
                      truncation=True, return_tensors="pt").to(DEV)
            se1 = te1(s1.input_ids, output_hidden_states=True).hidden_states[-2]
            so2 = te2(s2.input_ids, output_hidden_states=True)
            emb_s = torch.cat([se1, so2.hidden_states[-2]], dim=-1).to(torch.float16)
            pooled_s = so2.text_embeds.to(torch.float16)
        cache.append({"latent": lat[0].to(DT).cpu(),
                      "emb": emb[0].to(DT).cpu(),
                      "pooled": pooled[0].to(DT).cpu(),
                      "emb_short": emb_s[0].to(DT).cpu(),
                      "pooled_short": pooled_s[0].to(DT).cpu(),
                      "caption": m["caption"], "short": short,
                      "identity": m["identity"]})
        if (i + 1) % 50 == 0:
            log("  encoded %d/%d  (%.1fs)" % (i + 1, len(manifest), time.time() - t0))
    torch.save(cache, args.cache)
    log("cache written: %d entries, %.1fs" % (len(cache), time.time() - t0))
    del pipe, vae, te1, te2
    torch.cuda.empty_cache()

# ---------- Phase B: train LoRA on UNet ----------
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
from peft import LoraConfig, get_peft_model

log("device: %s (%s)" % (torch.cuda.get_device_name(0),
                          os.environ.get("CUDA_VISIBLE_DEVICES", "unset")))
log("loading unet...")
unet = UNet2DConditionModel.from_pretrained(
    args.model, subfolder="unet", torch_dtype=DT, variant="fp16", use_safetensors=True)
unet.requires_grad_(False)
unet.enable_gradient_checkpointing()
unet.to(DEV)

# exact SDXL noise schedule (scaled_linear, beta 0.00085 -> 0.012)
sched = EulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
alphas_cumprod = torch.tensor(sched.alphas_cumprod, dtype=torch.float32, device=DEV)
log("schedule: %d timesteps, ac[0]=%.4f ac[-1]=%.4f" %
    (len(alphas_cumprod), alphas_cumprod[0].item(), alphas_cumprod[-1].item()))

targets = ["to_q", "to_k", "to_v", "to_out.0"]
cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian",
                 target_modules=targets, lora_dropout=0.0)
unet = get_peft_model(unet, cfg)
unet.print_trainable_parameters()
trainable = [p for p in unet.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
                        eps=1e-8)

N = len(cache)
add_time_ids = torch.tensor([args.res, args.res, 0, 0, args.res, args.res],
                            dtype=DT, device=DEV).unsqueeze(0)


def get_batch(idx):
    lat = torch.stack([cache[j]["latent"] for j in idx]).to(DEV)
    emb_l, pool_l = [], []
    for j in idx:
        if random.random() < args.caption_dropout:
            emb_l.append(cache[j]["emb_short"])
            pool_l.append(cache[j]["pooled_short"])
        else:
            emb_l.append(cache[j]["emb"])
            pool_l.append(cache[j]["pooled"])
    emb = torch.stack(emb_l).to(DEV)
    pooled = torch.stack(pool_l).to(DEV)
    tids = add_time_ids.expand(len(idx), -1)
    return lat, emb, pooled, tids


warmup = 100
step = 0
t0 = time.time()
loss_acc, acc_n = 0.0, 0
perm = torch.randperm(N).tolist()
pos = 0

log("start training: %d steps, batch %d, rank %d, lr %g" % (args.steps, args.batch, args.rank, args.lr))
while step < args.steps:
    if pos + args.batch > N:
        perm = torch.randperm(N).tolist()
        pos = 0
    idx = perm[pos:pos + args.batch]
    pos += args.batch
    lat, emb, pooled, tids = get_batch(idx)
    noise = torch.randn_like(lat)
    bsz = lat.shape[0]
    timesteps = torch.randint(0, 1000, (bsz,), device=DEV, dtype=torch.long)

    ac = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    noisy = ac.sqrt() * lat + (1 - ac).sqrt() * noise

    with torch.autocast(DEV, dtype=DT):
        pred = unet(noisy, timesteps, encoder_hidden_states=emb,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}).sample
    loss = F.mse_loss(pred.float(), noise.float())

    # lr schedule: linear warmup then cosine
    if step < warmup:
        lr = args.lr * (step + 1) / warmup
    else:
        prog = (step - warmup) / max(1, args.steps - warmup)
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups:
        g["lr"] = lr

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()

    loss_acc += loss.item()
    acc_n += 1
    step += 1
    if step % args.log_every == 0:
        avg = loss_acc / acc_n
        sps = step / (time.time() - t0)
        eta = (args.steps - step) / max(1e-6, sps) / 60
        log("step %d/%d loss %.4f lr %.2e %.2fs/step eta %.1f min" %
            (step, args.steps, avg, lr, 1 / sps, eta))
        loss_acc, acc_n = 0.0, 0

    if step % args.save_every == 0 or step == args.steps:
        from peft import get_peft_model_state_dict
        sd = get_peft_model_state_dict(unet)
        p = os.path.join(args.out, "lora_step%d.safetensors" % step)
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()}, p)
        log("saved %s (%d tensors)" % (p, len(sd)))

log("done in %.1f min" % ((time.time() - t0) / 60))
