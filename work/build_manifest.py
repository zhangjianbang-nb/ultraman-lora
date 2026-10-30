"""Clean the raw downloads and build the training manifest.

Steps per identity:
  1. compute DINOv2 embeddings for every image
  2. keep images within `--sim` cosine of the identity centroid (drops toys/logos/posters
     that the search query dragged in)
  3. drop images closer to ANOTHER identity's centroid (cross-contamination)
  4. rank by similarity to centroid, cap at --max-per-id
  5. write data/manifest.json + data/clean/<id>/ copies

    python3 build_manifest.py [--max-per-id 32] [--sim 0.55] [--dry-run]
"""
import os, json, argparse, shutil
import torch
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--max-per-id", type=int, default=32)
ap.add_argument("--min-per-id", type=int, default=14)
ap.add_argument("--sim", type=float, default=0.55)
ap.add_argument("--sim-actor", type=float, default=0.38)
ap.add_argument("--max-actor", type=int, default=24)
ap.add_argument("--min-actor", type=int, default=6)
ap.add_argument("--margin", type=float, default=0.10,
                help="allow own to lose to the best other identity by this much "
                     "(designs that genuinely look alike)")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")

from transformers import Dinov2Model
model = Dinov2Model.from_pretrained("facebook/dinov2-base").eval()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEV)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _prep(im, size=224):
    im = im.convert("RGB").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a).permute(2, 0, 1)


def embed(paths, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        x = torch.stack([_prep(Image.open(p)) for p in chunk]).to(DEV)
        with torch.no_grad():
            r = model(pixel_values=x).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(r, dim=1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 768)


ids = [i for i in META if os.path.isdir(os.path.join(RAW, i))]
print("identities with images:", len(ids))

per_id = {}
for i in ids:
    d = os.path.join(RAW, i)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
    per_id[i] = paths
    print("  %-12s raw %d" % (i, len(paths)))

# first pass: centroid from the whole set (median-ish via trimmed mean)
emb_all, cent0 = {}, {}
for i in ids:
    print("  embedding %s (%d)" % (i, len(per_id[i])), flush=True)
    e = embed(per_id[i])
    emb_all[i] = e
    # initial centroid = mean of the 60% most self-similar images (robust to junk)
    c = torch.nn.functional.normalize(e.mean(0, keepdim=True), dim=1)
    sims = (e @ c.T).squeeze(1)
    keep = sims.argsort(descending=True)[:max(3, int(0.6 * len(sims)))]
    cent0[i] = torch.nn.functional.normalize(e[keep].mean(0, keepdim=True), dim=1)

# refine centroid once using the robust subset
for i in ids:
    e = emb_all[i]
    sims = (e @ cent0[i].T).squeeze(1)
    keep = sims.argsort(descending=True)[:max(3, int(0.6 * len(sims)))]
    cent0[i] = torch.nn.functional.normalize(e[keep].mean(0, keepdim=True), dim=1)

cent_mat = torch.cat([cent0[i] for i in ids])  # (n_id, 768)

manifest, report = [], {}
for i in ids:
    e = emb_all[i]
    own = (e @ cent0[i].T).squeeze(1)
    all_sims = e @ cent_mat.T
    best_other, best_other_idx = all_sims.max(dim=1)
    own_idx = ids.index(i)
    # "closer to another identity" check: compare to the best other identity
    other = all_sims.clone()
    other[:, own_idx] = -2.0
    best_other, _ = other.max(dim=1)

    kind = META[i]["kind"]
    thr = args.sim_actor if kind == "actor" else args.sim
    cap = args.max_actor if kind == "actor" else args.max_per_id
    min_n = args.min_actor if kind == "actor" else args.min_per_id
    keep_idx = [j for j in range(len(e))
                if own[j].item() >= thr
                and own[j].item() >= best_other[j].item() - args.margin]
    keep_idx.sort(key=lambda j: -own[j].item())
    keep_idx = keep_idx[:cap]

    report[i] = {"raw": len(e), "kept": len(keep_idx),
                 "sim_mean": round(float(np.mean([own[j].item() for j in keep_idx])), 3) if keep_idx else 0.0}
    print("  %-12s raw %3d -> kept %2d (mean sim %.3f)" % (i, len(e), len(keep_idx), report[i]["sim_mean"]))

    if args.dry_run or len(keep_idx) < min_n:
        continue
    outdir = os.path.join(CLEAN, i)
    os.makedirs(outdir, exist_ok=True)
    m = META[i]
    cap = "%s, %s, %s" % (m["en"], m["zh"], m["desc"])
    for n, j in enumerate(keep_idx):
        src = per_id[i][j]
        dst = os.path.join(outdir, "%s_%02d.jpg" % (i, n))
        shutil.copyfile(src, dst)
        manifest.append({"path": dst, "caption": cap, "identity": i,
                         "kind": m["kind"], "sim": round(own[j].item(), 4)})

if not args.dry_run:
    json.dump(manifest, open(os.path.join(ROOT, "data", "manifest.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(report, open(os.path.join(ROOT, "data", "clean_report.json"), "w"), indent=1)
    print("\nmanifest: %d images across %d identities" %
          (len(manifest), len({x['identity'] for x in manifest})))
