"""Second-stage data filter: drop collages, TV screenshots and magazine scans.

The centroid filter cannot catch these because a two-panel collage of the SAME
character scores high on identity similarity. Structural cues instead:

  * seam      -- a hard vertical/horizontal line spanning most of the frame
                 (two-panel web collage, magazine gutter)
  * letterbox -- flat black/white bands on an edge (TV screenshot, page scan)
  * textband  -- a thin horizontal strip packing most of the image's strong
                 horizontal edges (subtitle, watermark strip)

Images are ranked by DINOv2 similarity to the identity centroid, so if an identity
falls below --min-keep we add back its best-scoring dropped images.
"""
import os, sys, json, argparse, shutil
import numpy as np
from PIL import Image
import torch
from transformers import Dinov2Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "clean")
DST = os.path.join(ROOT, "data", "clean3")
META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
DEV = "cuda"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("--min-keep", type=int, default=12)
ap.add_argument("--max-keep", type=int, default=28)
ap.add_argument("--seam-ratio", type=float, default=3.0)
ap.add_argument("--seam-span", type=float, default=0.45)
ap.add_argument("--band-std", type=float, default=3.0)
ap.add_argument("--text-frac", type=float, default=0.30)
ap.add_argument("--text-height", type=float, default=0.14)
args = ap.parse_args()

S = 256


def metrics(p):
    im = Image.open(p).convert("RGB")
    a = np.asarray(im.resize((S, S), Image.BILINEAR), np.float32).mean(2)
    out = {}
    band = max(2, int(0.055 * S))
    edges = [a[:band], a[-band:], a[:, :band], a[:, -band:]]
    # A plain studio background is ALSO flat, so flatness alone is not letterboxing.
    # A real black bar / scanned page margin is flat AND very dark or very bright.
    out["border_std"] = float(min(e.std() for e in edges))
    out["border_mean"] = float(min(e.mean() for e in edges))
    out["letterbox"] = (out["border_std"] < 3.0
                        and (out["border_mean"] < 40 or out["border_mean"] > 215))

    dv = np.abs(np.diff(a, axis=1))          # vertical boundaries
    dh = np.abs(np.diff(a, axis=0))          # horizontal boundaries
    med = float(np.median(dv)) + 1e-6
    span_v = float((dv > 35).mean(0).max())  # best column: fraction of rows that are sharp
    out["seam_v"] = span_v * (float(dv.mean(0).max()) / med > args.seam_ratio)
    medh = float(np.median(dh)) + 1e-6
    span_h = float((dh > 35).mean(1).max())
    out["seam_h"] = span_h * (float(dh.mean(1).max()) / medh > args.seam_ratio)

    gx = np.abs(np.diff(a, axis=1))
    rows = (gx > 60).mean(1)                 # per-row fraction of strong edges
    h = max(2, int(args.text_height * S))
    win = np.convolve(rows, np.ones(h) / h, mode="valid")
    out["textband"] = float(win.max())
    return out


def keep(m):
    reasons = []
    if m.get("letterbox"):
        reasons.append("letterbox")
    if max(m["seam_v"], m["seam_h"]) > args.seam_span:
        reasons.append("seam")
    if m["textband"] > args.text_frac:
        reasons.append("textband")
    return reasons


model = Dinov2Model.from_pretrained("facebook/dinov2-base").eval().to(DEV)


def prep(p):
    im = Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)
    x = np.asarray(im, np.float32) / 255.0
    return torch.from_numpy((x - MEAN) / STD).permute(2, 0, 1)


def embed(paths, bs=32):
    o = []
    for i in range(0, len(paths), bs):
        x = torch.stack([prep(p) for p in paths[i:i + bs]]).to(DEV)
        with torch.no_grad():
            r = model(pixel_values=x).last_hidden_state[:, 0]
        o.append(torch.nn.functional.normalize(r, dim=1).cpu())
    return torch.cat(o)


if os.path.isdir(DST):
    shutil.rmtree(DST)
os.makedirs(DST)
report = {}
tot_kept = tot_drop = 0
for i, m in META.items():
    d = os.path.join(SRC, i)
    if not os.path.isdir(d):
        continue
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    if not files:
        continue
    paths = [os.path.join(d, f) for f in files]
    e = embed(paths)
    cent = torch.nn.functional.normalize(e.mean(0, keepdim=True), dim=1)
    sims = (e @ cent.T).squeeze(1).tolist()
    scored = []
    for f, p, s in zip(files, paths, sims):
        mm = metrics(p)
        scored.append({"f": f, "p": p, "sim": s, "reasons": keep(mm), "m": mm})
    scored.sort(key=lambda r: -r["sim"])
    good = [r for r in scored if not r["reasons"]]
    bad = [r for r in scored if r["reasons"]]
    sel = good[:args.max_keep]
    if len(sel) < args.min_keep:
        sel = sel + bad[:args.min_keep - len(sel)]
    os.makedirs(os.path.join(DST, i))
    for r in sel:
        shutil.copy2(r["p"], os.path.join(DST, i, r["f"]))
    tot_kept += len(sel)
    tot_drop += len(scored) - len(sel)
    reasons = {}
    for r in bad:
        for x in r["reasons"]:
            reasons[x] = reasons.get(x, 0) + 1
    report[i] = {"in": len(scored), "kept": len(sel), "dropped": len(bad),
                 "forced_back": max(0, args.min_keep - len(good[:args.max_keep])),
                 "reasons": reasons, "sim_mean": round(float(np.mean([r["sim"] for r in sel])), 3)}
    print("%-10s in=%-3d kept=%-3d dropped=%-3d %s" %
          (i, len(scored), len(sel), len(bad), reasons), flush=True)

print("\nTOTAL kept=%d dropped=%d" % (tot_kept, tot_drop))
json.dump(report, open(os.path.join(ROOT, "data", "clean3_report.json"), "w"),
          ensure_ascii=False, indent=2)
