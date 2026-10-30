"""Quantify how much the LoRA is copying training photos instead of generating.

For every rendered image, find the most similar TRAINING photo of the same identity
using a cheap 64x64 perceptual correlation. A memorised render correlates ~1.0 with
one specific training file; a genuine generation of the same character is ~0.3-0.6.
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "clean")
meta = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="data/review")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--thresh", type=float, default=0.90)
args = ap.parse_args()


def vec(p, s):
    im = Image.open(p).convert("L").resize((s, s), Image.BILINEAR)
    a = np.asarray(im, np.float32)
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else a


cache = {}
rows = []
for f in sorted(os.listdir(os.path.join(ROOT, args.renders))):
    if not f.endswith(".png"):
        continue
    i = f[:-4]
    if i not in meta or not os.path.isdir(os.path.join(CLEAN, i)):
        continue
    rv = vec(os.path.join(ROOT, args.renders, f), args.size)
    best, bestf = -1.0, ""
    for g in sorted(os.listdir(os.path.join(CLEAN, i))):
        if not g.endswith(".jpg"):
            continue
        key = os.path.join(CLEAN, i, g)
        if key not in cache:
            cache[key] = vec(key, args.size)
        c = float((rv * cache[key]).sum())
        if c > best:
            best, bestf = c, g
    rows.append({"id": i, "zh": meta[i]["zh"], "corr": round(best, 3), "match": bestf})

rows.sort(key=lambda r: -r["corr"])
n_mem = sum(1 for r in rows if r["corr"] >= args.thresh)
print("renders=%d  memorised(corr>=%.2f)=%d  mean_corr=%.3f"
      % (len(rows), args.thresh, n_mem, sum(r["corr"] for r in rows) / max(1, len(rows))))
print("\n%-10s %-10s corr   nearest training file" % ("id", "zh"))
for r in rows:
    print("  %-10s %-10s %.3f  %s%s" % (r["id"], r["zh"], r["corr"], r["match"],
                                        "  <== COPY" if r["corr"] >= args.thresh else ""))
json.dump({"n": len(rows), "memorised": n_mem,
           "mean_corr": round(sum(r["corr"] for r in rows) / max(1, len(rows)), 4),
           "rows": rows},
          open(os.path.join(ROOT, "data", "memorization.json"), "w"),
          ensure_ascii=False, indent=2)
