"""Audit the DINOv2 nearest-centroid metric itself, before trusting any LoRA number.

Splits each identity's clean images in half: centroid from half A, test on half B.
An identity whose own photos cannot be matched back to their own centroid is one the
metric cannot judge -- so we report LoRA accuracy only on the trustworthy subset.
"""
import os, json, argparse, torch, numpy as np
from PIL import Image
from transformers import Dinov2Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
CLEAN = os.path.join(ROOT, "data", "clean")
ap = argparse.ArgumentParser()
ap.add_argument("--ids", default=None, help="restrict to a comma list; the real-photo ceiling depends on candidate-set size")
args = ap.parse_args()
DEV = "cuda"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

model = Dinov2Model.from_pretrained("facebook/dinov2-base").eval().to(DEV)


def prep(p):
    im = Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)
    a = np.asarray(im, np.float32) / 255.0
    return torch.from_numpy((a - MEAN) / STD).permute(2, 0, 1)


def embed(paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([prep(p) for p in paths[i:i + bs]]).to(DEV)
        with torch.no_grad():
            r = model(pixel_values=x).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(r, dim=1).cpu())
    return torch.cat(out)


ids = [i for i in META if META[i]["kind"] in ("ultra", "kaiju")
       and os.path.isdir(os.path.join(CLEAN, i))]
if args.ids:
    ids = [x.strip() for x in args.ids.split(",")]
SUFFIX = "-8way" if args.ids else ""
A, B = {}, {}
for i in ids:
    ps = sorted(os.path.join(CLEAN, i, f) for f in os.listdir(os.path.join(CLEAN, i))
                if f.endswith(".jpg"))
    k = max(2, len(ps) // 2)
    A[i], B[i] = ps[:k], (ps[k:] or ps[-2:])

cent = torch.cat([torch.nn.functional.normalize(embed(A[i]).mean(0, keepdim=True), dim=1)
                  for i in ids])

rows = []
for j, i in enumerate(ids):
    e = embed(B[i])
    sims = e @ cent.T
    pred = sims.argmax(1).tolist()
    recall = sum(1 for b in pred if ids[b] == i) / len(pred)
    margin = float((sims[:, j] - sims.masked_fill(
        torch.eye(len(ids), dtype=torch.bool)[j].expand_as(sims), -9).max(1).values).mean())
    rows.append({"id": i, "zh": META[i]["zh"], "n": len(pred),
                 "recall": round(recall, 3), "margin": round(margin, 3)})

rows.sort(key=lambda r: -r["recall"])
tot = sum(r["recall"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
trusted = [r["id"] for r in rows if r["recall"] >= 0.7]
print("overall real-photo holdout top-1: %.1f%% over %d ids / %d photos"
      % (100 * tot, len(ids), sum(r["n"] for r in rows)))
print("metric-trustworthy subset (real recall >= 0.70): %d ids" % len(trusted))
print("  ", ",".join(trusted))
print("\nper-identity (sorted by real recall):")
for r in rows:
    print("  %-10s %-8s n=%-3d recall=%.2f margin=%+.3f%s"
          % (r["id"], r["zh"], r["n"], r["recall"], r["margin"],
             "  <-- metric cannot judge this id" if r["recall"] < 0.7 else ""))
json.dump({"overall": round(float(tot), 4), "trusted": trusted, "rows": rows},
          open(os.path.join(ROOT, "data", "metric_audit%s.json" % SUFFIX), "w"),
          ensure_ascii=False, indent=2)
