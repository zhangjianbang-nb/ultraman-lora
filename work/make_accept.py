"""Compose the two visual-acceptance contact sheets.

Sheet 1: one name-only generation per Ultraman (labels = identity id)
Sheet 2: remaining identities + the generic-prompt capability images + Chinese demos

    python3 make_accept.py --eval data/eval/lora --cap data/capcheck --demo samples
"""
import os, json, argparse
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="data/eval/lora")
ap.add_argument("--cap", default="data/capcheck")
ap.add_argument("--demo", default="samples")
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--thumb", type=int, default=190)
args = ap.parse_args()

META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
EVAL = os.path.join(ROOT, args.eval)
CAP = os.path.join(ROOT, args.cap)
DEMO = os.path.join(ROOT, args.demo)


def sheet(items, out, title):
    T, COLS, LAB = args.thumb, args.cols, 150
    rows = (len(items) + COLS - 1) // COLS
    canvas = Image.new("RGB", (LAB + COLS * T, rows * T + 30), (20, 20, 24))
    d = ImageDraw.Draw(canvas)
    d.text((8, 10), title, fill=(255, 255, 255))
    for k, (label, path) in enumerate(items):
        r, c = divmod(k, COLS)
        if not os.path.exists(path):
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((T - 6, T - 6))
        canvas.paste(im, (LAB + c * T + (T - im.width) // 2,
                          30 + r * T + (T - im.height) // 2))
        d.text((6, 30 + r * T + T // 2 - 6), label[:16], fill=(230, 230, 230))
    canvas.save(out, quality=90)
    print("%s  %dx%d  %d tiles" % (out, canvas.width, canvas.height, len(items)))


ultra = sorted([i for i in META if META[i]["kind"] == "ultra"])
kaiju = sorted([i for i in META if META[i]["kind"] == "kaiju"])

a = [(i, os.path.join(EVAL, "%s_0.png" % i)) for i in ultra[:18]]
sheet(a, os.path.join(ROOT, "accept_1_ultra.jpg"), "name-only prompt -> generated (ultra 1/2)")

b = [(i, os.path.join(EVAL, "%s_0.png" % i)) for i in ultra[18:] + kaiju]
c = [("CAP " + p[:10], os.path.join(CAP, "lora_%d.png" % k)) for k, p in enumerate([
    "cat", "car", "ramen", "mountain", "portrait", "robot", "beach", "sushi"])]
dd = [("DEMO %d" % k, os.path.join(DEMO, "demo_%d.png" % k)) for k in range(4)]
sheet(b + c + dd, os.path.join(ROOT, "accept_2_rest.jpg"),
      "name-only (ultra 2/2 + kaiju) | base capability | Chinese demos")
