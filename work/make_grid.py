"""Tile generated images into contact sheets for visual acceptance.

    python3 make_grid.py --dir data/eval/lora --out grid_lora.jpg --cols 5 --thumb 220
"""
import os, sys, argparse
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cols", type=int, default=5)
ap.add_argument("--thumb", type=int, default=220)
ap.add_argument("--label", default=None, help="comma separated labels in file order")
args = ap.parse_args()

files = sorted(f for f in os.listdir(args.dir) if f.endswith(".png"))
if args.label:
    labels = [x.strip() for x in args.label.split(",")]
else:
    labels = [os.path.splitext(f)[0] for f in files]

rows = (len(files) + args.cols - 1) // args.cols
T = args.thumb
LAB = 90
canvas = Image.new("RGB", (LAB + args.cols * T, rows * T), (22, 22, 26))
d = ImageDraw.Draw(canvas)
for k, f in enumerate(files):
    r, c = divmod(k, args.cols)
    im = Image.open(os.path.join(args.dir, f)).convert("RGB")
    im.thumbnail((T - 6, T - 6))
    canvas.paste(im, (LAB + c * T + (T - im.width) // 2, r * T + (T - im.height) // 2))
    d.text((6, r * T + T // 2 - 6), labels[k][:14], fill=(225, 225, 225))
canvas.save(args.out, quality=92)
print("%s  %dx%d  %d tiles" % (args.out, canvas.width, canvas.height, len(files)))
