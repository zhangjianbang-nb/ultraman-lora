import os, sys, json, random
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "clean")
META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
T = 110
COLS = 8
LAB = 150

def sheet(ids, out):
    rows = len(ids)
    W = LAB + COLS * T
    H = rows * T
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    for r, i in enumerate(ids):
        files = sorted(f for f in os.listdir(os.path.join(CLEAN, i)) if f.endswith(".jpg"))[:COLS]
        d.text((6, r * T + 40), i, fill=(230, 230, 230))
        d.text((6, r * T + 58), META[i]["kind"], fill=(150, 150, 150))
        for c, f in enumerate(files):
            im = Image.open(os.path.join(CLEAN, i, f)).convert("RGB")
            im.thumbnail((T - 6, T - 6))
            x = LAB + c * T + (T - im.width) // 2
            y = r * T + (T - im.height) // 2
            canvas.paste(im, (x, y))
    canvas.save(out, quality=92)
    print(out, canvas.size)

actors = [i for i in META if META[i]["kind"] == "actor" and os.path.isdir(os.path.join(CLEAN, i))]
sheet(sorted(actors)[:13], os.path.join(ROOT, "data", "sheet_actors.jpg"))

others = [i for i in META if META[i]["kind"] != "actor" and os.path.isdir(os.path.join(CLEAN, i))]
random.seed(7)
pick = random.sample(others, 14)
sheet(sorted(pick), os.path.join(ROOT, "data", "sheet_spot.jpg"))
