import os, json
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "clean")
META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
REP = json.load(open(os.path.join(ROOT, "data", "clean_report.json")))
T, COLS, LAB = 120, 8, 170

def sheet(ids, out, title):
    rows = len(ids)
    canvas = Image.new("RGB", (LAB + COLS * T, rows * T + 34), (20, 20, 24))
    d = ImageDraw.Draw(canvas)
    d.text((8, 10), title, fill=(255, 255, 255))
    for r, i in enumerate(ids):
        dd = os.path.join(CLEAN, i)
        files = sorted(f for f in os.listdir(dd) if f.endswith(".jpg"))[:COLS]
        d.text((8, 34 + r * T + 40), i, fill=(235, 235, 235))
        d.text((8, 34 + r * T + 58), META[i]["zh"], fill=(200, 200, 200))
        d.text((8, 34 + r * T + 76), "kept %d/%d" % (REP[i]["kept"], REP[i]["raw"]), fill=(150, 150, 150))
        for c, f in enumerate(files):
            im = Image.open(os.path.join(dd, f)).convert("RGB")
            im.thumbnail((T - 6, T - 6))
            canvas.paste(im, (LAB + c * T + (T - im.width) // 2,
                              34 + r * T + (T - im.height) // 2))
    canvas.save(out, quality=90)
    print(out, canvas.size)

ultra = [i for i in META if META[i]["kind"] == "ultra" and os.path.isdir(os.path.join(CLEAN, i))]
kaiju = [i for i in META if META[i]["kind"] == "kaiju" and os.path.isdir(os.path.join(CLEAN, i))]
actors = [i for i in META if META[i]["kind"] == "actor" and os.path.isdir(os.path.join(CLEAN, i))]
sheet(sorted(ultra), os.path.join(ROOT, "data", "sheet_ultra.jpg"), "ultra (%d identities)" % len(ultra))
sheet(sorted(kaiju), os.path.join(ROOT, "data", "sheet_kaiju.jpg"), "kaiju (%d identities)" % len(kaiju))
sheet(sorted(actors), os.path.join(ROOT, "data", "sheet_actors.jpg"), "actors (archived, NOT trained)")

lines = ["# 数据集清单（DINOv2 清洗后）", "",
         "| 身份 | 中文 | 类型 | 原始下载 | 清洗保留 | 平均相似度 |", "|---|---|---|---|---|---|"]
for i in sorted(META, key=lambda x: (META[x]["kind"], x)):
    if i not in REP: continue
    r = REP[i]
    lines.append("| %s | %s | %s | %d | %d | %.3f |" % (i, META[i]["zh"], META[i]["kind"],
                                                        r["raw"], r["kept"], r["sim_mean"]))
open(os.path.join(ROOT, "data", "dataset_report.md"), "w").write("\n".join(lines) + "\n")
print("report written")
