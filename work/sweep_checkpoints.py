"""Sweep every saved checkpoint through the identity metric and print a table.

    CUDA_VISIBLE_DEVICES=1 python3 sweep_checkpoints.py --dir out --n 1 --steps 28

Prints accuracy per checkpoint so the best one can be picked without guessing.
"""
import os, sys, json, glob, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="out")
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--pattern", default="lora_step*.safetensors")
ap.add_argument("--ids", default=None, help="comma separated subset for the sweep")
ap.add_argument("--lora-scale", type=float, default=1.0)
ap.add_argument("--style", default="name")
args = ap.parse_args()

ckpts = sorted(glob.glob(os.path.join(ROOT, args.dir, args.pattern)),
               key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit())))
print("%d checkpoints" % len(ckpts))

rows = []
for p in ckpts:
    tag = os.path.splitext(os.path.basename(p))[0]
    cmd = [sys.executable, "eval_identity.py", "--lora", p, "--n", str(args.n),
           "--steps", str(args.steps), "--tag", tag, "--lora-scale", str(args.lora_scale),
           "--style", args.style]
    if args.ids:
        cmd += ["--ids", args.ids]
    rc = subprocess.call(cmd, cwd=WORK)
    rep = os.path.join(ROOT, "data", "eval", tag, "report.json")
    acc = json.load(open(rep))["accuracy"] if os.path.exists(rep) else -1
    rows.append((tag, acc, rc))
    print("  %-24s accuracy %.1f%%  (exit %d)" % (tag, acc * 100, rc))

print("\n%-26s %s" % ("checkpoint", "top-1 identity accuracy"))
for tag, acc, _ in rows:
    print("%-26s %.1f%%" % (tag, acc * 100))
best = max(rows, key=lambda r: r[1])
print("\nBEST: %s  %.1f%%" % (best[0], best[1] * 100))
json.dump([{"ckpt": t, "accuracy": a} for t, a, _ in rows],
          open(os.path.join(ROOT, "data", "sweep.json"), "w"), indent=1)
