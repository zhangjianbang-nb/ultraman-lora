"""One-click acceptance for the Ultraman LoRA.

Four gates, in the order they can actually be trusted:

  1. capability regression (HARD) -- 8 non-tokusatsu prompts, same seed, base vs LoRA.
     A LoRA that only adds a concept keeps these close; one that wrecked the UNet makes
     them diverge. Pass line: mean DINOv2 similarity >= 0.75 and no degenerate images.
  2. name influence (HARD) -- change the name, keep the seed; change the seed, keep the
     name. ratio = cross-name pixel diff / cross-seed pixel diff must go UP versus the
     base model, i.e. the name has to matter more with the LoRA than without it.
  3. identity accuracy (INFO ONLY) -- DINOv2 nearest-centroid on the 8 identities where
     the metric itself is validated. It is printed, never gated: renders live in a
     different visual domain than the real-photo centroids, so a low number here does
     not mean the character is wrong (docs/evaluation.md, sections 8-10).
  4. render sanity (HARD) -- the Chinese-prompt demos must exist and not be blank.

Identity correctness itself is a human call: open compare_sheet.jpg (reference photo /
base / LoRA-name / LoRA-desc, same seed) and look at it.

    CUDA_VISIBLE_DEVICES=1 python3 verify.py --lora ultraman-lora-r32.safetensors

Exit code 0 = the automated gates PASS.
"""
import os, sys, json, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
TRUSTED = "baltan,redking,gomora,gatanothor,golza,seven,eleking,tyrant"
ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lora-scale", type=float, default=1.0)
ap.add_argument("--min-drift", type=float, default=0.75)
ap.add_argument("--ids", default=TRUSTED, help="metric-validated identities for gate 3")
ap.add_argument("--n", type=int, default=2, help="renders per identity in gate 3")
ap.add_argument("--skip-info", action="store_true", help="skip gate 3 (it costs GPU time)")
args = ap.parse_args()

PY = sys.executable
ok = True


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=WORK)


def report(path):
    return json.load(open(path)) if os.path.exists(path) else {}


# ---- gate 1: capability regression ------------------------------------------------
run([PY, "check_base_capability.py", "--lora", args.lora, "--rank", str(args.rank),
     "--lora-scale", str(args.lora_scale), "--min-drift-sim", str(args.min_drift)])
cap = report(os.path.join(ROOT, "data", "capcheck", "report.json"))
drift = cap.get("mean_drift_sim", 0)
print("\n[gate 1] capability drift similarity %.3f (need >= %.2f), degenerate %d"
      % (drift, args.min_drift, cap.get("degenerate", 0)))
ok = ok and bool(cap.get("pass"))

# ---- gate 2: name influence -------------------------------------------------------
run([PY, "name_influence.py", "--tag", "verify_base"])
run([PY, "name_influence.py", "--lora", args.lora, "--rank", str(args.rank),
     "--lora-scale", str(args.lora_scale), "--tag", "verify_lora"])
b = report(os.path.join(ROOT, "data", "nameinf", "verify_base", "report.json"))
l = report(os.path.join(ROOT, "data", "nameinf", "verify_lora", "report.json"))
print("\n[gate 2] name influence ratio base %.3f -> LoRA %.3f (must increase)"
      % (b.get("ratio", 0), l.get("ratio", 0)))
ok = ok and l.get("ratio", 0) > b.get("ratio", 0)

# ---- gate 3: identity accuracy, informational only --------------------------------
if not args.skip_info:
    run([PY, "eval_identity.py", "--ids", args.ids, "--n", str(args.n),
         "--style", "name", "--tag", "verify_info", "--lora", args.lora,
         "--rank", str(args.rank), "--lora-scale", str(args.lora_scale)])
    info = report(os.path.join(ROOT, "data", "eval", "verify_info", "report.json"))
    print("\n[gate 3] INFO ONLY: %.1f%% on %d metric-validated identities "
          "(real photos score 88.3%% on the same metric; renders vs photo centroids have a "
          "domain gap, so this number cannot judge the LoRA)"
          % (100 * info.get("accuracy", 0), len(args.ids.split(","))))

# ---- gate 4: demo renders ---------------------------------------------------------
import numpy as np
from PIL import Image

demos = [("生迪迦", "demo_diga.png"), ("生成赛罗奥特曼在东京街头战斗", "demo_zero.png"),
         ("哥莫拉在废墟里咆哮", "demo_gomora.png"), ("一只猫", "demo_cat.png")]
outdir = os.path.join(ROOT, "samples")
os.makedirs(outdir, exist_ok=True)
for q, name in demos:
    run([PY, "infer.py", "--lora", args.lora, "--rank", str(args.rank),
         "--lora-scale", str(args.lora_scale), "--prompt", q,
         "--out", os.path.join(outdir, name)])

print("\n[gate 4] demo renders")
for q, name in demos:
    p = os.path.join(outdir, name)
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    blank = float(a.std()) < 0.05
    print("  %-16s %-16s std=%.3f %s" % (q, name, a.std(), "BLANK!" if blank else "ok"))
    ok = ok and not blank

print("\nRESULT:", "PASS" if ok else "FAIL")
print("Identity correctness is a visual call: open compare_sheet.jpg "
      "(reference / base / LoRA-name / LoRA-desc).")
sys.exit(0 if ok else 1)
