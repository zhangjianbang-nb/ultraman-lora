"""One-click acceptance for the Ultraman LoRA.

Runs the two gates the model has to pass, then renders a few Chinese-prompt demos:

  1. identity accuracy  -- DINOv2 nearest-centroid top-1 over every trained identity
                           (base model vs LoRA, so the gain is visible)
  2. capability regression -- generic prompts must stay close to the base model
  3. demo renders       -- "生迪迦" etc. through the Chinese-name normaliser

    CUDA_VISIBLE_DEVICES=1 python3 verify.py --lora ultraman-lora-r32.safetensors --n 2

Exit code 0 = PASS.
"""
import os, sys, json, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--ids", default=None)
ap.add_argument("--min-acc", type=float, default=0.60)
ap.add_argument("--min-drift", type=float, default=0.75)
ap.add_argument("--skip-base", action="store_true", help="skip the base-model baseline run")
ap.add_argument("--skip-cap", action="store_true")
args = ap.parse_args()

PY = sys.executable


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=WORK)


def report(path):
    return json.load(open(path)) if os.path.exists(path) else {}


ok = True

# ---- gate 1: identity accuracy -------------------------------------------------
if not args.skip_base:
    run([PY, "eval_identity.py", "--n", str(args.n), "--tag", "base"] +
        (["--ids", args.ids] if args.ids else []))
run([PY, "eval_identity.py", "--lora", args.lora, "--rank", str(args.rank),
     "--n", str(args.n), "--tag", "lora"] +
    (["--ids", args.ids] if args.ids else []))

base = report(os.path.join(ROOT, "data", "eval", "base", "report.json"))
lora = report(os.path.join(ROOT, "data", "eval", "lora", "report.json"))
acc = lora.get("accuracy", 0.0)
print("\nidentity accuracy: base %.1f%% -> LoRA %.1f%% (need >= %.0f%%)" %
      (base.get("accuracy", 0) * 100, acc * 100, args.min_acc * 100))
ok = ok and acc >= args.min_acc

# ---- gate 2: capability regression ---------------------------------------------
if not args.skip_cap:
    run([PY, "check_base_capability.py", "--lora", args.lora, "--rank", str(args.rank),
         "--min-drift-sim", str(args.min_drift)])
    cap = report(os.path.join(ROOT, "data", "capcheck", "report.json"))
    print("\ncapability drift similarity: %.3f (need >= %.2f), degenerate %d" %
          (cap.get("mean_drift_sim", 0), args.min_drift, cap.get("degenerate", 0)))
    ok = ok and bool(cap.get("pass"))

# ---- gate 3: demo renders ------------------------------------------------------
demos = ["生迪迦", "生成赛罗奥特曼", "哥莫拉在废墟里咆哮", "一只猫"]
outdir = os.path.join(ROOT, "samples")
os.makedirs(outdir, exist_ok=True)
for k, q in enumerate(demos):
    run([PY, "infer.py", "--lora", args.lora, "--rank", str(args.rank),
         "--prompt", q, "--out", os.path.join(outdir, "demo_%d.png" % k)])

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
