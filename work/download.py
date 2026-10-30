"""Download scraped image URLs into data/raw/<id>/ with validation and dedupe.

Memory-frugal: results are consumed as they complete (as_completed), each image is
resized to <=max-side and written immediately, progress appended to a JSONL log so a
kill never loses work.

    python3 -u work/download.py work/urls.json [--workers 12] [--min-side 384]
"""
import os, sys, json, io, hashlib, argparse, urllib.request, concurrent.futures as cf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

ap = argparse.ArgumentParser()
ap.add_argument("urls")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--min-side", type=int, default=384)
ap.add_argument("--max-side", type=int, default=1024)
ap.add_argument("--timeout", type=int, default=20)
args = ap.parse_args()

data = json.load(open(args.urls))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)
LOG = os.path.join(ROOT, "data", "download_log.jsonl")

from PIL import Image


def fetch_one(job):
    ident, url = job
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Referer": "https://www.bing.com/",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        })
        raw = urllib.request.urlopen(req, timeout=args.timeout).read()
        if len(raw) < 4000:
            return ident, None, "too_small_bytes"
        im = Image.open(io.BytesIO(raw))
        im.load()
        w, h = im.size
        if min(w, h) < args.min_side:
            return ident, None, "low_res"
        if max(w, h) / max(1, min(w, h)) > 3.0:
            return ident, None, "extreme_aspect"
        im = im.convert("RGB")
        if max(w, h) > args.max_side:
            im.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        t = im.copy()
        t.thumbnail((64, 64))
        key = hashlib.md5(t.tobytes()).hexdigest()
        return ident, (key, im, w, h, url), None
    except Exception as e:
        return ident, None, type(e).__name__


jobs = [(i, u) for i, d in data.items() for u in d["urls"]]

# resume: skip URLs already saved (recorded in the jsonl log)
done_urls = set()
if os.path.exists(LOG):
    for line in open(LOG):
        try:
            done_urls.add(json.loads(line)["url"])
        except Exception:
            pass
before = len(jobs)
jobs = [j for j in jobs if j[1] not in done_urls]
print("total urls %d, already done %d, remaining %d" % (before, before - len(jobs), len(jobs)), flush=True)

seen = {}
counts = {}
for line in (open(LOG) if os.path.exists(LOG) else []):
    try:
        d = json.loads(line)
        counts[d["id"]] = counts.get(d["id"], 0) + 1
    except Exception:
        pass
for i in data:
    counts.setdefault(i, 0)
stats = {}
done = 0
with open(LOG, "a") as logf, cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = [ex.submit(fetch_one, j) for j in jobs]
    for fut in cf.as_completed(futs):
        ident, ok, err = fut.result()
        done += 1
        if ok is None:
            stats[err] = stats.get(err, 0) + 1
        else:
            key, im, w, h, url = ok
            if key in seen:
                stats["duplicate"] = stats.get("duplicate", 0) + 1
            else:
                seen[key] = ident
                d = os.path.join(RAW, ident)
                os.makedirs(d, exist_ok=True)
                n = counts[ident]
                p = os.path.join(d, "%s_%03d.jpg" % (ident, n))
                im.save(p, quality=95)
                counts[ident] = n + 1
                logf.write(json.dumps({"id": ident, "file": os.path.basename(p),
                                       "w": w, "h": h, "url": url}) + "\n")
                logf.flush()
        if done % 250 == 0:
            print("  %d/%d kept=%d" % (done, len(jobs), sum(counts.values())), flush=True)

print("kept per id:", flush=True)
for i, c in sorted(counts.items(), key=lambda x: -x[1]):
    print("  %-12s %3d" % (i, c), flush=True)
print("rejected:", stats, flush=True)
json.dump({"counts": counts, "rejected": stats},
          open(os.path.join(ROOT, "data", "raw_index.json"), "w"), indent=1)
print("total kept", sum(counts.values()), flush=True)
