"""Build a manifest straight from the hard-filtered clean2 tree."""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))
CLEAN = os.path.join(ROOT, "data", "clean3")
out = []
for i, m in META.items():
    if m["kind"] not in ("ultra", "kaiju"):
        continue
    d = os.path.join(CLEAN, i)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith(".jpg"):
            continue
        out.append({"path": os.path.join(d, f),
                    "caption": "%s, %s" % (m["en"], m["desc"]),
                    "identity": i, "kind": m["kind"]})
json.dump(out, open(os.path.join(ROOT, "data", "manifest2.json"), "w"), ensure_ascii=False, indent=1)
from collections import Counter
c = Counter(e["identity"] for e in out)
print("entries=%d ids=%d per-id min=%d max=%d" % (len(out), len(c), min(c.values()), max(c.values())))
