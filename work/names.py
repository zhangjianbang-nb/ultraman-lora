"""Chinese -> English prompt normaliser for the Ultraman LoRA.

CLIP was trained mostly on English, so a bare Chinese prompt ("迪迦") carries almost no
usable signal for the UNet. This module maps every Chinese name (and the short forms
people actually type) onto the exact English caption prefix the LoRA was trained on,
so `生迪迦` becomes `tiga ultraman, giant silver and purple hero ...`.

    from names import normalize
    normalize("生成迪迦奥特曼在东京")   # -> "tiga ultraman, <desc>, 在东京, <suffix>"
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = json.load(open(os.path.join(ROOT, "work", "identity_meta.json")))

# short forms people type: 迪迦 / 赛罗 / 泰罗 ... plus common romanisations
EXTRA = {
    "tiga": ["迪迦", "迪加"],
    "zero": ["赛罗", "塞罗"],
    "ultraman": ["初代", "初代奥特曼", "奥特曼"],
    "seven": ["赛文", "七星侠"],
    "jack": ["杰克"],
    "ace": ["艾斯", "Ace"],
    "taro": ["泰罗", "泰罗奥特曼"],
    "leo": ["雷欧", "雷欧奥特曼"],
    "eighty": ["爱迪", "尤莉安"],
    "dyna": ["戴拿"],
    "gaia": ["盖亚"],
    "agul": ["阿古茹"],
    "cosmos": ["高斯"],
    "nexus": ["奈克瑟斯", "奈克赛斯"],
    "mebius": ["梦比优斯", "梅比乌斯"],
    "hikari": ["希卡利"],
    "ginga": ["银河"],
    "orb": ["欧布"],
    "geed": ["捷德"],
    "z": ["泽塔", "泽塔奥特曼"],
    "trigger": ["特利迦"],
    "decker": ["德凯"],
    "blazar": ["布莱泽"],
    "arc": ["亚刻"],
    "gomora": ["哥莫拉"],
    "zetton": ["杰顿", "泽顿"],
    "baltan": ["巴尔坦", "巴尔坦星人"],
    "redking": ["雷德王", "红王"],
    "eleking": ["艾雷王"],
    "mephilas": ["美菲拉斯", "美菲拉斯星人"],
    "tyrant": ["泰兰特"],
    "gatanothor": ["加坦杰厄"],
    "golza": ["哥尔赞"],
    "empera": ["安培拉", "安培拉星人", "恩培拉"],
}

ALIASES = {}
for _id, _m in META.items():
    ALIASES[_m["zh"]] = _id
    ALIASES[_m["en"]] = _id
    for _a in EXTRA.get(_id, []):
        ALIASES[_a] = _id

# exactly the short caption the LoRA was trained on with caption dropout:
# "<english name>, tokusatsu, full body"
SUFFIX = "tokusatsu, full body, cinematic lighting, best quality"

# The training images are scraped web pictures: many carry subtitles, magazine watermarks
# and two-panel collages, and the LoRA learned to reproduce that. A negative prompt is the
# cheapest fix -- it costs nothing at training time and removes most of the pollution.
NEGATIVE = ("text, watermark, subtitle, logo, lettering, japanese text, "
            "multiple people, two characters, collage, split screen, "
            "close-up portrait, cropped head, extra limbs, deformed")


def _caption(i, style="desc"):
    """Two caption paths the LoRA was trained on.

    style="name"  -- name only; this is the pure binding test (does the NAME alone work?)
    style="desc"  -- name + the canonical appearance sentence; stronger for actual
                     generation, because it is the exact full caption the LoRA saw 60%
                     of the time during training.
    """
    if style == "name":
        return META[i]["en"]
    return "%s, %s" % (META[i]["en"], META[i]["desc"])


def find_names(text):
    """Longest-match scan; returns [(id, matched_text)] in order of appearance."""
    out = []
    for alias in sorted(ALIASES, key=len, reverse=True):
        for mt in re.finditer(re.escape(alias), text, flags=re.I):
            span = mt.span()
            if any(s <= span[0] < e for s, e, _ in out):
                continue
            out.append((span[0], span[1], ALIASES[alias]))
    out.sort()
    # drop overlaps, keep leftmost-longest
    picked, last_end = [], -1
    for s, e, i in out:
        if s >= last_end:
            picked.append((s, e, i))
            last_end = e
    return [(i, text[s:e]) for s, e, i in picked]


def normalize(text, suffix=SUFFIX, keep_extra=True, style="desc"):
    """Rewrite a Chinese prompt into the English tag space the LoRA knows."""
    hits = find_names(text)
    if not hits:
        return text  # not an Ultraman request -- leave it to the base model
    caps = []
    for i, _ in hits:
        if i not in [c.split(",")[0] for c in caps]:
            caps.append(_caption(i, style))
    rest = text
    for _, mt in hits:
        rest = rest.replace(mt, " ")
    rest = re.sub(r"\s+", " ", rest).strip(" ,，、")
    rest = re.sub(r"^(生成|请|帮我|给我|画|生|来|一只|一张|一个)+", "", rest).strip(" ,，、")
    # CLIP cannot read Chinese, so leftover Chinese scene text is dropped instead of
    # being fed in as noise; ASCII extras (e.g. "at night, raining") are kept.
    if rest and not rest.isascii():
        rest = " ".join(w for w in rest.split() if w.isascii())
    parts = caps + ([rest] if keep_extra and rest else [])
    if suffix:
        parts.append(suffix)
    # dedupe tags: "tokusatsu suit, tokusatsu, full body" would otherwise repeat itself
    seen, out = set(), []
    for p in parts:
        for tag in [t.strip() for t in p.split(",") if t.strip()]:
            k = tag.lower()
            if k not in seen:
                seen.add(k)
                out.append(tag)
    return ", ".join(out)


def known(text):
    return [i for i, _ in find_names(text)]


if __name__ == "__main__":
    import sys
    for q in (sys.argv[1:] or ["生迪迦", "生成赛罗和泽塔打杰顿", "一只猫"]):
        print("%-24s -> %s" % (q, normalize(q)))
