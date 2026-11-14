# -*- coding: utf-8 -*-
"""
Mention-level coverage (fix, 2026-09-19).

The claim extractor only emits noun-phrase heads, but CHAIR counts EVERY
COCO-vocabulary token: colour adjectives ('orange cat'), compound modifiers
('train station', 'computer mouse'), and fragments a token budget cuts off
('cups, bow'). Those mentions were never verified, so no policy could remove
them. This pass finds every CHAIR mention that no scored claim in the same
sentence already covers, verifies it with the same yes/no question, and
writes extras_<name>_<split>.jsonl. Extras are deletion-only claims.

Tokenisation mirrors msrc.chair_utils.caption_to_categories exactly, with
character offsets kept so the mention can be removed from the sentence.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python augment.py --backbone llava \
          --split ktest --tag _t20
"""
import argparse
import json
import os
import re

import unified as U
from msrc.chair_utils import WORD2CAT, DOUBLE_WORDS
from msrc.lattice import split_sentences, singularize

TOK = re.compile(r"[^\s,\.]+|[,\.]")


def mentions(sentence):
    """-> [(cat, word, start, end)] exactly as CHAIR would count them."""
    toks = [(m.group(0), m.start(), m.end()) for m in TOK.finditer(sentence)]
    words = [singularize(t.lower()) for t, _, _ in toks]
    out, i = [], 0
    while i < len(words):
        bg = " ".join(words[i:i + 2])
        if bg in DOUBLE_WORDS:
            out.append((DOUBLE_WORDS[bg], bg, toks[i][1], toks[i + 1][2]))
            i += 2
            continue
        if words[i] in WORD2CAT:
            out.append((WORD2CAT[words[i]], words[i], toks[i][1], toks[i][2]))
        i += 1
    return out


def uncovered(rec):
    """CHAIR mentions with no scored claim of the same category in the same
    sentence."""
    res = []
    for si, s in enumerate(split_sentences(rec["caption"])):
        cov = {WORD2CAT.get(c["head"]) for c in rec["claims"]
               if c["si"] == si and "v0" in c}
        for cat, w, a, b in mentions(s):
            if cat not in cov:
                res.append({"si": si, "word": w, "cat": cat, "start": a,
                            "end": b})
    return res


if __name__ == "__main__":
    from msrc import backbones as B
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--img_dir", default=os.path.join(os.environ.get("MSRC_COCO", "data/coco"), "val2014"))
    args = ap.parse_args()
    name = args.backbone + args.tag
    recs = U.load_scores(name, args.split)
    out_path = f"out/extras_{name}_{args.split}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["file"] for l in open(out_path)}
    todo = [(r, uncovered(r)) for r in recs if r["file"] not in done]
    n_m = sum(len(u) for _, u in todo)
    print(f"[augment] {name} {args.split}: {len(todo)} images, {n_m} "
          f"uncovered mentions to verify", flush=True)
    bb = B.load(args.backbone)
    with open(out_path, "a") as f:
        for k, (r, ex) in enumerate(todo, 1):
            path = os.path.join(args.img_dir, r["file"])
            cache = {}
            for e in ex:
                art = "an" if e["word"][0] in "aeiou" else "a"
                q = f"Is there {art} {e['word']} in the image?"
                if q not in cache:
                    cache[q] = B.sigmoid(bb.gap(q + B.VERIF_SUFFIX, path))
                e["v0"] = cache[q]
            f.write(json.dumps({"file": r["file"], "extra": ex}) + "\n")
            f.flush()
            if k % 200 == 0:
                print(f"[augment] {k}/{len(todo)}", flush=True)
    print(f"[augment] done -> {out_path}", flush=True)
