# -*- coding: utf-8 -*-
"""
GPU pass of the unified pipeline: generate once, score everything once.

For every image this writes ONE json line holding the frozen vanilla caption
and, for every object claim in it, every score any repair policy could ask
for:

  v0          rung-0 verification  "Is there a {head} in the image?"
  rungs       entailment-ladder rungs 1..K   (only when v0 < SCORE_CUTOFF)
  sub_plain   substitute from SUB_PROMPT + its verification score
  sub_pos     substitute from the positional prompt + its verification score
              (both only when v0 < SCORE_CUTOFF)

All repair policies (deletion, MSRC, every ablation) are then replayed on CPU
by repair.py from these records, so they see byte-identical captions, claims
and scores. Nothing policy-specific happens here.

SCORE_CUTOFF only bounds work: any policy whose tau0 is below it gets exactly
the scores it would have computed live. Calibrated tau0 values are far below
it on every backbone (0.73 - 0.97).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python score.py --backbone llava \
      --split eval --shard 0 --nshards 4
"""
import argparse
import json
import os
import random
import time

from msrc.chair_utils import WORD2CAT
from msrc.lattice import (extract_claims, question_for, n_rungs,
                          split_sentences, singularize)
from msrc.guards import reground_prompt, mentioned_words
from msrc import backbones as B

SCORE_CUTOFF = 0.999
SPLITS = {"calib": (0, 500), "eval": (500, 3500),
          # pre-registered certification split (out/prereg_fresh_cert.json):
          # never scored, never used for selection
          "fresh": (3500, 5000),
          # CEBC protocol (Mishra et al., ACL 2026): 500 Karpathy-test images
          # for evaluation; MSRC calibrates on 500 disjoint Karpathy-val images.
          "ktest": ("data/coco_karpathy_test.json", 500),
          "kcal": ("data/coco_karpathy_val.json", 500)}
KSEED = 242        # CEBC's CHAIR seed

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True, choices=list(B.BACKBONES))
ap.add_argument("--split", required=True, choices=list(SPLITS))
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--img_dir", default=os.path.join(os.environ.get("MSRC_COCO", "data/coco"), "val2014"))
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--out_dir", default="out")
ap.add_argument("--steal", action="store_true",
                help="work-stealing mode: ignore --shard partitioning, take "
                     "any image of the split not yet claimed by another "
                     "worker (atomic lock files), skip images already in any "
                     "scores file for this backbone/split")
ap.add_argument("--worker", default=None)
ap.add_argument("--max_new_tokens", type=int, default=None,
                help="caption token budget; CEBC uses 20")
ap.add_argument("--tag", default="",
                help="appended to the backbone name in output files, "
                     "e.g. _t20")
ap.add_argument("--lean", action="store_true",
                help="score only what the MAIN policy reads (v0, sub_plain); "
                     "skip ladder rungs and the positional proposal")
ap.add_argument("--captions", default=None,
                help="jsonl of frozen captions ({file, caption}); reuse them "
                     "instead of generating, e.g. the llava-hf vanilla run "
                     "the decoding baselines were scored against")
args = ap.parse_args()


def coco_image_list(img_dir, seed=13):
    """Identical to every previous lane: calib = [:500], eval = [500:3500]."""
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    random.Random(seed).shuffle(files)
    return files


lo, hi = SPLITS[args.split]
if isinstance(lo, str):
    ims = sorted({x["image"].split("/")[-1] for x in json.load(open(lo))})
    random.Random(KSEED).shuffle(ims)
    files = ims[:hi]
else:
    files = coco_image_list(args.img_dir, args.seed)[lo:hi]
if not args.steal:
    files = files[args.shard::args.nshards]
if args.limit:
    files = files[:args.limit]

os.makedirs(args.out_dir, exist_ok=True)
tag = (f"w{args.worker}" if args.steal
       else f"s{args.shard}of{args.nshards}")
NAME = args.backbone + args.tag
out_path = os.path.join(args.out_dir, f"scores_{NAME}_{args.split}"
                        f"_{tag}.jsonl")
lock_dir = os.path.join(args.out_dir, "locks", f"{NAME}_{args.split}")
os.makedirs(lock_dir, exist_ok=True)
done = set()
import glob as _glob
for _p in ([out_path] if not args.steal else _glob.glob(os.path.join(
        args.out_dir, f"scores_{NAME}_{args.split}_*.jsonl"))):
    if not os.path.exists(_p):
        continue
    for line in open(_p):
        try:
            done.add(json.loads(line)["file"])
        except Exception:
            pass


def claim(fname):
    """Atomic per-image claim so shard and stealing workers never overlap."""
    try:
        os.close(os.open(os.path.join(lock_dir, fname),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False
print(f"[score] {args.backbone} {args.split} shard {args.shard}/"
      f"{args.nshards}: {len(files)} images, {len(done)} already done",
      flush=True)

if args.max_new_tokens:
    B.MAX_NEW_TOKENS = args.max_new_tokens
bb = B.load(args.backbone)
print(f"[score] caption max_new_tokens = {B.MAX_NEW_TOKENS}", flush=True)


_SKIP = {"a", "an", "the", "one", "some", "object", "objects", "thing",
         "in", "image", "picture", "photo", "is", "it", "this", "that",
         "there", "of", "answer", "visible", "clearly", "instead",
         "similar", "actually", "part", "i", "see", "shows", "show"}


def parse_word(text):
    """First content word of a short answer. MiniGPT-4 answers 'A person.'
    or 'The object in the image is a dog'; taking the literal first token
    returned the article. Applied identically to every backbone."""
    for tok in text.lower().split():
        w = tok.strip(".,!?;:\"'()")
        if w and w not in _SKIP:
            return singularize(w) if w.isalpha() else ""
    return ""


FROZEN = None
if args.captions:
    FROZEN = {}
    for _l in open(args.captions):
        _d = json.loads(_l)
        FROZEN[_d["file"]] = _d["caption"]
    print(f"[score] reusing {len(FROZEN)} frozen captions from "
          f"{args.captions}", flush=True)

fout = open(out_path, "a")
t0 = time.time()
n_new = 0
for fname in files:
    if fname in done or not claim(fname):
        continue
    path = os.path.join(args.img_dir, fname)
    image_id = int(fname.split("_")[-1].split(".")[0])
    passes = {"gen": 0, "fwd": 0}
    cache = {}

    def V(question):
        if question not in cache:
            passes["fwd"] += 1
            cache[question] = B.sigmoid(bb.gap(question + B.VERIF_SUFFIX,
                                               path))
        return cache[question]

    def propose(prompt, head):
        passes["gen"] += 1
        raw = bb.short(prompt, path)
        w = parse_word(raw)
        if not w or not w.isalpha() or w == head:
            return {"raw": raw, "word": None, "v": 0.0}
        art = "an" if w[0] in "aeiou" else "a"
        return {"raw": raw, "word": w,
                "v": V(f"Is there {art} {w} in the image?")}

    if FROZEN is not None and fname in FROZEN:
        caption = FROZEN[fname]
    else:
        passes["gen"] += 1
        caption = bb.caption(path)
    sents = split_sentences(caption)
    recs, prior = [], ""
    for si, s in enumerate(sents):
        for ci, c in enumerate(extract_claims(s)):
            r = {"si": si, "ci": ci, "ctype": c.ctype, "head": c.head,
                 "plural": bool(c.plural), "coco": c.head in WORD2CAT}
            if c.ctype == "object" and c.head and c.head.isalpha():
                r["v0"] = V(question_for(c, 0))
                if r["v0"] < SCORE_CUTOFF and args.lean:
                    r["sub_plain"] = propose(
                        B.SUB_PROMPT.format(head=c.head), c.head)
                elif r["v0"] < SCORE_CUTOFF:
                    r["rungs"] = [V(question_for(c, k))
                                  for k in range(1, n_rungs(c) + 1)]
                    r["sub_plain"] = propose(
                        B.SUB_PROMPT.format(head=c.head), c.head)
                    np_text = s[c.span[0]:c.span[1]] if c.span else c.head
                    r["sub_pos"] = propose(
                        reground_prompt(c.head, np_text, s,
                                        mentioned_words(s, prior)), c.head)
            recs.append(r)
        prior = (prior + " " + s).strip()

    fout.write(json.dumps({"file": fname, "image_id": image_id,
                           "caption": caption, "claims": recs,
                           "passes": passes}) + "\n")
    fout.flush()
    n_new += 1
    if n_new % 10 == 0:
        el = time.time() - t0
        left = len(files) - len(done) - n_new
        print(f"[score] {n_new} new | {el / n_new:.1f}s/img | "
              f"ETA {left * el / n_new / 3600:.1f}h", flush=True)
print(f"[score] done -> {out_path}", flush=True)
