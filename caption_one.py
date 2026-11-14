# -*- coding: utf-8 -*-
"""
Caption ONE arbitrary image with vanilla / deletion / MSRC on a backbone,
using the unified pipeline unchanged.

GPU stage (--score, one backbone per call, run in that backbone's env):
  the per-image block of score.py (caption + v0 / rungs / substitutes for
  every object claim) followed by augment.py's uncovered-mention pass, i.e.
  exactly the record run_unified.py would replay. Written to
  out/single/<stem>_<backbone>.json.

CPU stage (--replay, any env): replays the stored MAIN-policy gates from
frozen/<backbone>_abl_main_0.05.json with unified.repair():
  vanilla   the frozen caption
  deletion  repair(rec, pol, tau_del, inf)
  MSRC      repair(rec, pol, tau0, tau_sub, tau_lad)
Nothing is recalibrated; the gates are the ones from the n=500 calibration.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python caption_one.py --score \
      --backbone llava --image path/to/image.jpg
  PYTHONPATH=. python caption_one.py --replay --image path/to/image.jpg
"""
import argparse
import json
import os

BACKBONES = ["llava", "qwen25", "iblip", "minigpt4"]
NAMES = {"llava": "LLaVA-1.5-7B", "qwen25": "Qwen2.5-VL-3B",
         "iblip": "InstructBLIP-7B", "minigpt4": "MiniGPT-4"}

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--backbone", choices=BACKBONES)
ap.add_argument("--score", action="store_true")
ap.add_argument("--replay", action="store_true")
ap.add_argument("--out_dir", default="out/single")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
stem = os.path.splitext(os.path.basename(args.image))[0]


def rec_path(bb):
    return os.path.join(args.out_dir, f"{stem}_{bb}.json")


# ------------------------------------------------------------------ GPU ----
def score(bbname):
    from msrc.chair_utils import WORD2CAT
    from msrc.lattice import (extract_claims, question_for, n_rungs,
                              split_sentences, singularize)
    from msrc.guards import reground_prompt, mentioned_words
    from msrc import backbones as B
    import augment

    _SKIP = {"a", "an", "the", "one", "some", "object", "objects", "thing",
             "in", "image", "picture", "photo", "is", "it", "this", "that",
             "there", "of", "answer", "visible", "clearly", "instead",
             "similar", "actually", "part", "i", "see", "shows", "show"}

    def parse_word(text):                      # verbatim from score.py
        for tok in text.lower().split():
            w = tok.strip(".,!?;:\"'()")
            if w and w not in _SKIP:
                return singularize(w) if w.isalpha() else ""
        return ""

    bb = B.load(bbname)
    path = args.image
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

    # ---- per-image block of score.py, unchanged ----
    SCORE_CUTOFF = 0.999
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
                if r["v0"] < SCORE_CUTOFF:
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
    rec = {"file": os.path.basename(path), "image_id": -1,
           "caption": caption, "claims": recs, "passes": passes}

    # ---- augment.py uncovered-mention pass, unchanged ----
    ex = augment.uncovered(rec)
    for e in ex:
        art = "an" if e["word"][0] in "aeiou" else "a"
        e["v0"] = V(f"Is there {art} {e['word']} in the image?")
    rec["extra"] = ex

    json.dump(rec, open(rec_path(bbname), "w"), indent=1)
    print(f"[caption_one] {bbname}: {len(recs)} claims, {len(ex)} extras, "
          f"{passes} -> {rec_path(bbname)}", flush=True)


# ------------------------------------------------------------------ CPU ----
def replay():
    import unified as U
    report = {"image": args.image, "prompt": None, "backbones": {}}
    for bb in BACKBONES:
        if not os.path.exists(rec_path(bb)):
            print(f"[replay] {bb}: not scored yet ({rec_path(bb)})")
            continue
        rec = json.load(open(rec_path(bb)))
        job = json.load(open(f"frozen/{bb}_abl_main_0.05.json"))
        pol, cal = job["policy"], job["calib"]
        t0, ts, tl, td = (cal["tau0"], cal["tau_sub"], cal["tau_lad"],
                          cal["tau_del"])
        ts = U.INF if ts is None else ts
        tl = U.INF if tl is None else tl
        cap_d, dec_d = U.repair(rec, pol, td, U.INF)
        cap_m, dec_m = U.repair(rec, pol, t0, ts, tl)
        report["backbones"][bb] = {
            "name": NAMES[bb],
            "gates": {"tau0": t0, "tau_sub": ts, "tau_lad": tl,
                      "tau_del": td},
            "vanilla": rec["caption"], "deletion": cap_d, "msrc": cap_m,
            "decisions_deletion": dec_d, "decisions_msrc": dec_m}
    from msrc import backbones as B
    report["prompt"] = B.CAPTION_PROMPT
    out = os.path.join(args.out_dir, f"{stem}_captions.json")
    json.dump(report, open(out, "w"), indent=1)

    for bb, r in report["backbones"].items():
        g = r["gates"]
        print("=" * 78)
        print(f"{r['name']}   tau0={g['tau0']:.3f}  tau_sub={g['tau_sub']:.3f}"
              f"  tau_del={g['tau_del']:.3f}")
        print("=" * 78)
        for lane in ("vanilla", "deletion", "msrc"):
            print(f"\n[{lane.upper()}]\n{r[lane]}")
        print("\nMSRC decisions (object claims below tau0):")
        acted = [d for d in r["decisions_msrc"] if d["act"] != "keep"]
        for d in acted:
            extra = " (extra mention)" if d.get("extra") else ""
            tail = (f" -> {d['sub']} (v={d['v']:.3f})" if d["act"] == "sub"
                    else f"  [{d.get('why', '')}]")
            print(f"   {d['act']:6s} {d['head']:<14s} v0={d['v0']:.3f}"
                  f"{tail}{extra}")
        if not acted:
            print("   none -- every verified claim passed tau0")
        print()
    # Plain-text copy for reading/comparing by eye.
    txt = os.path.join(args.out_dir, f"{stem}_captions.txt")
    with open(txt, "w") as f:
        f.write(f"image : {args.image}\nprompt: {report['prompt']}\n")
        for bb, r in report["backbones"].items():
            g = r["gates"]
            f.write("\n" + "=" * 78 + f"\n{r['name']}   tau0={g['tau0']:.3f}"
                    f"  tau_sub={g['tau_sub']:.3f}  tau_del={g['tau_del']:.3f}"
                    "\n" + "=" * 78 + "\n")
            for lane in ("vanilla", "deletion", "msrc"):
                f.write(f"\n[{lane.upper()}]\n{r[lane]}\n")
            acted = [d for d in r["decisions_msrc"] if d["act"] != "keep"]
            f.write("\nMSRC edits:\n")
            for d in acted:
                tail = (f" -> {d['sub']} (v={d['v']:.3f})" if d["act"] == "sub"
                        else f"  [{d.get('why', '')}]")
                f.write(f"   {d['act']:6s} {d['head']:<14s} v0={d['v0']:.3f}"
                        f"{tail}\n")
            if not acted:
                f.write("   none -- every verified claim passed tau0\n")
    missing = [b for b in BACKBONES if b not in report["backbones"]]
    print(f"[replay] -> {out}\n[replay] -> {txt}"
          + (f"\n[replay] not scored yet: {missing}" if missing else ""))


if args.score:
    score(args.backbone)
if args.replay:
    replay()
