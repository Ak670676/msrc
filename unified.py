# -*- coding: utf-8 -*-
"""
CPU half of the unified pipeline: calibrate, repair, evaluate -- identically
for every backbone, from the frozen score files written by score.py.

Every lane (vanilla, deletion, MSRC, every ablation) is a POLICY replayed on
the same frozen caption, the same claims and the same scores. Deletion is
MSRC with tau_sub = inf, so "matched deletion" is matched by construction.

Policy knobs (MAIN is the headline configuration; ablations flip one):
  scope      'coco'  verify COCO-80 object claims only       (MAIN)
             'all'   verify every visual object claim
  sub_prompt 'plain' salient-object prompt                  (MAIN)
             'pos'   positional prompt ("what is at that phrase")
  sub_vocab  'coco'  substitute must be a COCO-80 word       (MAIN)
             'open'  any visual noun (non-COCO subs are charged as errors
                     in calibration: worst case, no audit needed)
  g1         substitute must be a visual noun               (MAIN on)
  dup        'caption' | 'sentence' | 'off'  word-level duplicate guard
  catdup     block a substitute whose COCO category the caption already
             names (the denominator artifact)               (MAIN on)
  semantic   WordNet semantic-compatibility gate            (MAIN on)
  ladder     entailment ladder rungs before substitution    (MAIN off)
  floor      precision floor in tau_sub selection           (MAIN on)

Calibration (same code, same protocol, every backbone): conformal risk
control on the calib split, per-image hallucination loss over labelable
(COCO) claims, SELECT/CERTIFY halves (seed 23), alpha = 0.05, select margin
0.15, bound (n*R + 1)/(n + 1). Guards are applied during calibration exactly
as at deployment.
"""
import glob
import json
import os
import random
import re
from collections import defaultdict

from msrc.chair_utils import WORD2CAT, caption_to_categories
from msrc.lattice import (extract_claims, split_sentences, apply_repairs,
                          repair_edit_for, pluralize, is_ancestor)
from msrc.guards import (is_substitutable_head, duplicate_blocked,
                         semantic_substitute_compatible)

INF = float("inf")
MAIN = dict(scope="coco", sub_prompt="plain", sub_vocab="coco", g1=True,
            dup="sentence", catdup=False, semantic=True, ladder=False,
            floor="point", extras=True, k_cand=False)
# dup="sentence": a verified replacement may name an object the caption
# mentions in ANOTHER sentence (the repeat a reader does not notice) but not
# one already in the sentence being edited. The former caption-scope setting,
# plus the category-level guard, rejected ~96% of verified-correct repairs and
# is kept as the "caption-level duplicate scope" ablation.
# floor: "lcb"   substitute only if the 95% Clopper-Pearson LOWER bound on
#                SELECT substitute precision >= deletion's kept precision;
#                otherwise tau_sub = inf (fall back to deletion)   (MAIN)
#        "point" old rule: point estimate, silently relaxed when unmet
#        False   no floor
# extras: verify every CHAIR-countable COCO mention the claim extractor
#         missed (colour words, compound modifiers, cut-off fragments) as a
#         deletion-only claim; written by augment.py               (MAIN on)

_CLAIMS = {}


def claims_of(sentence):
    if sentence not in _CLAIMS:
        _CLAIMS[sentence] = extract_claims(sentence)
    return _CLAIMS[sentence]


def load_scores(backbone, split, out_dir="out"):
    recs = {}
    for p in sorted(glob.glob(f"{out_dir}/scores_{backbone}_{split}_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            recs[r["file"]] = r
    for kp in sorted(glob.glob(f"{out_dir}/subk_{backbone}_{split}_*.jsonl")):
        for line in open(kp):
            x = json.loads(line)
            r = recs.get(x["file"])
            if not r:
                continue
            for c in r["claims"]:
                got = x["cands"].get(f"{c.get('si')}|{c.get('ci')}")
                if got:
                    c["sub_k"] = got
    xp = f"{out_dir}/extras_{backbone}_{split}.jsonl"
    if os.path.exists(xp):
        for line in open(xp):
            x = json.loads(line)
            if x["file"] in recs:
                recs[x["file"]]["extra"] = x["extra"]
    out = [recs[k] for k in sorted(recs)]
    if split == "eval" and os.environ.get("MSRC_EVAL_N"):   # rebuttal: first-N eval subset
        out = out[:int(os.environ["MSRC_EVAL_N"])]
    return out


# ------------------------------------------------------------ candidates --
def candidate(r, pol, sentence, caption, used, used_cats):
    """Substitute candidate for a failed object claim under `pol`.
    -> (word, v) or (None, reason).

    With pol["k_cand"], the LVLM proposed several replacements for this claim
    (propose_k.py); each is screened by the SAME guards and the admissible one
    with the highest verification score is returned."""
    if pol.get("k_cand"):
        best, why = None, "no_proposal"
        for cand in r.get("sub_k") or []:
            ok, reason = _screen(cand["word"], r, pol, sentence, caption,
                                 used, used_cats)
            if ok:
                if best is None or cand["v"] > best[1]:
                    best = (cand["word"], cand["v"])
            elif why == "no_proposal" or why == "":
                why = reason
        return (best, "ok") if best else (None, why)
    s = r.get("sub_" + pol["sub_prompt"])
    if not s or not s["word"]:
        return None, "no_proposal"
    w = s["word"]
    ok, reason = _screen(w, r, pol, sentence, caption, used, used_cats)
    return ((w, s["v"]), "ok") if ok else (None, reason)


def _screen(w, r, pol, sentence, caption, used, used_cats):
    """The guard chain, shared by the single- and k-candidate paths."""
    if pol["sub_vocab"] == "coco" and w not in WORD2CAT:
        return False, "non_coco"
    if pol["g1"] and not is_substitutable_head(w):
        return False, "not_visual"
    if pol["dup"] != "off":
        blocked = duplicate_blocked(
            w, sentence, caption if pol["dup"] == "caption" else "",
            pol["dup"])
        if blocked or w in used:
            return False, "duplicate"
    if pol["catdup"] and w in WORD2CAT and (
            WORD2CAT[w] in {c for c, _ in caption_to_categories(caption)}
            or WORD2CAT[w] in used_cats):
        return False, "category_duplicate"
    if pol["semantic"]:
        try:
            ok, why, _ = semantic_substitute_compatible(r["head"], w)
        except Exception:           # malformed WordNet entry: reject
            ok, why = False, "semantic_error"
        if not ok:
            return False, why or "semantic"
    return True, "ok"


def in_scope(r, pol):
    if r["ctype"] != "object" or "v0" not in r:
        return False
    if pol["scope"] == "coco":
        return r["coco"]
    return r["coco"] or is_substitutable_head(r["head"])


# ------------------------------------------------------------------ repair --
def _merge_overlaps(edits):
    """Resolve PARTIALLY overlapping edits (apply_repairs only handles edits
    fully nested inside a deletion). Happens when a CHAIR bigram mention
    ('hot dog') overlaps a claim of another category ('dog'). Overlapping
    deletions merge into one deletion over their union; a deletion that
    overlaps a replacement is dropped (the replacement wins)."""
    reps = [e for e in edits if e[2] != ""]
    dels = sorted((a, b) for a, b, r in edits if r == "")
    merged = []
    for a, b in dels:
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    keep = [(a, b, "") for a, b in merged
            if not any(a < rb and ra < b and not (a <= ra and rb <= b)
                       for ra, rb, _ in reps)]
    return reps + keep


def repair(rec, pol, tau0, tau_sub, tau_lad=INF):
    """Post-hoc repair of one frozen caption. -> (caption, decisions)."""
    caption = rec["caption"]
    sents = split_sentences(caption)
    by_pos = {(r["si"], r["ci"]): r for r in rec["claims"]}
    out, dec, used, used_cats = [], [], set(), set()
    for si, s in enumerate(sents):
        edits, dropped = [], set()
        for ci, c in enumerate(claims_of(s)):
            r = by_pos.get((si, ci))
            if r is None or r["head"] != c.head or not in_scope(r, pol):
                continue
            d = {"head": r["head"], "coco": r["coco"], "v0": r["v0"],
                 "act": "keep", "sub": None, "v": None}
            if r["v0"] < tau0:
                if pol["ladder"] and tau_lad <= 1.0:
                    for k, vr in enumerate(r.get("rungs", []), 1):
                        if vr >= tau_lad:
                            d.update(act="ladder", sub=c.ladder[k - 1], v=vr)
                            e = repair_edit_for(c, k)
                            if e:
                                edits.append(e)
                            break
                if d["act"] == "keep":
                    cand, why = (None, "tau_sub=inf")
                    if tau_sub <= 1.0:
                        cand, why = candidate(r, pol, s, caption, used,
                                              used_cats)
                    if cand and cand[1] >= tau_sub:
                        w = cand[0]
                        d.update(act="sub", sub=w, v=cand[1])
                        edits.append((c.head_span[0], c.head_span[1],
                                      pluralize(w) if c.plural else w))
                        used.add(w)
                        if w in WORD2CAT:
                            used_cats.add(WORD2CAT[w])
                    else:
                        d.update(act="delete",
                                 why=why if not cand else "below_tau_sub",
                                 cand_v=cand[1] if cand else None)
                        e = repair_edit_for(c, -1)
                        if e:
                            edits.append(e)
                        dropped.add(c.head)
            dec.append(d)
        if pol.get("extras"):
            for e in rec.get("extra", []):
                if e["si"] != si or "v0" not in e:
                    continue
                d = {"head": e["word"], "coco": True, "v0": e["v0"],
                     "act": "keep", "sub": None, "v": None, "extra": True}
                if e["v0"] < tau0:
                    d.update(act="delete", why="extra_mention")
                    edits.append((e["start"], e["end"], ""))
                dec.append(d)
        # FIX: only a sentence we EDITED can be dropped for being gutted to
        # < 3 words. The old rule also dropped untouched short sentences
        # (e.g. the fragment a token budget cuts off), which changed captions
        # the policy never edited.
        if edits and pol.get("grammar_span"):
            # grammar-preserving deletion: remove the smallest syntactic unit
            # containing sigma (msrc/grammar_delete.py); '' drops the sentence
            from msrc.grammar_delete import apply_grammar
            s2 = apply_grammar(s, _merge_overlaps(edits))
            if s2:
                out.append(s2)
        elif edits and pol.get("strict_span"):
            # Eq. (4) exactly: remove sigma only, never drop the sentence
            s2 = apply_repairs(s, _merge_overlaps(edits), strict=True)
            if s2:
                out.append(s2)
        elif edits:
            s2 = apply_repairs(s, _merge_overlaps(edits))
            if len(s2.split()) >= 3:
                out.append(s2)
        else:
            out.append(s)
    return " ".join(out).strip(), dec


# ------------------------------------------------------------- calibration --
def claim_table(recs, pol, gt):
    """Per-image list of labelable claims with everything calibration needs.
    The candidate is computed with the deployment guards (exchangeability).
    Duplicate guards depend on earlier accepted substitutes; calibration uses
    the per-claim guard outcome against the original caption, which is the
    same approximation recal_crc_v3 made."""
    per_img = {}
    for rec in recs:
        cats = gt.get(rec["image_id"], set())
        sents = split_sentences(rec["caption"])
        rows = []
        for r in rec["claims"]:
            if not in_scope(r, pol) or not r["coco"]:
                continue
            h0 = 0 if WORD2CAT[r["head"]] in cats else 1
            cand, hs, vs, lad = None, None, None, []
            if r["v0"] < 1.0:
                cand, _ = candidate(r, pol, sents[r["si"]], rec["caption"],
                                    set(), set())
                if cand:
                    w, vs = cand
                    if w in WORD2CAT:
                        hs = 0 if WORD2CAT[w] in cats else 1
                    elif any(is_ancestor(w, g.split()[-1]) for g in cats):
                        hs = 0
                    else:
                        hs = 1          # unlabelable open-vocab: charged
                if pol["ladder"]:
                    cl = claims_of(sents[r["si"]])
                    c = cl[r["ci"]] if r["ci"] < len(cl) else None
                    if c is None or c.head != r["head"]:
                        rows.append({"v0": r["v0"], "h0": h0, "vs": vs,
                                     "hs": hs, "lad": []})
                        continue
                    for k, vr in enumerate(r.get("rungs", []), 1):
                        rung = c.ladder[k - 1]
                        ok = any(g.split()[-1] == rung or
                                 is_ancestor(rung, g.split()[-1])
                                 for g in cats)
                        lad.append((vr, 0 if ok else 1))
            rows.append({"v0": r["v0"], "h0": h0, "vs": vs, "hs": hs,
                         "lad": lad})
        if pol.get("extras"):
            for e in rec.get("extra", []):
                if "v0" in e:
                    rows.append({"v0": e["v0"],
                                 "h0": 0 if e["cat"] in cats else 1,
                                 "vs": None, "hs": None, "lad": []})
        if rows:
            per_img[rec["image_id"]] = rows
    return per_img


def _emit(c, t0, ts, tl):
    """-> (loss, kind) kind: 1 kept, 2 substituted, 3 ladder, 0 deleted."""
    if c["v0"] >= t0:
        return c["h0"], 1
    for vr, hr in c["lad"]:
        if vr >= tl:
            return hr, 3
    if c["vs"] is not None and c["vs"] >= ts:
        return c["hs"], 2
    return 0, 0


def _risk(P, imgs, t0, ts, tl=INF):
    return sum(sum(_emit(c, t0, ts, tl)[0] for c in P[i]) / len(P[i])
               for i in imgs) / max(len(imgs), 1)


def _bound(P, imgs, t0, ts, tl=INF):
    n = len(imgs)
    return (n * _risk(P, imgs, t0, ts, tl) + 1) / (n + 1)


def _yield(P, imgs, t0, ts, tl=INF):
    tot = good = 0
    for i in imgs:
        for c in P[i]:
            tot += 1
            h, k = _emit(c, t0, ts, tl)
            good += (k > 0) * (1 - h)
    return 100.0 * good / max(tot, 1)


def _prec(P, imgs, t0, ts, kind, tl=INF):
    tot = bad = 0
    for i in imgs:
        for c in P[i]:
            h, k = _emit(c, t0, ts, tl)
            if k == kind:
                tot += 1
                bad += h
    return (1 - bad / tot) if tot else None


def cp_lower(k, n, conf=0.95):
    """One-sided Clopper-Pearson lower bound on a binomial proportion."""
    if n == 0 or k == 0:
        return 0.0
    from math import comb
    def cdf_ge(p):                      # P[X >= k | p]
        return sum(comb(n, j) * p ** j * (1 - p) ** (n - j)
                   for j in range(k, n + 1))
    lo, hi = 0.0, k / n
    for _ in range(60):                 # largest p with P[X>=k|p] <= 1-conf
        mid = (lo + hi) / 2
        if cdf_ge(mid) <= 1 - conf:
            lo = mid
        else:
            hi = mid
    return lo


def _sub_counts(P, imgs, t0, ts, tl=INF):
    good = tot = 0
    for i in imgs:
        for c in P[i]:
            h, k = _emit(c, t0, ts, tl)
            if k == 2:
                tot += 1
                good += 1 - h
    return good, tot


def calibrate(recs, pol, gt, alpha=0.05, seed=23, margin=0.15):
    P = claim_table(recs, pol, gt)
    imgs = sorted(P)
    random.Random(seed).shuffle(imgs)
    SEL, CER = imgs[:len(imgs) // 2], imgs[len(imgs) // 2:]
    a_sel = alpha * (1 - margin)
    v0s = sorted({c["v0"] for i in imgs for c in P[i]})
    grid0 = sorted({v0s[int(q / 50 * (len(v0s) - 1))] for q in range(51)})
    grid_s = sorted({c["vs"] for i in imgs for c in P[i]
                     if c["vs"] is not None})
    lv = sorted({vr for i in imgs for c in P[i] for vr, _ in c["lad"]})
    # 51 quantile points, like grid0: searching every distinct rung score
    # made the ladder ablation take hours on InstructBLIP
    grid_l = sorted({lv[int(q / 50 * (len(lv) - 1))] for q in range(51)}) \
        if lv else []

    t_del = next((t for t in grid0 if _bound(P, SEL, t, INF) <= a_sel), None)
    res = {"alpha": alpha, "n_calib_images": len(imgs),
           "n_calib_claims": sum(len(P[i]) for i in imgs), "tau_del": t_del}
    if t_del is None:
        res["certified"] = False
        return res
    del_prec = _prec(P, SEL, t_del, INF, 1)
    res.update(del_bound_certify=_bound(P, CER, t_del, INF),
               del_yield=_yield(P, CER, t_del, INF), del_kept_prec=del_prec)

    fl = pol["floor"]
    fl = "lcb" if fl is True else fl
    best, floor_used = None, fl
    modes = {"lcb": ("lcb",), "point": ("point", False)}.get(fl, (False,))
    for use_floor in modes:
        for t0 in grid0:
            for tl in (grid_l + [INF]) if pol["ladder"] else [INF]:
                for ts in grid_s + [INF]:
                    if _bound(P, SEL, t0, ts, tl) > a_sel:
                        continue
                    if use_floor == "point" and ts < INF:
                        p = _prec(P, SEL, t0, ts, 2, tl)
                        if p is not None and p < del_prec:
                            continue
                    if use_floor == "lcb" and ts < INF:
                        # no evidence -> no substitution: a tau_sub with no
                        # (or too few) SELECT substitutes cannot clear the
                        # bound, so the search falls through to ts = inf
                        g, n = _sub_counts(P, SEL, t0, ts, tl)
                        if cp_lower(g, n) < del_prec:
                            continue
                    y = _yield(P, SEL, t0, ts, tl)
                    if best is None or y > best[0]:
                        best = (y, t0, ts, tl)
                    break
        if best is not None:
            floor_used = use_floor
            break
    _, T0, TS, TL = best
    res.update(tau0=T0, tau_sub=TS, tau_lad=TL, floor_met=floor_used,
               msrc_bound_certify=_bound(P, CER, T0, TS, TL),
               msrc_yield=_yield(P, CER, T0, TS, TL),
               sub_prec_select=_prec(P, SEL, T0, TS, 2, TL),
               n_sub_certify=sum(1 for i in CER for c in P[i]
                                 if _emit(c, T0, TS, TL)[1] == 2))
    res["certified"] = (res["msrc_bound_certify"] <= alpha
                        and res["del_bound_certify"] <= alpha)
    return res


# --------------------------------------------------------------- metrics --
def image_stats(image_id, cap, gt):
    cats = gt.get(image_id, set())
    ment = caption_to_categories(cap)
    uset = {c for c, _ in ment}
    return {"n_ment": len(ment),
            "n_hall": sum(1 for c, _ in ment if c not in cats),
            "cap_hall": int(any(c not in cats for c, _ in ment)),
            "n_ment_d": len(uset), "n_hall_d": len(uset - cats),
            "cov": (len(uset & cats) / len(cats)) if cats else None,
            "words": len(cap.split())}


def aggregate(stats):
    n = max(len(stats), 1)
    S = lambda k: sum(s[k] for s in stats)
    cov = [s["cov"] for s in stats if s["cov"] is not None]
    return {"CHAIRs": 100 * S("cap_hall") / n,
            "CHAIRi": 100 * S("n_hall") / max(S("n_ment"), 1),
            "CHAIRi_dedup": 100 * S("n_hall_d") / max(S("n_ment_d"), 1),
            "hall_per_img": S("n_hall") / n,
            "coverage": 100 * sum(cov) / max(len(cov), 1),
            "mentions": S("n_ment"), "words": S("words") / n,
            "n": len(stats)}


METRICS = ["CHAIRs", "CHAIRi", "CHAIRi_dedup", "hall_per_img", "coverage"]


def paired_bootstrap(sa, sb, reps=1000, seed=11):
    """Delta b - a with percentile 95% CI, images resampled jointly."""
    n = len(sa)
    rng = random.Random(seed)
    A, Bm = aggregate(sa), aggregate(sb)
    draws = defaultdict(list)
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        a = aggregate([sa[i] for i in idx])
        b = aggregate([sb[i] for i in idx])
        for m in METRICS:
            draws[m].append(b[m] - a[m])
    out = {}
    for m in METRICS:
        v = sorted(draws[m])
        out[m] = {"a": A[m], "b": Bm[m], "delta": Bm[m] - A[m],
                  "lo": v[int(.025 * reps)], "hi": v[int(.975 * reps) - 1]}
    return out
