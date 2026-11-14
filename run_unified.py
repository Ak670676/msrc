# -*- coding: utf-8 -*-
"""
Runs the whole unified protocol on the frozen score files, per backbone:

  1. calibrate the policy on calib (files[:500])     -- same code everywhere
  2. replay vanilla / deletion / MSRC on eval (files[500:3500])
  3. paired bootstrap (images resampled jointly)
  4. every ablation = one policy knob flipped, RE-CALIBRATED, re-evaluated

Usage:
  PYTHONPATH=. python run_unified.py --backbones llava qwen25 iblip minigpt4
"""
import argparse
import json
import os
from multiprocessing import Pool

import unified as U
from msrc.chair_utils import load_gt, WORD2CAT

INSTANCES = os.path.join(os.environ.get("MSRC_COCO", "data/coco"), "annotations", "instances_val2014.json")

ABLATIONS = {
    "main": {},
    "no_semantic_gate": {"semantic": False},
    "caption_dup_scope": {"dup": "caption", "catdup": True},
    "no_dup_guards": {"catdup": False, "dup": "off"},
    "no_G1": {"g1": False},
    "positional_prompt": {"sub_prompt": "pos"},
    "open_vocab_subs": {"sub_vocab": "open"},
    "scope_all_objects": {"scope": "all"},
    "ladder_on": {"ladder": True},
    "no_precision_floor": {"floor": False},
    "strict_floor_lcb": {"floor": "lcb"},
    "no_mention_coverage": {"extras": False},
    "only_dup_guard": {"g1": False, "catdup": False, "semantic": False},
    "only_catdup_guard": {"g1": False, "dup": "off", "semantic": False},
    "only_semantic_gate": {"g1": False, "dup": "off", "catdup": False},
    "dup_sentence_scope": {"dup": "sentence"},
    "dup_sentence_k": {"dup": "sentence", "k_cand": True},
    "k_candidates": {"k_cand": True},
    "only_visual_guard": {"dup": "off", "catdup": False, "semantic": False},
    "no_guards_openvocab": {"g1": False, "dup": "off", "catdup": False,
                            "semantic": False, "sub_vocab": "open"},
    "no_guards_at_all": {"semantic": False, "catdup": False, "dup": "off",
                         "g1": False},
}
ALPHAS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

GT = None


def _gt():
    global GT
    if GT is None:
        GT = load_gt(INSTANCES)
    return GT


def lane(recs, pol, t0, ts, tl=U.INF):
    """Replay one policy. t0=None -> vanilla (caption untouched)."""
    caps, stats = {}, []
    acts = {"keep": 0, "delete": 0, "sub": 0, "ladder": 0}
    sub_good = sub_tot = 0
    gt = _gt()
    for r in recs:
        if t0 is None:
            cap, dec = r["caption"], []
        else:
            cap, dec = U.repair(r, pol, t0, ts, tl)
        caps[r["image_id"]] = cap
        stats.append(U.image_stats(r["image_id"], cap, gt))
        cats = gt.get(r["image_id"], set())
        for d in dec:
            acts[d["act"]] += 1
            if d["act"] == "sub" and d["sub"] in WORD2CAT:
                sub_tot += 1
                sub_good += WORD2CAT[d["sub"]] in cats
    return caps, stats, {"actions": acts, "n_sub_coco": sub_tot,
                         "sub_prec_gt": (sub_good / sub_tot) if sub_tot
                         else None}


def evaluate(bb, name, over, cal_recs, ev_recs, alpha=0.05):
    pol = dict(U.MAIN, **over)
    cal = U.calibrate(cal_recs, pol, _gt(), alpha=alpha)
    out = {"backbone": bb, "ablation": name, "alpha": alpha, "policy": pol,
           "calib": cal}
    if cal.get("tau_del") is None:
        return out, None
    T0, TS, TL, TD = (cal["tau0"], cal["tau_sub"], cal["tau_lad"],
                      cal["tau_del"])
    c_m, s_m, i_m = lane(ev_recs, pol, T0, TS, TL)
    c_d, s_d, i_d = lane(ev_recs, pol, T0, U.INF)        # matched deletion
    c_o, s_o, i_o = lane(ev_recs, pol, TD, U.INF)        # own-tau deletion
    out["eval"] = {"msrc": U.aggregate(s_m),
                   "deletion_matched": U.aggregate(s_d),
                   "deletion_own": U.aggregate(s_o),
                   "msrc_info": i_m, "del_matched_info": i_d,
                   "del_own_info": i_o}
    out["boot_vs_matched"] = U.paired_bootstrap(s_d, s_m)
    out["boot_vs_own"] = U.paired_bootstrap(s_o, s_m)
    return out, ({"msrc": c_m, "deletion_matched": c_d, "deletion_own": c_o},
                 s_m)


def job(spec):
    """Each job persists its own result so one failure never discards the
    others; a rerun skips jobs already on disk."""
    bb, kind, name, over, alpha = spec
    path = f"out/jobs/{bb}_{kind}_{name}_{alpha}.json"
    if os.path.exists(path):
        return json.load(open(path))
    try:
        r = _job(spec)
    except Exception as e:
        import traceback
        print(f"[unified] FAILED {spec}: {e}\n{traceback.format_exc()}",
              flush=True)
        return {"backbone": bb, "kind": kind, "ablation": name,
                "alpha": alpha, "error": repr(e), "calib": {}}
    os.makedirs("out/jobs", exist_ok=True)
    json.dump(r, open(path, "w"))
    return r


def _job(spec):
    bb, kind, name, over, alpha = spec
    cal = U.load_scores(bb, "calib")
    ev = U.load_scores(bb, "eval")
    r, extra = evaluate(bb, name, over, cal, ev, alpha)
    r["kind"], r["n_eval"], r["n_calib"] = kind, len(ev), len(cal)
    if kind == "abl" and name == "main" and extra:
        caps, s_m = extra
        caps_v, s_v, _ = lane(ev, U.MAIN, None, None)
        r["eval"]["vanilla"] = U.aggregate(s_v)
        r["boot_vanilla_to_msrc"] = U.paired_bootstrap(s_v, s_m)
        allr = cal + ev
        r["passes"] = {k: sum(x["passes"][k] for x in allr) / len(allr)
                       for k in ("gen", "fwd")}
        caps["vanilla"] = caps_v
        os.makedirs("out/captions", exist_ok=True)
        for k, v in caps.items():
            json.dump([{"image_id": i, "caption": c} for i, c in v.items()],
                      open(f"out/captions/{bb}_{k}.json", "w"), indent=1)
    print(f"[unified] {bb:9s} {kind:5s} {name:22s} alpha={alpha}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["llava", "qwen25", "iblip", "minigpt4"])
    ap.add_argument("--only_main", action="store_true")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--out", default="out/unified_results.json")
    args = ap.parse_args()
    jobs = []
    for bb in args.backbones:
        for name, over in ABLATIONS.items():
            if args.only_main and name != "main":
                continue
            jobs.append((bb, "abl", name, over, 0.05))
        if not args.only_main:
            jobs += [(bb, "alpha", "main", {}, a) for a in ALPHAS if a != 0.05]
            # same curve under the OLD precision floor, where substitution
            # still fires -> shows #subs > 0 across the alpha grid
    _gt()
    with Pool(min(args.procs, len(jobs))) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[unified] wrote {args.out}")


if __name__ == "__main__":
    main()
