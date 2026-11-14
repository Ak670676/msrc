# -*- coding: utf-8 -*-
"""Numerical certification table for the COCO policy (dbeQ W4 / GSTT W1).

lambda-hat = (tau0, tau_sub) was selected on the SELECT half of the 500
calibration images only. Two held-out sets never used for selection:
  cert   the CERTIFY half of the calibration images (the paper's split)
  eval   the 3,000 evaluation images (disjoint from all calibration images)
One value per IMAGE: the mean charged claim loss, exactly the quantity the
reported risk averages. Single pre-specified lambda-hat => one test, no
multiplicity correction. Bounds: Hoeffding, empirical Bernstein (Maurer-
Pontil), Waudby-Smith-Ramdas betting (corrected 1/(1-m) positivity cap).
"""
import os
import json, math, random
import unified as U
from msrc.chair_utils import load_gt


def hoeffding(x, d):
    n = len(x); return min(1.0, sum(x)/n + math.sqrt(math.log(1/d)/(2*n)))


def emp_bernstein(x, d):
    n = len(x); m = sum(x)/n
    v = sum((a-m)**2 for a in x)/(n-1); t = math.log(2/d)
    return min(1.0, m + math.sqrt(2*v*t/n) + 7*t/(3*(n-1)))


def wsr(x, d, c=0.5):
    n = len(x); rhat = sum(x)/n; thr = 1/d

    def rejected(m):
        if m >= 1: return False
        cap, pref, sq, mu, var = 1.0, 0.0, 0.0, 0.5, 0.25
        for k, xi in enumerate(x, 1):
            lam = math.sqrt(2*math.log(1/d)/(var*k*math.log(1+k)))
            lam = min(lam, c/(1-m))
            term = 1 + lam*(m-xi)
            if term <= 0: return False
            cap *= term
            if cap > thr: return True
            pref += xi; mu = (0.5+pref)/(1+k)
            sq += (xi-mu)**2; var = (0.25+sq)/(1+k)
        return False
    if not rejected(1-1e-12): return 1.0
    lo, hi = rhat, 1-1e-12
    if rejected(lo): return rhat
    for _ in range(60):
        mid = (lo+hi)/2
        if rejected(mid): hi = mid
        else: lo = mid
    return max(lo, rhat)


def losses(P, ids, t0, ts, tl):
    return [sum(U._emit(c, t0, ts, tl)[0] for c in P[i])/len(P[i]) for i in ids if P[i]]


gt = load_gt(os.path.join(os.environ.get("MSRC_COCO", "data/coco"), "annotations", "instances_val2014.json"))
cfg = {(r["backbone"], r["ablation"]): r for r in json.load(open("out/unified_results.json")) if r["kind"] == "abl"}
ALPHA = 0.05
out = {}
print(f"COCO certification of the selected policy, alpha={ALPHA}, image = sampling unit")
print(f"{'backbone':8s} {'lane':5s} {'set':5s} {'n':>5s} {'Rhat':>7s} {'plug-in':>8s} | "
      f"{'delta':>5s} {'Hoeff':>7s} {'EB':>7s} {'WSR':>7s} {'pass?':>6s}")
for b in ("llava", "qwen25", "iblip"):
    c = cfg[(b, "main")]; pol, cal = c["policy"], c["calib"]
    T0, TS, TL = cal["tau0"], cal["tau_sub"], cal.get("tau_lad", U.INF)
    Pc = U.claim_table(U.load_scores(b, "calib"), pol, gt)
    imgs = sorted(Pc); random.Random(23).shuffle(imgs)
    CER = imgs[len(imgs)//2:]
    Pe = U.claim_table(U.load_scores(b, "eval"), pol, gt)
    for lane, ts in (("MSRC", TS), ("VOF", U.INF)):
        for name, P, ids in (("cert", Pc, CER), ("eval", Pe, sorted(Pe))):
            x = losses(P, ids, T0, ts, TL)
            rng = random.Random(7); x2 = x[:]; rng.shuffle(x2)   # WSR is order-dependent: fixed random order
            n = len(x); r = sum(x)/n; plug = (n*r+1)/(n+1)
            for d in (0.10, 0.05):
                H, E, W = hoeffding(x, d), emp_bernstein(x, d), wsr(x2, d)
                best = min(H, E, W)
                print(f"{b:8s} {lane:5s} {name:5s} {n:5d} {r:7.4f} {plug:8.4f} | {d:5.2f} "
                      f"{H:7.4f} {E:7.4f} {W:7.4f} {'PASS' if best <= ALPHA else 'fail':>6s}")
                out[f"{b}|{lane}|{name}|{d}"] = dict(n=n, rhat=r, plugin=plug, hoeffding=H, emp_bernstein=E, wsr=W, certified=best <= ALPHA)
json.dump(out, open("out/certify_coco.json", "w"), indent=1)
print("[cert] -> out/certify_coco.json")
