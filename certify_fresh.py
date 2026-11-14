# -*- coding: utf-8 -*-
"""Pre-registered certification on images never scored before
(frozen/prereg_fresh_cert.json, written 2026-11-11 20:59 IST, before any of
these images was scored).

  images  seed-13 shuffled val2014 list, indices [3500, 5000)
  policy  MAIN, lambda-hat frozen from the September calibration
  test    certify iff the WSR betting UCB at delta = 0.05 is <= alpha = 0.05
          (same wsr() as certify_coco.py, same fixed shuffle seed 7)
  other   Hoeffding and empirical Bernstein printed for information only
"""
import os
import hashlib
import json
import random

import unified as U
from msrc.chair_utils import load_gt

PRE = "frozen/prereg_fresh_cert.json"
raw = open(PRE).read()
pre = json.loads(raw)
print(f"[prereg] {PRE} sha256 {hashlib.sha256(raw.encode()).hexdigest()} "
      f"created {pre['created']}")

# the bound functions, verbatim from certify_coco.py (which runs on import)
src = open("certify_coco.py").read()
ns = {}
exec(src.split("def losses")[0].split('"""', 2)[2], ns)
hoeffding, emp_bernstein, wsr = ns["hoeffding"], ns["emp_bernstein"], ns["wsr"]

gt = load_gt(os.path.join(os.environ.get("MSRC_COCO", "data/coco"), "annotations", "instances_val2014.json"))
cfg = {(r["backbone"], r["ablation"]): r
       for r in json.load(open("out/unified_results.json")) if r["kind"] == "abl"}
A, D = pre["alpha"], pre["delta"]
out = {}
print(f"{'backbone':8s} {'lane':5s} {'n':>5s} {'Rhat':>7s} | {'WSR':>7s} "
      f"{'decision':>9s} | {'Hoeff':>7s} {'EB':>7s}  (info only)")
for b in ("llava", "qwen25", "iblip"):
    c = cfg[(b, "main")]
    lam = pre["lambda"][b]
    assert abs(lam["tau0"] - c["calib"]["tau0"]) < 1e-12
    pol = c["policy"]
    recs = U.load_scores(b, "fresh")
    assert len(recs) == 1500, len(recs)
    P = U.claim_table(recs, pol, gt)
    for lane, ts in (("MSRC", lam["tau_sub"]), ("VOF", U.INF)):
        x = [sum(U._emit(cl, lam["tau0"], ts, U.INF)[0] for cl in P[i]) / len(P[i])
             for i in sorted(P) if P[i]]
        x2 = x[:]
        random.Random(7).shuffle(x2)
        n = len(x)
        r = sum(x) / n
        W = wsr(x2, D)
        H, E = hoeffding(x, D), emp_bernstein(x, D)
        ok = W <= A
        print(f"{b:8s} {lane:5s} {n:5d} {r:7.4f} | {W:7.4f} "
              f"{'CERTIFIED' if ok else 'not cert.':>9s} | {H:7.4f} {E:7.4f}")
        out[f"{b}|{lane}"] = dict(n=n, rhat=r, wsr=W, certified=ok,
                                  hoeffding=H, emp_bernstein=E)
json.dump(out, open("out/certify_fresh.json", "w"), indent=1)
print("-> out/certify_fresh.json")
