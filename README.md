# MSRC — method code

Calibrated claim-level verification and repair for LVLM image captions.
Every object claim in a caption is verified by the model itself and then
KEPT, SUBSTITUTED or DELETED (Eq. 4). Two thresholds, `(tau0, tau_sub)`, are
chosen by conformal risk control on 500 calibration images (alpha = 0.05).
VOF is the same policy with substitution switched off (`tau_sub = inf`).

The pipeline has two stages:

1. **GPU stage** (`score.py`, `augment.py`): generate each caption once and
   write, per claim, every score a repair policy could ask for (verification
   score `V`, the one proposed substitute and its `V`).
2. **CPU stage** (`unified.py`, `run_unified.py`): calibrate the thresholds, replay
   vanilla / VOF / MSRC on the stored scores, score with CHAIR, bootstrap.

## Files

| Path | Purpose |
|---|---|
| `msrc/backbones.py` | LLaVA-1.5-7B, Qwen2.5-VL-3B, InstructBLIP-7B behind one interface (caption, yes/no verification, one-word proposal) |
| `msrc/lattice.py` | claim extraction (spaCy), span edits, deletion exactly as Eq. 4 |
| `msrc/guards.py` | substitution guards: COCO vocabulary, visual noun, duplicate, semantic compatibility |
| `msrc/chair_utils.py` | CHAIR scorer and COCO ground truth (instances + reference captions) |
| `score.py` | GPU stage: caption, verify, propose |
| `augment.py` | GPU stage: verify COCO mentions the claim extractor missed (deletion only) |
| `unified.py` | policy (`MAIN`), calibration (`calibrate`), repair, evaluation |
| `run_unified.py` | CPU stage: calibrate, replay, bootstrap |
| `caption_one.py` | run vanilla / VOF / MSRC on one arbitrary image |
| `certify_coco.py`, `certify_fresh.py` | Waudby-Smith-Ramdas bound on the evaluation split / on fresh images |
| `run_all.sh` | the whole pipeline for one backbone |
| `frozen/` | the calibrated thresholds used in the paper and the pre-registration file |

Not included: model checkpoints, COCO, the baselines (decoding methods,
ConfLVLM, Woodpecker) and MiniGPT-4.

## Setup

Python 3.10+, one GPU with at least 24 GB for the scoring stage.

```bash
pip install torch transformers nltk spacy numpy scipy pillow accelerate sentencepiece
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(x) for x in ('wordnet','omw-1.4')]"
```

Versions used for the reported numbers: torch 2.11, transformers 4.45.2 for
LLaVA and InstructBLIP, transformers 4.51.3 for Qwen2.5-VL (use a separate
environment for it). LLaVA also needs the `llava` package from the official
LLaVA repository (https://github.com/haotian-liu/LLaVA).

Data and checkpoints, laid out as:

```
data/coco/val2014/*.jpg
data/coco/annotations/instances_val2014.json
data/coco/annotations/captions_val2014.json
models/llava-v1.5-7b/   models/Qwen2.5-VL-3B-Instruct/   models/instructblip-vicuna-7b/
```

or point to them elsewhere:

```bash
export MSRC_COCO=/path/to/coco        # contains val2014/ and annotations/
export MSRC_MODELS=/path/to/models
```

## Run

```bash
bash run_all.sh llava        # or qwen25, iblip
```

which is equivalent to

```bash
export PYTHONPATH=.
for SPLIT in calib eval; do
  python score.py   --backbone llava --split $SPLIT
  python augment.py --backbone llava --split $SPLIT
done
python run_unified.py --backbones llava --only_main
```

Image lists come from one seeded shuffle of `val2014` (seed 13):
`calib` = images 0-499, `eval` = 500-3,499, `fresh` = 3,500-4,999 (used only
for certification). `score.py` accepts `--shard k --nshards n` to split a
split across GPUs; run every shard, then `augment.py` and `run_unified.py`.

Outputs:

- `out/scores_<backbone>_<split>_*.jsonl` — frozen captions and scores
- `out/unified_results.json` — calibrated `(tau0, tau_sub)`, CHAIR (C_S, C_I, coverage) for vanilla, VOF and MSRC, with paired bootstrap intervals
- `out/captions/<backbone>_{vanilla,msrc,...}.json` — the final captions

`run_unified.py` caches each job in `out/jobs/`; delete that folder to
recalibrate from scratch. Drop `--only_main` to also run the ablations and the
alpha sweep.

Calibrated thresholds reported in the paper (`frozen/`):

| Backbone | tau0 | tau_sub |
|---|---|---|
| LLaVA-1.5-7B | 0.7606 | 0.8758 |
| Qwen2.5-VL-3B | 0.7782 | 0.02299 |
| InstructBLIP-7B | 0.6654 | 0.9334 |

## Caption one image

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python caption_one.py --score --backbone llava --image path/to/image.jpg
PYTHONPATH=. python caption_one.py --replay --image path/to/image.jpg
```

The first command writes the caption and all scores to `out/single/`; the
second prints the vanilla, deletion (VOF) and MSRC captions using the frozen
thresholds in `frozen/`. No recalibration happens.

## Certification

After scoring the `fresh` split (`score.py --split fresh`, `augment.py --split fresh`):

```bash
PYTHONPATH=. python certify_fresh.py     # WSR test fixed in frozen/prereg_fresh_cert.json
PYTHONPATH=. python certify_coco.py      # same bound on the calibration CERTIFY half and the evaluation split
```

## Policy settings

The headline configuration is `MAIN` in `unified.py`. Calibration: 500
images, SELECT / CERTIFY halves (seed 23), per-image mean loss over in-scope
COCO claims, bound `(n*R + 1)/(n + 1)` with margin 0.15, alpha = 0.05.
The substitution prompt, the guards and the one-candidate-per-failed-claim
rule are in `msrc/backbones.py`, `msrc/guards.py` and `unified.py`.

## Scope

The verifier and the risk are object-level and defined over the 80 COCO
categories. Attributes, relations and counts are not re-verified.
