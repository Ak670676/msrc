#!/bin/bash
# End-to-end MSRC pipeline for one backbone (llava | qwen25 | iblip).
#   bash run_all.sh llava            # needs 1 GPU (>= 24 GB)
# Environment: MSRC_COCO (COCO root, default ./data/coco), MSRC_MODELS (checkpoint dir, default ./models)
set -e
cd "$(dirname "$0")"
B=${1:-llava}
export PYTHONPATH=.
for SPLIT in calib eval; do               # 500 calibration images, 3,000 evaluation images
  python score.py   --backbone $B --split $SPLIT     # caption + verify + propose, written once
  python augment.py --backbone $B --split $SPLIT     # verify mentions CHAIR counts but the extractor missed
done
python run_unified.py --backbones $B --only_main     # calibrate on CPU, replay vanilla / VOF / MSRC, bootstrap
echo "results: out/unified_results.json"
