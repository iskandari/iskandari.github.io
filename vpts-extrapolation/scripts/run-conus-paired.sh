#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/vpts-extrapolation
PY=/opt/vpts-venv/bin/python
S3_RUN="s3://vpts-extrapolation-863683271215/training-runs/${RUN_ID}"
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "Starting CONUS paired experiment: ${RUN_ID}"
nvidia-smi
aws s3 sync s3://vpts-extrapolation-863683271215/production/profile_sampled_archive_v1/ data/source/ --only-show-errors

# Pilot proves preprocessing, CUDA, both target transforms, and reporting.
"$PY" - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('config/conus-paired-gpu.json').read_text())
c['years']=[2013]
Path('config/conus-pilot.json').write_text(json.dumps(c))
PY
"$PY" training/prepare-conus-paired.py --source data/source --output data/pilot \
  --config config/conus-pilot.json --profile-cap 1000
"$PY" training/train-conus-paired.py --prepared data/pilot --output artifacts/pilot \
  --device cuda --rounds 20
aws s3 sync artifacts/ "${S3_RUN}/artifacts/" --only-show-errors --exclude '*.tmp.ubj'
echo "CUDA pilot completed. Preparing ALL source profiles."
"$PY" training/prepare-conus-paired.py --source data/source --output data/prepared
aws s3 sync data/prepared/ "${S3_RUN}/prepared/" --only-show-errors
echo "Full CONUS data prepared and archived. Starting both 5000-round fits."
"$PY" training/train-conus-paired.py --prepared data/prepared --output artifacts/full --device cuda
echo "Both full CONUS fits and reporting evaluations completed."
