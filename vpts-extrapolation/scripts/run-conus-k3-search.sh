#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/vpts-extrapolation
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/opt/vpts-venv/bin/python
PARENT=s3://vpts-extrapolation-863683271215/training-runs/conus-edges-k1to5-paired-20260923-04
echo 'Downloading frozen parent rows, excluding the 2025 test entirely.'
nvidia-smi
aws s3 sync "${PARENT}/prepared/" data/frozen/ --exclude 'test/*' --only-show-errors
aws s3 sync "${PARENT}/artifacts/full/cuberoot/predictions/" data/parent-cuberoot/predictions/ --only-show-errors
"$PY" - <<'PY'
from pathlib import Path
import shutil
import pyarrow.parquet as pq
src, dst = Path('data/frozen'), Path('data/pilot')
dst.mkdir()
for name in ['config.json', 'manifest.json', 'STATUS']:
    shutil.copyfile(src/name, dst/name)
for role in ['train', 'valid_stop', 'valid_report']:
    (dst/role).mkdir()
    table = pq.read_table(sorted((src/role).glob('*.parquet'))[0]).slice(0, 5000)
    pq.write_table(table, dst/role/'pilot.parquet')
PY
"$PY" training/search-conus-k3.py --prepared data/pilot --previous unused \
  --output artifacts/pilot --device cuda --smoke
echo 'Running k3 diagnosis and eight pooled capacity/weight trials.'
"$PY" training/search-conus-k3.py --prepared data/frozen --previous data/parent-cuberoot \
  --output artifacts/search --device cuda
