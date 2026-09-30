#!/usr/bin/env bash
# Prepend export RUN_ID=... in the launch payload. No credentials in user-data.
set -Eeuo pipefail
: "${RUN_ID:?}"
export RUN_ID AWS_DEFAULT_REGION=us-east-1
ROOT=/opt/vpts-extrapolation
S3_RUN="s3://vpts-extrapolation-863683271215/training-runs/${RUN_ID}"
mkdir -p "$ROOT/artifacts"
exec > >(tee -a /var/log/vpts-conus-bootstrap.log) 2>&1
finish() {
  status=$?
  trap - EXIT
  set +e
  [[ -n "${UPLOADER_PID:-}" ]] && kill "$UPLOADER_PID"
  timeout 600 aws s3 sync "$ROOT/artifacts/" "${S3_RUN}/artifacts/" --only-show-errors --exclude '*.tmp.ubj'
  if [[ -d "$ROOT/data/prepared" ]]; then
    timeout 600 aws s3 sync "$ROOT/data/prepared/" "${S3_RUN}/prepared/" --only-show-errors --exclude '*.tmp'
  fi
  for log in /var/log/vpts-conus*.log; do
    timeout 60 aws s3 cp "$log" "${S3_RUN}/logs/$(basename "$log")" --only-show-errors
  done
  marker=_FAILED
  [[ "$status" -eq 0 ]] && marker=_SUCCESS
  printf '{"exit_code":%d}\n' "$status" | timeout 60 aws s3 cp - "${S3_RUN}/${marker}" --only-show-errors
  shutdown -h now
  exit "$status"
}
trap finish EXIT
# Additional finite-runtime guard; active work is allowed beyond the idle limit.
shutdown -h +1440
apt-get update -qq
apt-get install -y --no-install-recommends python3-venv unzip curl
if ! command -v aws >/dev/null; then
  curl -fsSLo /tmp/aws.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
  unzip -q /tmp/aws.zip -d /tmp/awscli
  /tmp/awscli/aws/install
fi
aws s3 cp "${S3_RUN}/source/bundle.tgz" /tmp/conus-source.tgz --only-show-errors
tar -xzf /tmp/conus-source.tgz -C "$ROOT"
python3 -m venv /opt/vpts-venv
/opt/vpts-venv/bin/pip install --disable-pip-version-check \
  'numpy>=2,<3' 'pandas>=2.2,<4' 'pyarrow>=20,<26' 'xxhash>=3,<5' \
  'xgboost==3.2.0' 'cupy-cuda12x>=13,<15' boto3 psutil
/opt/vpts-venv/bin/pip freeze > "$ROOT/artifacts/requirements-resolved.txt"
TOKEN=$(curl -fsS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' http://169.254.169.254/latest/api/token)
export INSTANCE_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
unset TOKEN
cd "$ROOT"
# This watchdog is outside the workload's descendant tree.
/opt/vpts-venv/bin/python scripts/conus-idle-watchdog.py > /var/log/vpts-conus-idle.log 2>&1 &
WATCHDOG_PID=$!
(
  while true; do
    sleep 60
    aws s3 sync "$ROOT/artifacts/" "${S3_RUN}/artifacts/" --only-show-errors --exclude '*.tmp.ubj' || true
    if [[ -d "$ROOT/data/prepared" ]]; then
      aws s3 sync "$ROOT/data/prepared/" "${S3_RUN}/prepared/" --only-show-errors --exclude '*.tmp' || true
    fi
    aws s3 cp /var/log/vpts-conus-workload.log "${S3_RUN}/logs/workload-live.log" --only-show-errors || true
    aws s3 cp "$ROOT/idle-state.json" "${S3_RUN}/idle-state.json" --only-show-errors || true
  done
) &
UPLOADER_PID=$!
bash scripts/run-conus-paired.sh > /var/log/vpts-conus-workload.log 2>&1 &
WORKER_PID=$!
echo "$WORKER_PID" > "$ROOT/worker.pid"
wait "$WORKER_PID"
