#!/usr/bin/env python3
"""Terminate after three hours without workload CPU/GPU activity or log progress.

The uploader is a sibling, not a descendant of the workload, so periodic S3
syncs cannot keep a hung workload alive. CloudWatch independently watches the
WorkActive metric. Missing samples are initially ignored so the new alarm
cannot mistake pre-launch history for an idle running instance.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import boto3
import psutil


def is_active(cpu_seconds, gpu_percent, log_changed):
    return cpu_seconds > 0.05 or gpu_percent > 0 or log_changed


def main():
    root = Path("/opt/vpts-extrapolation")
    instance_id = os.environ["INSTANCE_ID"]
    cw = boto3.client("cloudwatch", region_name="us-east-1")
    last_use = time.monotonic()
    previous, old_size = {}, -1
    while True:
        current, cpu = {}, 0.0
        try:
            pid = int((root/"worker.pid").read_text())
            process = psutil.Process(pid)
            for p in [process]+process.children(recursive=True):
                try:
                    v = p.cpu_times()
                    key = (p.pid, p.create_time())
                    current[key] = v.user+v.system
                    cpu += max(0, current[key]-previous.get(key, 0))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (OSError, ValueError, psutil.NoSuchProcess):
            pass
        previous = current
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
            gpu = max(float(v) for v in result.stdout.split())
        except (ValueError, subprocess.TimeoutExpired):
            gpu = 0
        log = Path("/var/log/vpts-conus-workload.log")
        size = log.stat().st_size if log.exists() else 0
        active = is_active(cpu, gpu, size != old_size)
        old_size = size
        if active:
            last_use = time.monotonic()
        idle = time.monotonic()-last_use
        state = {"instance_id": instance_id, "active": active, "idle_seconds": idle,
                 "cpu_seconds_delta": cpu, "gpu_utilization": gpu, "timestamp": time.time()}
        (root/"idle-state.json").write_text(json.dumps(state))
        try:
            cw.put_metric_data(Namespace="VPTS/Extrapolation", MetricData=[{
                "MetricName": "WorkActive", "Dimensions": [{"Name": "InstanceId", "Value": instance_id}],
                "Value": int(active), "Unit": "Count"}])
        except Exception as e:
            print(f"Metric publish failed: {e}", flush=True)
        if idle >= 10800:
            subprocess.run(["shutdown", "-h", "now"], check=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
