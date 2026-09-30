#!/usr/bin/env python3
"""Matched pooled cube-root/linear fits using GPU quantile matrices from batches."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb


def batches(files, columns):
    for file in files:
        yield from pq.ParquetFile(file).iter_batches(batch_size=100000, columns=columns)


def values(batch, name):
    return batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)


def matrix(batch, features, cube):
    x = np.column_stack([values(batch, f) for f in features]).astype(np.float32)
    if cube:
        mask = [f.startswith("slot_") for f in features]
        x[:, mask] = np.cbrt(x[:, mask])
    return x


def decode(pred, cube):
    pred = np.asarray(pred, dtype=np.float64)
    return np.maximum(0, pred**3 if cube else pred)


class Iterator(xgb.DataIter):
    def __init__(self, files, features, cube, weighted, cuda):
        self.files, self.features, self.cube = files, features, cube
        self.weighted, self.cuda = weighted, cuda
        self.columns = features+["label", "observed_vid"]
        super().__init__(release_data=True)
        self.reset()

    def reset(self):
        self.iterator = iter(batches(self.files, self.columns))

    def next(self, input_data):
        try:
            b = next(self.iterator)
        except StopIteration:
            return False
        x = matrix(b, self.features, self.cube)
        y = values(b, "label").astype(np.float32)
        if self.cube:
            y = np.cbrt(y)
        w = values(b, "observed_vid").astype(np.float32) if self.weighted else None
        if self.cuda:
            import cupy as cp
            x, y = cp.asarray(x), cp.asarray(y)
            if w is not None:
                w = cp.asarray(w)
        input_data(data=x, label=y, weight=w, feature_names=self.features)
        return True


class Checkpoint(xgb.callback.TrainingCallback):
    def __init__(self, out):
        self.out = out

    def after_iteration(self, model, epoch, evals_log):
        if (epoch+1) % 100 == 0:
            tmp = self.out/"checkpoint.tmp.ubj"
            model.save_model(tmp)
            os.replace(tmp, self.out/"checkpoint.ubj")
            (self.out/"progress.json").write_text(json.dumps({"round": epoch+1, "time": time.time(),
                "stopping_rmse": evals_log["valid_stop"]["macro_fraction_rmse"][-1]}))
        return False


def sufficient(y, p):
    error = p-y
    return np.array([len(y), y.sum(), np.dot(y, y), np.dot(error, error),
                     error.sum(), (p<y).sum()], dtype=np.float64)


def scores(stats):
    n, sy, sy2, se2, se, under = stats
    sst = sy2-sy*sy/n
    return {"rows": int(n), "r2": 1-se2/sst if sst > 0 else None,
            "rmse": float(np.sqrt(se2/n)), "bias": float(se/n),
            "underprediction_fraction": float(under/n)}


def report(model, files, features, cube, cuda, out):
    totals = {}
    columns = features+["label", "observed_vid", "k", "radar", "example_id"]
    columns = list(dict.fromkeys(columns))
    prediction_dir = out/"predictions"
    prediction_dir.mkdir()
    for number, b in enumerate(batches(files, columns)):
        x = matrix(b, features, cube)
        if cuda:
            import cupy as cp
            p = model.inplace_predict(cp.asarray(x), iteration_range=(0, model.best_iteration+1)).get()
        else:
            p = model.inplace_predict(x, iteration_range=(0, model.best_iteration+1))
        p = decode(p, cube)
        y, vid, k = values(b, "label"), values(b, "observed_vid"), values(b, "k")
        gap, radar = values(b, "gap_to_lowest_observed_m"), values(b, "radar")
        assert np.isfinite(p).all()
        groups = {"overall": np.ones(len(y), dtype=bool)}
        for kval in np.unique(k):
            groups[f"k{kval}"] = k == kval
            for g in np.unique(gap[k==kval]):
                groups[f"k{kval}_gap{g}"] = (k==kval)&(gap==g)
        for r in np.unique(radar):
            groups[f"radar_{r}"] = radar==r
        for group, mask in groups.items():
            for scale in ["normalized", "density"]:
                s = vid[mask] if scale == "density" else 1
                key = (group, scale)
                totals[key] = totals.get(key, np.zeros(6))+sufficient(y[mask]*s, p[mask]*s)
        pq.write_table(pa.table({"example_id": values(b, "example_id"), "k": k,
            "gap_to_lowest_observed_m": gap, "observed_vid": vid,
            "observed_normalized": y, "predicted_normalized": p,
            "observed_density": y*vid, "predicted_density": p*vid}),
            prediction_dir/f"batch-{number:05d}.parquet", compression="zstd")
    metrics = pd.DataFrame([{"group": g, "scale": s, **scores(v)} for (g, s), v in totals.items()])
    metrics.to_csv(out/"metrics.csv", index=False)
    print(metrics[metrics.group.isin(["overall"]+[f"k{k}" for k in range(1,6)])].to_string(index=False), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--prepared", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--device", default="cuda")
    a.add_argument("--rounds", type=int)
    a.add_argument("--max-files", type=int, default=0, help="Pilot only")
    args = a.parse_args()
    prepared, out = Path(args.prepared), Path(args.output)
    assert (prepared/"STATUS").read_text().strip() == "complete"
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((prepared/"config.json").read_text())
    manifest = json.loads((prepared/"manifest.json").read_text())
    features = manifest["features"]
    files = {role: sorted((prepared/role).glob("*.parquet"))
             for role in ["train", "valid_stop", "valid_report"]}
    if args.max_files:
        files = {r: f[:args.max_files] for r, f in files.items()}
    assert all(files.values())
    # Test files are deliberately never opened during model comparison.
    label, kval = [], []
    for b in batches(files["valid_stop"], ["label", "k"]):
        label.append(values(b, "label")); kval.append(values(b, "k"))
    label, kval = np.concatenate(label), np.concatenate(kval)
    group_n = np.bincount(kval, minlength=6)[1:6]
    assert (group_n > 0).all()
    params = cfg["params"] | {"device": args.device, "seed": cfg["seed"]}
    rounds = args.rounds or cfg["nrounds"]
    cuda = args.device.startswith("cuda")
    if cuda:
        import cupy as cp
        assert xgb.build_info()["USE_CUDA"]
        print(f"GPU count: {cp.cuda.runtime.getDeviceCount()}", flush=True)
    (out/"run_config.json").write_text(json.dumps(cfg | {"actual_params": params, "rounds": rounds,
        "features": features, "xgboost_version": xgb.__version__, "pilot": bool(args.max_files),
        "test_evaluated": False}, indent=2))
    for arm in cfg["variants"]:
        cube = arm == "cuberoot"
        arm_out = out/arm
        arm_out.mkdir()
        print(f"Starting {arm}: construct train quantile matrix", flush=True)
        train = xgb.QuantileDMatrix(Iterator(files["train"], features, cube, True, cuda),
            max_bin=256, nthread=params["nthread"], max_quantile_batches=8)
        valid = xgb.QuantileDMatrix(Iterator(files["valid_stop"], features, cube, False, cuda),
            max_bin=256, nthread=params["nthread"], ref=train)
        assert valid.num_row() == len(label)

        def metric(prediction, dmatrix):
            error = decode(prediction, cube)-label
            mse = np.bincount(kval, weights=error**2, minlength=6)[1:6]/group_n
            return "macro_fraction_rmse", float(np.sqrt(mse).mean())

        history = {}
        start = time.time()
        model = xgb.train(params, train, num_boost_round=rounds, evals=[(valid, "valid_stop")],
            custom_metric=metric, maximize=False, evals_result=history, verbose_eval=50,
            callbacks=[xgb.callback.EarlyStopping(rounds=cfg["early_stopping_rounds"],
                metric_name="macro_fraction_rmse", data_name="valid_stop", maximize=False), Checkpoint(arm_out)])
        model.save_model(arm_out/"model.ubj")
        (arm_out/"history.json").write_text(json.dumps(history))
        (arm_out/"fit.json").write_text(json.dumps({"best_rounds": model.best_iteration+1,
            "best_score": model.best_score, "stopped_at_cap": model.num_boosted_rounds()==rounds,
            "train_rows": train.num_row(), "stop_rows": valid.num_row(),
            "seconds": time.time()-start}, indent=2))
        # Free training matrices before streaming reporting predictions.
        del train, valid
        gc.collect()
        report(model, files["valid_report"], features, cube, cuda, arm_out)
        (arm_out/"STATUS").write_text("complete\n")
        del model
        gc.collect()
        if cuda:
            cp.get_default_memory_pool().free_all_blocks()
    (out/"STATUS").write_text("complete\n")


if __name__ == "__main__":
    main()
