#!/usr/bin/env python3
"""Prepare all archived CONUS profiles, with bounded-memory target expansion.

The source is the existing hourly, nighttime, VID>=6.6 archive, not every raw
scan. 2025 remains test-only. No station, k, date, ASL height or intensity feature.
"""
import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

SLOTS = [f"slot_{j:02d}" for j in range(29)]
FEATURES = SLOTS + ["gap_to_lowest_observed_m", "target_agl_m",
                    "solar_sin", "solar_cos", "doy_sin", "doy_cos"]


def upper_sum(density):
    # Identical ordered accumulation whether top padding is present or absent.
    # np.nansum can use a different reduction tree for arrays of different widths,
    # making a VID exactly at the threshold disagree with its padded version.
    total = np.zeros(len(density), dtype=np.float64)
    for j in range(density.shape[1]):
        total += np.where(np.isnan(density[:, j]), 0, density[:, j])
    return total


def sample_edges(density, ids, ground, k, cfg):
    """Same seeded uniform-without-replacement rank rule as the coastal R code."""
    n, width = density.shape
    edges = np.arange(k, width)
    valid = np.zeros((n, len(edges)), dtype=bool)
    for j, edge in enumerate(edges):
        vid = 0.1 * upper_sum(density[:, edge:])
        valid[:, j] = ((vid >= cfg["upper_vid_min"])
                       & np.isfinite(density[:, edge])
                       & np.isfinite(density[:, edge-k:edge]).all(axis=1)
                       & ((edge-k)*100 >= ground-cfg["ground_tolerance_m"]))
    count = valid.sum(axis=1)
    for draw in range(1, cfg["edges_per_k"]+1):
        remaining = valid.sum(axis=1)
        uniform = np.array([int(xxhash.xxh64_hexdigest(
            (f'{cfg["seed"]} observed-edge {pid}' + (f' {draw}' if draw > 1 else "")).encode())[:7], 16)/16**7
            for pid in ids])
        rank = np.floor(uniform*remaining).astype(int)+1
        matches = valid & (np.cumsum(valid, axis=1) == rank[:, None])
        selected = np.argmax(matches, axis=1)
        use = np.flatnonzero(remaining > 0)
        yield use, edges[selected[use]], draw, count[use]
        valid[use, selected[use]] = False


def expand(density, meta, cfg):
    assert density.shape[1] == 30
    assert np.all(np.isnan(density) | (np.isfinite(density) & (density >= 0)))
    ids = meta.profile_id.to_numpy()
    ground = meta.ground_m_asl.to_numpy()
    assert np.isfinite(ground).all()
    for k in cfg["k_values"]:
        for use, edges, draw, counts in sample_edges(density, ids, ground, k, cfg):
            if not len(use):
                continue
            raw = np.full((len(use), 29), np.nan)
            for j in range(29):
                have = edges+j < 30
                raw[have, j] = density[use[have], edges[have]+j]
            vid = 0.1*upper_sum(raw)
            assert (vid >= cfg["upper_vid_min"]).all()
            norm = raw/vid[:, None]
            assert np.allclose(np.nansum(norm, axis=1), 10, rtol=0, atol=1e-10)
            base = meta.iloc[use].reset_index(drop=True).copy()
            base["k"] = k
            base["edge_draw"] = draw
            base["observed_edge_bin"] = edges
            base["n_valid_edges"] = counts
            base["observed_vid"] = vid
            for j, slot in enumerate(SLOTS):
                base[slot] = norm[:, j]
            parts = []
            for gap in range(1, k+1):
                r = base.copy()
                target = edges-gap
                r["gap_to_lowest_observed_m"] = gap*100
                r["target_height_m"] = target*100
                r["target_agl_m"] = target*100-ground[use]
                r["label"] = density[use, target]/vid
                r["example_id"] = [f"{pid}:edge:{e}:block:{k}:target:{t}"
                                   for pid, e, t in zip(ids[use], edges, target)]
                assert (r.target_agl_m >= -cfg["ground_tolerance_m"]).all()
                assert np.isfinite(r.label).all()
                parts.append(r)
            yield pd.concat(parts, ignore_index=True)


def migration_mask(nights, cfg):
    mmdd = pd.to_datetime(nights).dt.strftime("%m-%d")
    return (((mmdd >= cfg["spring"][0]) & (mmdd <= cfg["spring"][1])) |
            ((mmdd >= cfg["fall"][0]) & (mmdd <= cfg["fall"][1]))).to_numpy()


def load_year(files, year, cfg, stations):
    columns = ["radar", "datetime", "height", "height_reference", "dens", "archive_vid",
               "night_key", "sunset0", "sunrise", "sunset1", "split",
               "source_snapshot_id", "vpi_snapshot_id"]
    table = pa.concat_tables([pq.ParquetFile(f).read(columns=columns) for f in files])
    # Verify provenance before assuming ASL for missing height-reference metadata.
    for key in ["source_snapshot_id", "vpi_snapshot_id"]:
        assert set(table[key].unique().to_pylist()) == {cfg[key]}, key
    refs = set(table["height_reference"].unique().to_pylist())
    assert refs <= {None, "sea"}, refs
    missing_ref = table["height_reference"].null_count
    table = table.filter(pa.compute.less(table["height"], 3000))
    frame = table.to_pandas()
    del table
    frame = frame.sort_values(["radar", "datetime", "height"]).reset_index(drop=True)
    assert len(frame) % 30 == 0
    density = frame.dens.to_numpy().reshape(-1, 30).copy()
    assert np.array_equal(frame.height.to_numpy(), np.tile(np.arange(30)*100, len(density)))
    meta = frame.iloc[::30].copy().reset_index(drop=True)
    assert not meta.duplicated(["radar", "datetime"]).any()
    # Check grouping has not mixed layer rows from different timestamps/stations.
    assert np.array_equal(frame.radar.to_numpy(), np.repeat(meta.radar.to_numpy(), 30))
    assert np.array_equal(frame.datetime.to_numpy(), np.repeat(meta.datetime.to_numpy(), 30))
    del frame
    # Apply migration seasons to station-night dates BEFORE edge selection,
    # upper-VID filtering, normalization, splitting validation, or expansion.
    season = migration_mask(meta.night_key, cfg)
    density, meta = density[season], meta.loc[season].reset_index(drop=True)
    assert migration_mask(meta.night_key, cfg).all()
    assert (meta.archive_vid >= cfg["source_archive_vid_min"]).all()
    assert set(meta.split) <= {"train", "valid", "test"}
    assert ((meta.split == "test") == (year == cfg["test_year"])).all()
    meta["year"] = year
    meta["ground_m_asl"] = meta.radar.map(stations["asl.min"])
    assert np.isfinite(meta.ground_m_asl).all()
    meta["profile_id"] = [hashlib.sha256(f"{r}|{t.isoformat()}".encode()).hexdigest()
                          for r, t in zip(meta.radar, meta.datetime)]
    t = meta.datetime.astype("int64").to_numpy()
    sunset = meta.sunset0.astype("int64").to_numpy()
    sunrise = meta.sunrise.astype("int64").to_numpy()
    assert ((sunset <= t) & (t < sunrise)).all()
    assert (sunrise > sunset).all()
    phase = np.pi*(t-sunset)/(sunrise-sunset)
    meta["solar_sin"], meta["solar_cos"] = np.sin(phase), np.cos(phase)
    doy = pd.to_datetime(meta.night_key).dt.dayofyear
    meta["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    meta["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    # Preserve original train/valid assignments. Divide valid by radar-night,
    # using only IDs, never labels/intensity, into stopping and reporting sets.
    valid_role = ["valid_stop" if xxhash.xxh64_intdigest(
        f'{cfg["seed"]} conus-valid {r} {pd.Timestamp(n).date()}'.encode()) % 2 == 0
        else "valid_report" for r, n in zip(meta.radar, meta.night_key)]
    meta["role"] = np.where(meta.split == "valid", valid_role, meta.split)
    keys = meta[["radar", "night_key", "role"]].drop_duplicates()
    assert not keys.duplicated(["radar", "night_key"]).any()
    keep = ["profile_id", "radar", "datetime", "night_key", "year", "split", "role",
            "ground_m_asl", "solar_sin", "solar_cos", "doy_sin", "doy_cos"]
    return density, meta[keep], missing_ref


def prepare_year(year, source, out, cfg, stations, profile_cap):
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    checkpoint = out/f"completed-{year}.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        assert saved["preparation_version"] == 2 and saved["profile_cap"] == profile_cap
        meta = pq.read_table(out/f"profiles-{year}.parquet").to_pandas()
        print(f"Reusing completed preparation {year}", flush=True)
        return (pd.read_csv(out/f"counts-{year}.csv"), saved["profiles"], saved["checksums"],
                meta[["radar", "night_key", "role"]].drop_duplicates())
    counts, profiles, checksums, nights = [], [], [], []
    files = sorted(Path(source).glob(f"year={year}/*.parquet"))
    assert files, year
    for file in files:
        checksums.append({"path": str(file), "sha256": hashlib.file_digest(file.open("rb"), "sha256").hexdigest()})
    print(f"Preparing CONUS {year}", flush=True)
    density, meta, missing_ref = load_year(files, year, cfg, stations)
    if profile_cap:
        # Select deterministically across stations for smoke checks only.
        idx = np.argsort(meta.profile_id.to_numpy())[:profile_cap]
        density, meta = density[idx], meta.iloc[idx].reset_index(drop=True)
    print(f"{year}: source profiles {len(meta):,}; radars {meta.radar.nunique()}", flush=True)
    nights.append(meta[["radar", "night_key", "role"]].drop_duplicates())
    profiles.append({"year": year, "profiles": len(meta), "radars": meta.radar.nunique(),
                     "missing_height_reference_layers": missing_ref})
    # Store one row per source profile for split and exclusion audits.
    pq.write_table(pa.Table.from_pandas(meta, preserve_index=False), out/f"profiles-{year}.parquet", compression="zstd")
    for start in range(0, len(meta), 10000):
        chunk = meta.iloc[start:start+10000].reset_index(drop=True)
        parts = list(expand(density[start:start+10000], chunk, cfg))
        if not parts:
            continue
        rows = pd.concat(parts, ignore_index=True)
        assert not rows.example_id.duplicated().any()
        for role, r in rows.groupby("role", sort=True):
            directory = out/role
            directory.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pandas(r, preserve_index=False),
                           directory/f"{year}-{start:07d}.tmp", compression="zstd", row_group_size=100000)
            (directory/f"{year}-{start:07d}.tmp").replace(directory/f"{year}-{start:07d}.parquet")
        summary = rows.groupby(["role", "radar", "k"]).agg(rows=("label", "size"),
            profiles=("profile_id", "nunique"), min_agl=("target_agl_m", "min")).reset_index()
        summary["year"] = year
        counts.append(summary)
    print(f"Finished {year}", flush=True)
    result = (pd.concat(counts, ignore_index=True), profiles[0], checksums, nights[0])
    result[0].to_csv(out/f"counts-{year}.csv", index=False)
    checkpoint.write_text(json.dumps({"year": year, "profiles": profiles[0], "checksums": checksums,
                                     "preparation_version": 2, "profile_cap": profile_cap}))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="config/conus-paired-gpu.json")
    p.add_argument("--profile-cap", type=int, default=0, help="Smoke only; per year")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    assert args.workers >= 1
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        previous = json.loads((out/"config.json").read_text())
        assert {k:v for k,v in previous.items() if k!="run_id"} == {k:v for k,v in cfg.items() if k!="run_id"}
    (out/"config.json").write_text(json.dumps(cfg, indent=2))
    stations = pd.read_csv("metadata/radarInfo.csv").set_index("radar")
    counts, profiles, checksums, nights = [], [], [], []
    with ProcessPoolExecutor(max_workers=min(args.workers,len(cfg["years"])),
            mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(prepare_year, year, args.source, out, cfg, stations,
                    args.profile_cap) for year in cfg["years"]]
        for future in futures:
            c, p, checks, n = future.result()
            counts.append(c); profiles.append(p); checksums.extend(checks); nights.append(n)
    all_nights = pd.concat(nights, ignore_index=True).drop_duplicates()
    assert not all_nights.duplicated(["radar", "night_key"]).any(), "Night crosses split boundaries"
    all_nights.to_csv(out/"night_split.csv", index=False)
    counts = pd.concat(counts, ignore_index=True)
    counts.to_csv(out/"cohort_counts.csv", index=False)
    pd.DataFrame(profiles).to_csv(out/"source_counts.csv", index=False)
    (out/"input_checksums.json").write_text(json.dumps(checksums, indent=2))
    (out/"manifest.json").write_text(json.dumps({"features": FEATURES,
        "source_profiles": sum(r["profiles"] for r in profiles),
        "smoke": bool(args.profile_cap), "rows_by_role": counts.groupby("role").rows.sum().to_dict()}, indent=2))
    (out/"STATUS").write_text("complete\n")
    print(counts.groupby(["role", "k"]).rows.sum().to_string(), flush=True)


if __name__ == "__main__":
    main()
