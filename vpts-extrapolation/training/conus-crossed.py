#!/usr/bin/env python3
"""Frozen-example spatial × temporal experiment; no holdout-based tuning."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

spec = importlib.util.spec_from_file_location('search', Path(__file__).with_name('search-conus-k3.py'))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
p = s.p
FIT_ROLES = ['train', 'valid_stop', 'valid_report']
TEST_ROLES = ['test_temporal', 'test_spatial', 'test_spatiotemporal']
ROLES = FIT_ROLES + TEST_ROLES


def select_sites(counts, stations, cfg):
    """Use pre-test participation and location only, never response/error values."""
    ids = sorted(counts.loc[counts.year < cfg['test_year'], 'radar'].unique())
    sites = stations.set_index('radar').loc[ids].reset_index()
    assert sites[['lat', 'lon']].notna().all().all()
    bands = pd.cut(sites.lon, cfg['longitude_edges'], labels=False)
    assert bands.notna().all()
    sites['stratum'] = bands.astype(str) + '_' + (sites.lat >= cfg['latitude_boundary']).astype(int).astype(str)
    sites['selection_hash'] = [hashlib.sha256(f"{cfg['seed']} spatial {r}".encode()).hexdigest() for r in sites.radar]
    sites['held_out'] = False
    sizes = sites.groupby('stratum').size()
    target = int(np.floor(len(sites) * cfg['holdout_fraction'] + .5))
    quota = sizes * cfg['holdout_fraction']
    allocation = np.floor(quota).astype(int).clip(lower=1)
    assert allocation.sum() <= target, 'Too few held-out sites for the geographic strata'
    while allocation.sum() < target:
        eligible = allocation < sizes - 1
        key = (quota - allocation).where(eligible, -np.inf).idxmax()
        assert eligible[key]
        allocation[key] += 1
    for stratum, group in sites.groupby('stratum'):
        n = int(allocation[stratum])
        assert n < len(group), 'Stratum must retain training sites'
        sites.loc[group.sort_values('selection_hash').index[:n], 'held_out'] = True
    return sites[['radar', 'lat', 'lon', 'asl.min', 'stratum', 'selection_hash', 'held_out']]


def assign_roles(frame, held, cfg):
    assert set(frame.role) <= {'train', 'valid_stop', 'valid_report', 'test'}
    assert frame[['radar', 'year', 'night_key', 'role']].notna().all().all()
    assert (pd.to_datetime(frame.night_key).dt.year.to_numpy() == frame.year.to_numpy()).all()
    future = frame.year == cfg['test_year']
    assert (frame.year <= cfg['test_year']).all()
    assert (future == (frame.role == 'test')).all(), 'Parent test/year mismatch'
    unseen = frame.radar.isin(held)
    roles = frame.role.to_numpy(dtype=object).copy()
    roles[future & ~unseen] = 'test_temporal'
    roles[~future & unseen] = 'test_spatial'
    roles[future & unseen] = 'test_spatiotemporal'
    return roles


def prepare(source, out, site_file, cfg):
    assert (source/'STATUS').read_text().strip() == 'complete'
    sites = pd.read_csv(site_file)
    assert sites.held_out.dtype == bool and not sites.radar.duplicated().any()
    held = set(sites.loc[sites.held_out, 'radar'])
    out.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        (out/role).mkdir()
    sites.to_csv(out/'site_split.csv', index=False)
    base = json.loads((source/'config.json').read_text())
    assert base['test_year'] == cfg['test_year']
    manifest = json.loads((source/'manifest.json').read_text())
    assert not manifest['smoke']
    counts, input_files = [], []
    night_roles = {}
    for parent_role in FIT_ROLES + ['test']:
        files = sorted((source/parent_role).glob('*.parquet'))
        assert files, parent_role
        parent_n = 0
        for i, file in enumerate(files):
            with file.open('rb') as handle:
                checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
            input_files.append({'role': parent_role, 'file': file.name, 'sha256': checksum})
            for j, batch in enumerate(p.batches([file], None)):
                frame = batch.to_pandas()
                assert frame.role.eq(parent_role).all()
                assert set(frame.radar) <= set(sites.radar), 'Unknown radar: freeze a new site manifest'
                parent_n += len(frame)
                frame['parent_role'] = frame.role
                frame['role'] = assign_roles(frame, held, cfg)
                frame['split'] = frame.role
                for r, n, role in frame[['radar', 'night_key', 'role']].drop_duplicates().itertuples(index=False, name=None):
                    key = (r, str(n))
                    assert night_roles.setdefault(key, role) == role, 'Night crosses roles'
                for role, group in frame.groupby('role'):
                    pq.write_table(pa.Table.from_pandas(group, preserve_index=False),
                        out/role/f'{parent_role}-{i:04d}-{j:03d}.parquet', compression='zstd')
                    counts.append(group.groupby(['role', 'radar', 'year', 'k']).size().rename('rows').reset_index())
            print('Repartitioned', parent_role, file.name, flush=True)
        assert parent_n == manifest['rows_by_role'][parent_role], 'Incomplete parent download'
    counts = pd.concat(counts).groupby(['role', 'radar', 'year', 'k'], as_index=False).rows.sum()
    assert set(counts.role) == set(ROLES), 'Empty evaluation cell'
    counts.to_csv(out/'cohort_counts.csv', index=False)
    pd.DataFrame([(r, n, role) for (r, n), role in night_roles.items()],
        columns=['radar', 'night_key', 'role']).to_csv(out/'night_split.csv', index=False)
    (out/'config.json').write_text(json.dumps(base, indent=2))
    (out/'experiment.json').write_text(json.dumps(cfg, indent=2))
    (out/'input_checksums.json').write_text(json.dumps(input_files, indent=2))
    (out/'manifest.json').write_text(json.dumps({'features': manifest['features'],
        'smoke': False, 'held_out_sites': sorted(held),
        'rows_by_role': counts.groupby('role').rows.sum().to_dict()}, indent=2))
    (out/'STATUS').write_text('complete\n')
    print(counts.groupby('role').rows.sum().to_string(), flush=True)


def site_summary(stats, seed, replicates):
    """Equal-site RMSE = sqrt(mean(site MSE)); CI resamples entire radars."""
    mse = stats[:, 3] / stats[:, 0]
    bias = stats[:, 4] / stats[:, 0]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(stats), size=(replicates, len(stats)))
    boot = np.sqrt(mse[draws].mean(axis=1))
    return {'sites': len(stats), 'equal_site_rmse': float(np.sqrt(mse.mean())),
        'equal_site_bias': float(bias.mean()),
        'equal_site_rmse_ci_low': float(np.quantile(boot, .025)) if len(stats) > 1 else None,
        'equal_site_rmse_ci_high': float(np.quantile(boot, .975)) if len(stats) > 1 else None}


def report(model, files, features, cuda, out, cfg):
    out.mkdir()
    (out/'predictions').mkdir()
    totals = {}
    columns = list(dict.fromkeys(features + ['label', 'observed_vid', 'k', 'radar',
        'example_id', 'profile_id', 'night_key', 'year', 'role']))
    for number, b in enumerate(p.batches(files, columns)):
        x = p.matrix(b, features, True)
        if cuda:
            import cupy as cp
            pred = model.inplace_predict(cp.asarray(x), iteration_range=(0, model.best_iteration+1)).get()
        else:
            pred = model.inplace_predict(x, iteration_range=(0, model.best_iteration+1))
        pred = p.decode(pred, True)
        assert np.isfinite(pred).all()
        y, vid, k, radar, gap = (p.values(b, c) for c in ['label', 'observed_vid', 'k', 'radar', 'gap_to_lowest_observed_m'])
        groups = {'overall': np.ones(len(y), dtype=bool)}
        for kval in np.unique(k):
            groups[f'k{kval}'] = k == kval
            for g in np.unique(gap[k == kval]):
                groups[f'k{kval}_gap{g}'] = (k == kval) & (gap == g)
        for r in np.unique(radar):
            for group, member in groups.items():
                mask = member & (radar == r)
                if not mask.any():
                    continue
                for scale in ['normalized', 'density']:
                    factor = vid[mask] if scale == 'density' else 1
                    key = (r, group, scale)
                    totals[key] = totals.get(key, np.zeros(6)) + p.sufficient(y[mask]*factor, pred[mask]*factor)
        pq.write_table(pa.table({**{c: p.values(b, c) for c in ['example_id', 'profile_id', 'radar',
            'night_key', 'year', 'role', 'k', 'gap_to_lowest_observed_m', 'observed_vid']},
            'observed_normalized': y, 'predicted_normalized': pred,
            'observed_density': y*vid, 'predicted_density': pred*vid}),
            out/'predictions'/f'batch-{number:05d}.parquet', compression='zstd')
    pd.DataFrame([{'radar': r, 'group': g, 'scale': scale, **p.scores(v)}
        for (r, g, scale), v in totals.items()]).to_csv(out/'by_site.csv', index=False)
    rows = []
    for group, scale in sorted({(g, scale) for _, g, scale in totals}):
        stats = np.stack([v for (r, g, sc), v in sorted(totals.items()) if g == group and sc == scale])
        rows.append({'group': group, 'scale': scale, **p.scores(stats.sum(axis=0)),
            **site_summary(stats, cfg['seed'], cfg['bootstrap_replicates'])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv', index=False)
    print(out.name, metrics[(metrics.group == 'k3') & (metrics.scale == 'density')].to_string(index=False), flush=True)
    return metrics


def fit(root, out, cfg, device, smoke=False):
    assert (root/'STATUS').read_text().strip() == 'complete'
    assert json.loads((root/'experiment.json').read_text()) == cfg
    base = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    features = manifest['features']
    assert not set(features) & {'radar', 'year', 'role', 'parent_role', 'night_key', 'profile_id', 'example_id'}
    files = {r: sorted((root/r).glob('*.parquet')) for r in ROLES}
    assert all(files.values())
    # Audit fitting identifiers before constructing matrices. No evaluation labels read.
    for role in FIT_ROLES:
        for b in p.batches(files[role], ['radar', 'year', 'role']):
            assert not set(p.values(b, 'radar')) & set(manifest['held_out_sites'])
            assert (p.values(b, 'year') < cfg['test_year']).all()
            assert (p.values(b, 'role') == role).all()
    out.mkdir(parents=True, exist_ok=False)
    params = base['params'] | cfg['params'] | {'device': device, 'seed': base['seed']}
    cuda = device.startswith('cuda')
    if not cuda:
        params['nthread'] = 4
    arrays = {c: [] for c in ['label', 'observed_vid', 'k']}
    for b in p.batches(files['valid_stop'], list(arrays)):
        for c in arrays:
            arrays[c].append(p.values(b, c))
    label, vid, k = (np.concatenate(arrays[c]) for c in arrays)
    assert (k == 3).any()
    record = cfg | {'parameters': params, 'features': features, 'smoke': smoke,
        'held_out_sites': manifest['held_out_sites'], 'xgboost_version': xgb.__version__}
    (out/'run_config.json').write_text(json.dumps(record, indent=2))
    train = xgb.QuantileDMatrix(p.Iterator(files['train'], features, True, True, cuda),
        max_bin=256, nthread=params['nthread'], max_quantile_batches=8)
    valid = xgb.QuantileDMatrix(p.Iterator(files['valid_stop'], features, True, False, cuda),
        max_bin=256, nthread=params['nthread'], ref=train)
    assert valid.num_row() == len(label)
    history = {}
    start = time.time()
    rounds = 5 if smoke else cfg['nrounds']
    print('Training crossed cube-root model', train.num_row(), 'rows', flush=True)
    model = xgb.train(params, train, num_boost_round=rounds, evals=[(valid, 'valid_stop')],
        custom_metric=lambda pred, dm: ('k3_density_rmse', s.k3_rmse(pred, label, vid, k)),
        maximize=False, evals_result=history, verbose_eval=100,
        callbacks=[xgb.callback.EarlyStopping(rounds=cfg['early_stopping_rounds'],
            metric_name='k3_density_rmse', data_name='valid_stop', maximize=False), s.Checkpoint(out)])
    model.save_model(out/'model.ubj')
    (out/'history.json').write_text(json.dumps(history))
    # Freeze the model/iteration before evaluation. No tuning follows these reports.
    (out/'fit.json').write_text(json.dumps(record | {'best_rounds': model.best_iteration+1,
        'best_stop_k3_density_rmse': float(model.best_score), 'train_rows': train.num_row(),
        'stop_rows': valid.num_row(), 'seconds': time.time()-start,
        'stopped_at_cap': model.num_boosted_rounds() == rounds}, indent=2))
    del train, valid
    gc.collect()
    metrics = []
    for role in ['valid_report'] + TEST_ROLES:
        table = report(model, files[role], features, cuda, out/role, cfg)
        table.insert(0, 'role', role)
        metrics.append(table)
    pd.concat(metrics).to_csv(out/'comparison.csv', index=False)
    (out/'STATUS').write_text('complete\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['select', 'prepare', 'fit'])
    parser.add_argument('--config', default='config/conus-crossed.json')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sites', type=Path, default=Path('metadata/conus-crossed-sites.csv'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.action == 'select':
        select_sites(pd.read_csv(args.source/'cohort_counts.csv'), pd.read_csv('metadata/radarInfo.csv'), cfg).to_csv(args.output, index=False)
    elif args.action == 'prepare':
        prepare(args.source, args.output, args.sites, cfg)
    else:
        fit(args.source, args.output, cfg, args.device, args.smoke)


if __name__ == '__main__':
    main()
