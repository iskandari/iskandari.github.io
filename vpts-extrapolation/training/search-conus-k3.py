#!/usr/bin/env python3
"""Fixed-cohort capacity/weight search; select on stopping nights only."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

spec = importlib.util.spec_from_file_location('paired', Path(__file__).with_name('train-conus-paired.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def k3_rmse(pred, label, vid, k):
    mask = k == 3
    return float(np.sqrt(np.mean(((p.decode(pred[mask], True)-label[mask])*vid[mask])**2)))


class Checkpoint(xgb.callback.TrainingCallback):
    def __init__(self, out):
        self.out = out

    def after_iteration(self, model, epoch, evals_log):
        if (epoch+1) % 100 == 0:
            tmp = self.out/'checkpoint.tmp.ubj'
            model.save_model(tmp)
            os.replace(tmp, self.out/'checkpoint.ubj')
            (self.out/'progress.json').write_text(json.dumps({'round': epoch+1,
                'time': time.time(), 'k3_density_rmse': evals_log['valid_stop']['k3_density_rmse'][-1]}))
        return False


def diagnose(prepared, previous, out):
    """Join parent predictions to frozen reporting metadata, verifying exact row order."""
    frames = []
    files = sorted((prepared/'valid_report').glob('*.parquet'))
    predictions = sorted((previous/'predictions').glob('*.parquet'))
    count = 0
    for i, b in enumerate(p.batches(files, ['example_id', 'radar', 'k'])):
        q = pq.read_table(predictions[i]).to_pandas()
        assert np.array_equal(q.example_id.to_numpy(), p.values(b, 'example_id'))
        q['radar'] = p.values(b, 'radar')
        frames.append(q[q.k == 3].copy())
        count += 1
    assert count == len(predictions)
    q = pd.concat(frames, ignore_index=True)
    q['intensity_decile'] = pd.qcut(q.observed_vid, 10, duplicates='drop').astype(str)
    total_sse = np.square(q.predicted_density-q.observed_density).sum()
    rows = []
    for field in ['gap_to_lowest_observed_m', 'radar', 'intensity_decile']:
        for key, g in q.groupby(field, observed=True):
            y, pred = g.observed_density.to_numpy(), g.predicted_density.to_numpy()
            stats = p.sufficient(y, pred)
            rows.append({'dimension': field, 'group': str(key), **p.scores(stats),
                         'squared_error_share': stats[3]/total_sse})
    errors = np.sort(np.square(q.predicted_density-q.observed_density).to_numpy())[::-1]
    (out/'diagnosis.json').write_text(json.dumps({'k3_rows': len(q),
        'parent_k3_density': p.scores(p.sufficient(q.observed_density.to_numpy(), q.predicted_density.to_numpy())),
        'top_1pct_rows_sse_share': float(errors[:int(np.ceil(.01*len(q)))].sum()/total_sse),
        'intensity_bins': 'reporting-row upper VID deciles; descriptive only'}, indent=2))
    pd.DataFrame(rows).to_csv(out/'diagnosis.csv', index=False)
    print('Parent k3 diagnosis saved', flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--prepared', required=True)
    a.add_argument('--previous', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--config', default='config/conus-k3-search.json')
    a.add_argument('--device', default='cuda')
    a.add_argument('--smoke', action='store_true')
    args = a.parse_args()
    prepared, out = Path(args.prepared), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    assert (prepared/'STATUS').read_text().strip() == 'complete'
    cfg = json.loads(Path(args.config).read_text())
    base = json.loads((prepared/'config.json').read_text())
    features = json.loads((prepared/'manifest.json').read_text())['features']
    # Only these three roles are ever read. Test is not downloaded or opened.
    files = {r: sorted((prepared/r).glob('*.parquet')) for r in ['train', 'valid_stop', 'valid_report']}
    assert all(files.values())
    if not args.smoke:
        diagnose(prepared, Path(args.previous), out)
    columns = ['label', 'observed_vid', 'k']
    arrays = {f: [] for f in columns}
    for b in p.batches(files['valid_stop'], columns):
        for f in columns:
            arrays[f].append(p.values(b, f))
    label, vid, k = (np.concatenate(arrays[f]) for f in columns)
    assert set(np.unique(k)) == set(range(1, 6))
    cuda = args.device.startswith('cuda')
    if cuda:
        import cupy as cp
        assert xgb.build_info()['USE_CUDA']
        print('GPU count:', cp.cuda.runtime.getDeviceCount(), flush=True)
    rounds = 5 if args.smoke else cfg['nrounds']
    trials = cfg['trials'][:2] if args.smoke else cfg['trials']
    (out/'run_config.json').write_text(json.dumps(cfg | {'features': features,
        'base_params': base['params'], 'smoke': args.smoke, 'rounds': rounds,
        'test_evaluated': False, 'xgboost_version': xgb.__version__}, indent=2))
    results = []
    for trial in trials:
        dest = out/trial['name']
        dest.mkdir()
        params = base['params'] | trial['params'] | {'device': args.device, 'seed': base['seed']}
        print('Starting trial', trial['name'], params, flush=True)
        train = xgb.QuantileDMatrix(p.Iterator(files['train'], features, True, trial['weighted'], cuda),
            max_bin=256, nthread=params['nthread'], max_quantile_batches=8)
        valid = xgb.QuantileDMatrix(p.Iterator(files['valid_stop'], features, True, False, cuda),
            max_bin=256, nthread=params['nthread'], ref=train)
        assert valid.num_row() == len(label)
        def metric(pred, dm):
            return 'k3_density_rmse', k3_rmse(pred, label, vid, k)
        history = {}
        start = time.time()
        model = xgb.train(params, train, num_boost_round=rounds, evals=[(valid, 'valid_stop')],
            custom_metric=metric, maximize=False, evals_result=history, verbose_eval=100,
            callbacks=[xgb.callback.EarlyStopping(rounds=cfg['early_stopping_rounds'],
                metric_name='k3_density_rmse', data_name='valid_stop', maximize=False), Checkpoint(dest)])
        model.save_model(dest/'model.ubj')
        (dest/'history.json').write_text(json.dumps(history))
        fit = {'name': trial['name'], 'params': params, 'weighted': trial['weighted'],
            'best_rounds': model.best_iteration+1, 'best_stop_k3_density_rmse': float(model.best_score),
            'stopped_at_cap': model.num_boosted_rounds() == rounds, 'seconds': time.time()-start,
            'train_rows': train.num_row(), 'stop_rows': valid.num_row()}
        (dest/'fit.json').write_text(json.dumps(fit, indent=2))
        results.append(fit)
        pd.DataFrame([{k:v for k,v in r.items() if k != 'params'} for r in results]).to_csv(out/'search.csv', index=False)
        del train, valid, model
        gc.collect()
        if cuda:
            cp.get_default_memory_pool().free_all_blocks()
    winner = min(results, key=lambda r: r['best_stop_k3_density_rmse'])
    # Freeze the recommendation BEFORE computing any new reporting metrics.
    (out/'selection.json').write_text(json.dumps({'winner': winner,
        'criterion': 'minimum k3 raw-density RMSE on valid_stop', 'report_used_for_selection': False}, indent=2))
    for name in dict.fromkeys(['depth6_vid', winner['name']]):
        model = xgb.Booster()
        model.load_model(out/name/'model.ubj')
        model.set_param({'device': args.device})
        p.report(model, files['valid_report'], features, True, cuda, out/name)
        (out/name/'STATUS').write_text('complete\n')
    comparison = []
    for name in dict.fromkeys(['depth6_vid', winner['name']]):
        table = pd.read_csv(out/name/'metrics.csv')
        for kval in range(1, 6):
            row = table[(table.group == f'k{kval}') & (table.scale == 'density')].iloc[0]
            comparison.append({'trial': name, 'k': kval, 'r2_density': row.r2,
                               'rmse_density': row.rmse, 'bias_density': row.bias})
    pd.DataFrame(comparison).to_csv(out/'report_comparison.csv', index=False)
    recommended = next(r for r in comparison if r['trial'] == winner['name'] and r['k'] == 3)
    control = next(r for r in comparison if r['trial'] == 'depth6_vid' and r['k'] == 3)
    verdict = ('The selected configuration improves k=3 reporting RMSE over the new depth-6 control.'
               if recommended['rmse_density'] < control['rmse_density'] else
               'No k=3 reporting improvement over the new depth-6 control was demonstrated; do not claim the search improves generalization.')
    (out/'recommendation.md').write_text('# Hyperparameter recommendation\n\n'
        + f"Selected **{winner['name']}** using stopping validation only.\n\n"
        + '```json\n'+json.dumps(winner, indent=2)+'\n```\n\n'
        + f"k=3 reporting R²: {recommended['r2_density']:.4f}; depth-6 control: {control['r2_density']:.4f}.\n\n"
        + verdict+' See report_comparison.csv for every k and density bias.\n\n'
        + 'The 2025 test remains untouched. Selection does not guarantee reporting R² ≥0.8.\n')
    (out/'STATUS').write_text('complete\n')
    print('Search, frozen selection, and reporting complete:', winner['name'], flush=True)


if __name__ == '__main__':
    main()
