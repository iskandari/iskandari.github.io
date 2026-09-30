import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('crossed', ROOT/'training/conus-crossed.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
CFG = json.loads((ROOT/'config/conus-crossed.json').read_text())


class CrossedTests(unittest.TestCase):
    def test_selection_is_geographic_reproducible_and_fifteen_percent(self):
        sites = pd.read_csv(ROOT/'metadata/conus-crossed-sites.csv')
        counts = pd.DataFrame({'radar': sites.radar, 'year': 2024})
        stations = pd.read_csv(ROOT/'metadata/radarInfo.csv')
        a = c.select_sites(counts, stations, CFG)
        b = c.select_sites(counts.sample(frac=1, random_state=9), stations.sample(frac=1, random_state=2), CFG)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.held_out.sum(), 21)
        self.assertEqual(len(a), 141)
        self.assertTrue(a.groupby('stratum').held_out.any().all())
        pd.testing.assert_frame_equal(a, sites)

    def test_all_crossed_cells_and_parent_validation_roles(self):
        frame = pd.DataFrame({'radar': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'year': [2024, 2024, 2024, 2025]*2,
            'night_key': ['2024-05-01']*3+['2025-05-01']+['2024-05-01']*3+['2025-05-01'],
            'role': ['train', 'valid_stop', 'valid_report', 'test']*2})
        self.assertEqual(list(c.assign_roles(frame, {'B'}, CFG)),
            ['train', 'valid_stop', 'valid_report', 'test_temporal',
             'test_spatial', 'test_spatial', 'test_spatial', 'test_spatiotemporal'])
        frame.loc[0, 'year'] = 2025
        with self.assertRaises(AssertionError):
            c.assign_roles(frame, {'B'}, CFG)

    def test_equal_site_summary_does_not_weight_busy_sites_more(self):
        stats = np.array([[1000, 0, 0, 1000, 1000, 0], [1, 0, 0, 9, 3, 0]], dtype=float)
        result = c.site_summary(stats, 1, 2000)
        self.assertAlmostEqual(result['equal_site_rmse'], np.sqrt(5))
        self.assertEqual(result['equal_site_bias'], 2)
        self.assertEqual(result['equal_site_rmse_ci_low'], 1)
        self.assertEqual(result['equal_site_rmse_ci_high'], 3)

    def test_full_prepare_fit_report_smoke_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'source'
            source.mkdir()
            base = json.loads((ROOT/'config/conus-paired-gpu.json').read_text())
            features = ['slot_00', 'gap_to_lowest_observed_m']
            sizes = {}
            for parent_role in c.FIT_ROLES+['test']:
                (source/parent_role).mkdir()
                rows = []
                year = 2025 if parent_role == 'test' else 2024
                day = {'train': 1, 'valid_stop': 2, 'valid_report': 3, 'test': 4}[parent_role]
                for radar in ['A', 'B', 'C', 'D']:
                    for k in range(1, 6):
                        for i in range(12):
                            rows.append({'radar': radar, 'year': year, 'night_key': f'{year}-05-{day:02d}',
                                'role': parent_role, 'split': parent_role, 'k': k,
                                'profile_id': f'{radar}-{year}-{day}-{i}',
                                'example_id': f'{radar}-{year}-{day}-{i}-{k}',
                                'slot_00': float(i+1), 'gap_to_lowest_observed_m': 100*k,
                                'label': float(i+1)/10, 'observed_vid': 20.})
                frame = pd.DataFrame(rows)
                sizes[parent_role] = len(frame)
                pq.write_table(pa.Table.from_pandas(frame), source/parent_role/'fixture.parquet')
            (source/'config.json').write_text(json.dumps(base))
            (source/'manifest.json').write_text(json.dumps({'smoke': False, 'features': features, 'rows_by_role': sizes}))
            (source/'STATUS').write_text('complete\n')
            sites = root/'sites.csv'
            pd.DataFrame({'radar': ['A','B','C','D'], 'held_out': [False,True,False,True]}).to_csv(sites, index=False)
            prepared = root/'prepared'
            c.prepare(source, prepared, sites, CFG)
            counts = pd.read_csv(prepared/'cohort_counts.csv')
            self.assertEqual(counts.rows.sum(), sum(sizes.values()))
            self.assertEqual(set(counts.loc[counts.role.isin(c.FIT_ROLES), 'radar']), {'A', 'C'})
            c.fit(prepared, root/'fit', CFG, 'cpu', smoke=True)
            self.assertEqual((root/'fit'/'STATUS').read_text().strip(), 'complete')
            metrics = pd.read_csv(root/'fit'/'comparison.csv')
            self.assertEqual(set(metrics.role), {'valid_report', *c.TEST_ROLES})
            self.assertTrue(np.isfinite(metrics.rmse).all())
            self.assertTrue(metrics.sites.eq(2).all())
            # Even an otherwise plausible prepared directory must not fit an unseen radar.
            file = next((prepared/'train').glob('*.parquet'))
            bad = pq.read_table(file).to_pandas()
            bad.loc[0, 'radar'] = 'B'
            pq.write_table(pa.Table.from_pandas(bad), file)
            with self.assertRaises(AssertionError):
                c.fit(prepared, root/'must-not-fit', CFG, 'cpu', smoke=True)


if __name__ == '__main__':
    unittest.main()
