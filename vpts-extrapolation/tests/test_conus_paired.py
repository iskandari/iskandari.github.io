import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("conus_prepare", ROOT/"training/prepare-conus-paired.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class ConusPreparationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/"config/conus-paired-gpu.json").read_text())

    def test_season_boundaries_and_winter_exclusion(self):
        nights = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-29", "2024-03-01",
            "2024-06-15", "2024-06-16", "2024-07-31", "2024-08-01",
            "2024-11-15", "2024-11-16", "2024-12-31"]))
        np.testing.assert_array_equal(prep.migration_mask(nights, self.cfg),
            [False, False, True, True, False, False, True, True, False, False])

    def test_all_blocks_distinct_edges_and_normalized_labels(self):
        meta = pd.DataFrame({"profile_id": ["fixture"], "ground_m_asl": [101.]})
        rows = pd.concat(list(prep.expand(np.full((1, 30), 10.), meta, self.cfg)))
        self.assertEqual(len(rows), 45)
        self.assertFalse(rows.example_id.duplicated().any())
        self.assertTrue((rows.target_agl_m >= -1).all())
        np.testing.assert_allclose(rows[prep.SLOTS].sum(axis=1), 10)
        np.testing.assert_allclose(rows.label*rows.observed_vid, 10)
        self.assertTrue(rows.groupby("k").observed_edge_bin.nunique().eq(3).all())
        for k in range(1, 6):
            self.assertEqual(set(rows.loc[rows.k == k, "gap_to_lowest_observed_m"]),
                             set(range(100, k*100+1, 100)))

    def test_labels_do_not_enter_upper_denominator(self):
        cfg = self.cfg | {"k_values": [3], "upper_vid_min": 5.0}
        meta = pd.DataFrame({"profile_id": ["fixture"], "ground_m_asl": [0.]})
        density = np.full((1, 30), np.nan)
        density[0, :4] = [10, 20, 30, 50]
        a = pd.concat(list(prep.expand(density, meta, cfg)))
        density[0, 0] = 10000
        b = pd.concat(list(prep.expand(density, meta, cfg)))
        np.testing.assert_array_equal(a[prep.SLOTS], b[prep.SLOTS])
        np.testing.assert_array_equal(a.observed_vid, [5, 5, 5])
        self.assertEqual(float(b.loc[b.target_height_m == 0, "label"].iloc[0]), 2000)

    def test_sampling_is_order_invariant(self):
        density = np.full((2,30), 10.)
        a = list(prep.sample_edges(density, ["a", "b"], np.array([0,0]), 3, self.cfg))
        b = list(prep.sample_edges(density[::-1], ["b", "a"], np.array([0,0]), 3, self.cfg))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x[1], y[1][::-1])

    def test_vid_boundary_is_identical_with_top_padding(self):
        tail = (np.arange(2700).reshape(100,27)%17+1).astype(float)
        tail = tail/tail.sum(axis=1)[:,None]*100
        padded = np.column_stack([tail, np.full((100,2),np.nan)])
        np.testing.assert_array_equal(prep.upper_sum(tail), prep.upper_sum(padded))
        density = np.column_stack([np.ones((100,3)),tail])
        meta = pd.DataFrame({"profile_id": [f"boundary-{j}" for j in range(100)],
                             "ground_m_asl": np.zeros(100)})
        cfg = self.cfg | {"k_values": [3], "upper_vid_min": 10.0}
        rows = pd.concat(list(prep.expand(density, meta, cfg)))
        self.assertTrue((rows.observed_vid >= 10).all())
        self.assertTrue((rows.observed_edge_bin == 3).all())


if __name__ == "__main__":
    unittest.main()
