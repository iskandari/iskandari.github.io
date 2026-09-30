import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('search', Path(__file__).parents[1]/'training/search-conus-k3.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class MetricTests(unittest.TestCase):
    def test_metric_decodes_scales_and_uses_only_k3(self):
        # k3 errors in physical density: (8-1)*2=14, (1-4)*3=-9.
        actual = s.k3_rmse(np.array([2., 1., 100.]), np.array([1., 4., 0.]),
                           np.array([2., 3., 999.]), np.array([3, 3, 1]))
        self.assertAlmostEqual(actual, np.sqrt((14**2+9**2)/2))

    def test_negative_predictions_are_clipped_consistently(self):
        self.assertEqual(s.k3_rmse(np.array([-2.]), np.array([1.]),
                                  np.array([10.]), np.array([3])), 10.)


if __name__ == '__main__':
    unittest.main()
