import unittest
from pyphenopop.mixpopid import get_optimization_bounds


class TestGetOptimizationBounds(unittest.TestCase):

    def test_single_subpopulation(self):
        num_subpop = 1
        bounds_model = {
            'alpha': (0.1, 1.0),
            'b': (0.1, 1.0),
            'E': (0.1, 1.0),
            'n': (0.1, 1.0)
        }
        bounds_sigma_low = (1e-05, 5000.0)
        bounds_sigma_high = (1e-05, 10000.0)

        bnds, lb, ub = get_optimization_bounds(num_subpop, bounds_model, bounds_sigma_low, bounds_sigma_high)

        expected_bnds = (
            (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0),
            (1e-05, 10000.0), (1e-05, 5000.0)
        )
        expected_lb = [0.1, 0.1, 0.1, 0.1, 1e-05, 1e-05]
        expected_ub = [1.0, 1.0, 1.0, 1.0, 10000.0, 5000.0]

        self.assertEqual(bnds, expected_bnds)
        self.assertEqual(lb, expected_lb)
        self.assertEqual(ub, expected_ub)

    def test_multiple_subpopulations(self):
        num_subpop = 3
        bounds_model = {
            'alpha': (0.1, 1.0),
            'b': (0.1, 1.0),
            'E': (0.1, 1.0),
            'n': (0.1, 1.0)
        }
        bounds_sigma_low = (1e-05, 5000.0)
        bounds_sigma_high = (1e-05, 10000.0)

        bnds, lb, ub = get_optimization_bounds(num_subpop, bounds_model, bounds_sigma_low, bounds_sigma_high)

        expected_bnds = (
            (0.0, 0.5), (0.0, 0.5),
            (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0),
            (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0),
            (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.1, 1.0),
            (1e-05, 10000.0), (1e-05, 5000.0)
        )
        expected_lb = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1e-05, 1e-05]
        expected_ub = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000.0, 5000.0]

        self.assertEqual(bnds, expected_bnds)
        self.assertEqual(lb, expected_lb)
        self.assertEqual(ub, expected_ub)


if __name__ == '__main__':
    unittest.main()
