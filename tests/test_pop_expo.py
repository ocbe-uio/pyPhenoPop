import unittest
import numpy as np
from pyphenopop.mixpopid import pop_expo, rate_expo


class TestPopExpo(unittest.TestCase):

    def test_pop_expo_basic(self):
        parameters = [0.1, 0.5, 1.0, 2.0]
        concentrations = np.array([0.1, 1.0, 10.0])
        timepoints = np.array([0, 24, 48])
        expected_shape = (len(concentrations), len(timepoints))

        result = pop_expo(parameters, concentrations, timepoints)

        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result >= 0), "All population counts should be non-negative")

    def test_pop_expo_zero_timepoints(self):
        parameters = [0.1, 0.5, 1.0, 2.0]
        concentrations = np.array([0.1, 1.0, 10.0])
        timepoints = np.array([0])
        expected_shape = (len(concentrations), len(timepoints))

        result = pop_expo(parameters, concentrations, timepoints)

        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result == 1), "Population counts at time zero should be 1")

    def test_pop_expo_high_concentration(self):
        parameters = [0.1, 0.5, 1.0, 2.0]
        concentrations = np.array([1000.0])
        timepoints = np.array([0, 24, 48])
        expected_shape = (len(concentrations), len(timepoints))

        result = pop_expo(parameters, concentrations, timepoints)

        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result >= 0), "All population counts should be non-negative")

    def test_pop_expo_edge_case(self):
        parameters = [0.0, 0.0, 0.0, 0.0]
        concentrations = np.array([0.0])
        timepoints = np.array([0])
        expected_shape = (len(concentrations), len(timepoints))

        result = pop_expo(parameters, concentrations, timepoints)

        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result == 1), "Population counts at time zero should be 1")

    def test_pop_expo_large_timepoints(self):
        parameters = [0.1, 0.5, 1.0, 2.0]
        concentrations = np.array([0.1, 1.0, 10.0])
        timepoints = np.array([0, 1000, 2000])
        expected_shape = (len(concentrations), len(timepoints))

        result = pop_expo(parameters, concentrations, timepoints)

        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(result >= 0), "All population counts should be non-negative")


if __name__ == '__main__':
    unittest.main()
