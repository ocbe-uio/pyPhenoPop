import unittest
import numpy as np
from pyphenopop.mixpopid import rate_expo


class TestRateExpo(unittest.TestCase):

    def test_rate_expo_basic_functionality(self):
        parameters = [0.1, 0.5, 2.0, 1.0]
        concentrations = np.array([0.1, 1.0, 10.0])
        expected_output = np.array([0.1 + np.log(0.5 + (1 - 0.5) / (1 + (0.1 / 2.0) ** 1.0)),
                                    0.1 + np.log(0.5 + (1 - 0.5) / (1 + (1.0 / 2.0) ** 1.0)),
                                    0.1 + np.log(0.5 + (1 - 0.5) / (1 + (10.0 / 2.0) ** 1.0))])
        np.testing.assert_array_almost_equal(rate_expo(parameters, concentrations), expected_output)

    def test_rate_expo_zero_concentration(self):
        parameters = [0.1, 0.5, 2.0, 1.0]
        concentrations = np.array([0.0])
        expected_output = np.array([0.1 + np.log(0.5 + (1 - 0.5) / (1 + (0.0 / 2.0) ** 1.0))])
        np.testing.assert_array_almost_equal(rate_expo(parameters, concentrations), expected_output)


if __name__ == '__main__':
    unittest.main()
