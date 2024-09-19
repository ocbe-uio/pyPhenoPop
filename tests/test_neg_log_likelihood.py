import unittest
import numpy as np
from pyphenopop.mixpopid import neg_log_likelihood


class TestNegLogLikelihood(unittest.TestCase):

    def setUp(self):
        # Setting up common variables for tests
        self.max_subpop = 2
        self.parameters = np.array([0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        measurements = np.array([[[10, 20, 30], [15, 25, 35]], [[12, 22, 32], [18, 28, 38]]])
        measurements = measurements.reshape((3, 2, 2))
        self.measurements = measurements
        self.concvec = np.array([0.1, 0.2])
        self.timevec = np.array([0, 24, 48])
        self.num_replicates = 2
        self.model = 'expo'
        self.num_timepoints_high = 1
        self.num_conc_high_noise = 1
        self.num_noise_high = 1
        self.num_noise_low = 1

    def test_neg_log_likelihood(self):
        # Test the neg_log_likelihood function with valid inputs
        result = neg_log_likelihood(self.max_subpop, self.parameters, self.measurements, self.concvec, self.timevec,
                                    self.num_replicates, self.model, self.num_timepoints_high, self.num_conc_high_noise,
                                    self.num_noise_high, self.num_noise_low)
        self.assertIsInstance(result, float)

    def test_neg_log_likelihood_invalid_model(self):
        # Test the neg_log_likelihood function with an invalid model
        with self.assertRaises(NotImplementedError):
            neg_log_likelihood(self.max_subpop, self.parameters, self.measurements, self.concvec, self.timevec,
                               self.num_replicates, 'invalid_model', self.num_timepoints_high, self.num_conc_high_noise,
                               self.num_noise_high, self.num_noise_low)

    def test_neg_log_likelihood_invalid_parameters(self):
        # Test the neg_log_likelihood function with invalid parameters length
        invalid_parameters = np.array([0.5, 0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(KeyError):
            neg_log_likelihood(self.max_subpop, invalid_parameters, self.measurements, self.concvec, self.timevec,
                               self.num_replicates, self.model, self.num_timepoints_high, self.num_conc_high_noise,
                               self.num_noise_high, self.num_noise_low)


if __name__ == '__main__':
    unittest.main()
