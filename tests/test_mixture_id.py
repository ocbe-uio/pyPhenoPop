import unittest
import numpy as np
import pandas as pd
from pyphenopop.mixpopid import mixture_id
import os


class TestMixtureId(unittest.TestCase):

    def setUp(self):
        # Create a temporary CSV file with mock data
        self.data_file = 'test_data.csv'
        data = np.random.rand(10*2, 3)  # 10 timepoints, 3 concentrations
        pd.DataFrame(data).to_csv(self.data_file, header=False, index=False)

        self.max_subpop = 3
        self.timepoints = np.linspace(0, 48, 10)  # 10 timepoints from 0 to 48 hours
        self.concentrations = np.linspace(0.01, 10, 3)  # 3 concentrations from 0.01 to 10
        self.num_replicates = 2
        self.model = 'expo'
        self.bounds_model = {'alpha': (0.0, 0.1), 'b': (0.0, 1.0), 'E': (1e-06, 15), 'n': (0.01, 10)}
        self.bounds_sigma_high = (1e-05, 10000.0)
        self.bounds_sigma_low = (1e-05, 5000.0)
        self.optimizer_options = {'method': 'L-BFGS-B', 'options': {'disp': False, 'ftol': 1e-12}}
        self.num_optim = 5
        self.selection_method = 'BIC'

    def test_mixture_id(self):
        results = mixture_id(
            self.max_subpop,
            self.data_file,
            self.timepoints,
            self.concentrations,
            self.num_replicates,
            self.model,
            self.bounds_model,
            self.bounds_sigma_high,
            self.bounds_sigma_low,
            self.optimizer_options,
            self.num_optim,
            self.selection_method
        )

        self.assertIn('summary', results)
        self.assertIn('estimated_num_populations', results['summary'])
        self.assertIn('final_neg_log_likelihood', results['summary'])
        self.assertIn('final_parameters', results['summary'])

    def tearDown(self):
        os.remove(self.data_file)


if __name__ == '__main__':
    unittest.main()
