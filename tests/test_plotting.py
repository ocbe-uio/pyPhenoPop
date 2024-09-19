import unittest
from unittest.mock import patch
from pyphenopop.plotting import plot_neg_llh, plot_bic, plot_aic, plot_gr50, plot_gr50_subplot

import matplotlib.pyplot as plt


class TestPlotting(unittest.TestCase):

    def setUp(self):
        self.results_dict = {
            'summary': {
                'estimated_num_populations': 3,
                'final_parameters': [0.3, 0.5, 0.2]
            },
            '3_subpopulations': {
                'gr50': [0.1, 1.0, 10.0]
            }
        }
        self.results_list = [self.results_dict, self.results_dict]
        self.concentrations = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.subpopulation_indices = [3, 3]

    @patch('pyphenopop.plotting.plt.show')
    def test_plot_neg_llh(self, mock_show):
        # Mock data
        results = {
            '1_subpopulations': {'fval': [10, 12, 11, 13, 14]},
            '2_subpopulations': {'fval': [8, 9, 7, 10, 6]},
            '3_subpopulations': {'fval': [6, 5, 7, 8, 4]},
            '4_subpopulations': {'fval': [4, 3, 5, 2, 1]},
            'summary': {'estimated_num_populations': 4,
                        'final_neg_llh': 1,
                        'best_optimization_idx': 4}
        }

        # Call the function
        fig = plot_neg_llh(results)

        # Check if the figure is created
        self.assertIsInstance(fig, plt.Figure)

        # Check if the plot has the correct number of points
        ax = fig.gca()
        self.assertEqual(len(ax.lines[0].get_xdata()), len(results) - 1)
        self.assertEqual(len(ax.lines[0].get_ydata()), len(results) - 1)

        # Check if the y-data matches the expected negative log-likelihood values
        expected_nllhs = [10, 6, 4, 1]
        self.assertListEqual(list(ax.lines[0].get_ydata()), expected_nllhs)

        # Check labels
        self.assertEqual(ax.get_ylabel(), 'Negative log-likelihood')
        self.assertEqual(ax.get_xlabel(), 'Number of inferred populations')

    @patch('pyphenopop.plotting.plt.show')
    def test_plot_bic(self, mock_show):
        # Mock data
        results = {
            '1_subpopulations': {'BIC': 100},
            '2_subpopulations': {'BIC': 80},
            '3_subpopulations': {'BIC': 60},
            '4_subpopulations': {'BIC': 40},
            'summary': {'estimated_num_populations': 4,
                        'final_neg_llh': 1,
                        'best_optimization_idx': 4}
        }

        # Call the function
        fig = plot_bic(results)

        # Check if the figure is created
        self.assertIsInstance(fig, plt.Figure)

        # Check if the plot has the correct number of points
        ax = fig.gca()
        self.assertEqual(len(ax.lines[0].get_xdata()), len(results) - 1)
        self.assertEqual(len(ax.lines[0].get_ydata()), len(results) - 1)

        # Check if the y-data matches the expected BIC values
        expected_bic = [100, 80, 60, 40]
        self.assertListEqual(list(ax.lines[0].get_ydata()), expected_bic)

        # Check labels
        self.assertEqual(ax.get_ylabel(), 'BIC')
        self.assertEqual(ax.get_xlabel(), 'Number of inferred populations')

    @patch('pyphenopop.plotting.plt.show')
    def test_plot_aic(self, mock_show):
        # Mock data
        results = {
            '1_subpopulations': {'AIC': 90},
            '2_subpopulations': {'AIC': 70},
            '3_subpopulations': {'AIC': 50},
            '4_subpopulations': {'AIC': 30},
            'summary': {'estimated_num_populations': 4,
                        'final_neg_llh': 1,
                        'best_optimization_idx': 4}
        }

        # Call the function
        fig = plot_aic(results)

        # Check if the figure is created
        self.assertIsInstance(fig, plt.Figure)

        # Check if the plot has the correct number of points
        ax = fig.gca()
        self.assertEqual(len(ax.lines[0].get_xdata()), len(results) - 1)
        self.assertEqual(len(ax.lines[0].get_ydata()), len(results) - 1)

        # Check if the y-data matches the expected AIC values
        expected_aic = [90, 70, 50, 30]
        self.assertListEqual(list(ax.lines[0].get_ydata()), expected_aic)

        # Check labels
        self.assertEqual(ax.get_ylabel(), 'AIC')
        self.assertEqual(ax.get_xlabel(), 'Number of inferred populations')

    def test_plot_gr50_with_dict(self):
        fig = plot_gr50(self.results_dict, self.concentrations, 'best')
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_gr50_with_invalid_results(self):
        with self.assertRaises(TypeError):
            plot_gr50("invalid_results", self.concentrations, 'best')

    def test_plot_gr50_with_mismatched_lengths(self):
        with self.assertRaises(Exception):
            plot_gr50(self.results_list, self.concentrations, [3])

    def test_plot_gr50_subplot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
        plot_gr50_subplot(ax1, ax2, self.results_dict, self.concentrations, 'best')
        self.assertTrue(len(ax1.patches) > 0)  # Check if pie chart is plotted
        self.assertTrue(len(ax2.lines) > 0)    # Check if GR50 lines are plotted


if __name__ == '__main__':
    unittest.main()
