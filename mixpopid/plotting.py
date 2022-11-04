import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from mixpopid.mixpopid import rate_expo


def plot_growth_curves(results: Dict,
                       concentrations: np.ndarray):
    final_pop_idx = results['summary']['estimated_num_populations']
    x_final = results['summary']['final_parameters']
    ax = plt.figure(figsize=(10, 8))
    for i in range(final_pop_idx):
        param = x_final[4 * i + final_pop_idx - 1:4 * i + final_pop_idx + 3]
        plt.semilogx(concentrations, [rate_expo(param, x) for x in sorted(concentrations)], '-*', linewidth=3,
                     label="Subpopulation #%s" % (i + 1))

    plt.xlabel('Drug Concentration')
    plt.ylabel('Growth rate')
    plt.title('Estimated growth rates')
    plt.legend()
    return ax


def plot_elbow(results: Dict):
    final_nllhs = [np.min(results[f'{idx}_subpopulations']['fval']) for idx in range(1, len(results))]
    ax = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(results)), final_nllhs, 'o-')
    plt.ylabel('Negative log-likelihood')
    plt.xlabel('Number of inferred populations')
    return ax


def plot_bic(results: Dict):
    final_bic = [results[f'{idx}_subpopulations']['BIC'] for idx in range(1, len(results))]
    ax = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(results)), final_bic, 'o-')
    plt.ylabel('BIC')
    plt.xlabel('Number of inferred populations')
    return ax
