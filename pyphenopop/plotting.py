import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union
from pyphenopop.mixpopid import rate_expo


def plot_growth_curves(results: Dict,
                       concentrations: np.ndarray,
                       subpopulation_index: Union[int, str] = 'best'):
    if subpopulation_index == 'best':
        subpopulation_index = results['summary']['estimated_num_populations']
    x_final = results[f'{subpopulation_index}_subpopulations']['final_parameters']
    ax = plt.figure(figsize=(10, 8))
    for i in range(subpopulation_index):
        param = x_final[4 * i + subpopulation_index - 1:4 * i + subpopulation_index + 3]
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


def plot_aic(results: Dict):
    final_bic = [results[f'{idx}_subpopulations']['AIC'] for idx in range(1, len(results))]
    ax = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(results)), final_bic, 'o-')
    plt.ylabel('AIC')
    plt.xlabel('Number of inferred populations')
    return ax
