import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union
from pyphenopop.mixpopid import rate_expo


def plot_growth_curves(results: Dict,
                       concentrations: Union[list, np.ndarray],
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


def plot_in_time(measurements: np.ndarray,
                 num_concentrations: int,
                 num_replicates: int,
                 timepoints: Union[list, np.ndarray],
                 concentrations: Union[list, np.ndarray],
                 title: str = None):
    conc_colors = [plt.cm.viridis(1 - 1 / (len(concentrations)) - conc_idx / (len(concentrations))) for conc_idx in
                   range(len(concentrations))]
    meanval = np.mean(measurements, axis=0)
    stdval = np.std(measurements, axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    for conc_index in range(num_concentrations):
        ax.errorbar(timepoints, meanval[conc_index, :], yerr=stdval[conc_index, :], color=conc_colors[conc_index],
                    label="Concentration " + str(concentrations[conc_index]))
        for rep_index in range(num_replicates):
            ax.scatter(timepoints, measurements[rep_index, conc_index, :], color=conc_colors[conc_index], label=None)
    plt.xlabel("Time")
    plt.ylabel("Cell count")
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    return ax


def plot_in_conc(measurements: np.ndarray,
                 num_timepoints: int,
                 num_replicates: int,
                 timepoints: Union[list, np.ndarray],
                 concentrations: Union[list, np.ndarray],
                 title: str = None):
    time_colors = [plt.cm.plasma(1 - 1 / num_timepoints - time_idx / num_timepoints) for time_idx in
                   range(num_timepoints)]
    meanval = np.mean(measurements, axis=0)
    stdval = np.std(measurements, axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    for time_index in range(1, num_timepoints):
        ax.errorbar(concentrations, meanval[:, time_index], yerr=stdval[:, time_index], color=time_colors[time_index],
                    label=str(timepoints[time_index]) + " hours")
        for rep_index in range(num_replicates):
            if rep_index == 0:
                ax.scatter(concentrations, measurements[rep_index, :, time_index], color=time_colors[time_index])
            else:
                ax.scatter(concentrations, measurements[rep_index, :, time_index], color=time_colors[time_index])
    plt.xlabel("Drug concentration")
    plt.ylabel("Cell count")
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
