import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union
from pyphenopop.mixpopid import rate_expo
import pandas as pd
import copy
from matplotlib.ticker import MaxNLocator


def plot_neg_llh(results: Dict):
    """
    Plots the negative log-likelihood values for all considered models
    Arguments:
        * results: Dictionary with results returned by mixture_id.
    """
    final_nllhs = [np.min(results[f'{idx}_subpopulations']['fval']) for idx in range(1, len(results))]
    ax = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(results)), final_nllhs, 'o-')
    plt.ylabel('Negative log-likelihood')
    plt.xlabel('Number of inferred populations')
    return ax


def plot_bic(results: Dict):
    """
    Plots the BIC values for all considered models
    Arguments:
        * results: Dictionary with results returned by mixture_id.
    """
    final_bic = [results[f'{idx}_subpopulations']['BIC'] for idx in range(1, len(results))]
    ax = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(results)), final_bic, 'o-')
    plt.ylabel('BIC')
    plt.xlabel('Number of inferred populations')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax


def plot_aic(results: Dict):
    """
    Plots the AIC values for all considered models
    Arguments:
        * results: Dictionary with results returned by mixture_id.
    """
    final_bic = [results[f'{idx}_subpopulations']['AIC'] for idx in range(1, len(results))]
    ax = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(results)), final_bic, 'o-')
    plt.ylabel('AIC')
    plt.xlabel('Number of inferred populations')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def plot_in_time(data_file: str,
                 num_replicates: int,
                 timepoints: Union[list, np.ndarray],
                 concentrations: Union[list, np.ndarray],
                 title: str = None):
    """
    Plots the data for each concentration over time.
    Arguments:
        * data_file: Name of the file containing the measured cell counts.
        * num_replicates: number of replicates
        * timepoints: List of time points measured in hours.
        * concentrations: List of concentrations considered.
        * title: Title used for plotting.
    """
    num_concentrations = len(concentrations)
    num_timepoints = len(timepoints)
    measurements = np.array(pd.read_csv(data_file, header=None))
    measurements = measurements.reshape((num_timepoints, num_replicates, num_concentrations))
    measurements = measurements.transpose(1, 2, 0)
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


def plot_in_conc(data_file: str,
                 num_replicates: int,
                 timepoints: Union[list, np.ndarray],
                 concentrations: Union[list, np.ndarray],
                 title: str = None):
    """
    Plots the data for each timepoint over concentrations.
    Arguments:
        * data_file: Name of the file containing the measured cell counts.
        * num_replicates: number of replicates
        * timepoints: List of time points measured in hours.
        * concentrations: List of concentrations considered.
        * title: Title used for plotting.
    """
    num_concentrations = len(concentrations)
    num_timepoints = len(timepoints)
    measurements = np.array(pd.read_csv(data_file, header=None))
    measurements = measurements.reshape((num_timepoints, num_replicates, num_concentrations))
    measurements = measurements.transpose(1, 2, 0)

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
    return ax


def plot_gr50(results: Union[Dict, list],
              concentrations: Union[list, np.ndarray],
              subpopulation_indices: Union[int, str, list]):
    """
    Arguments:
        * results: Results dictionary.
        * concentrations: List of concentrations considered.
        * subpopulation_indices: Number of subpopulations (either an integer or 'best', if the best model should be
        taken).
    """
    if isinstance(results, list):
        if len(results) != len(concentrations):
            raise Exception('Result and concentration lists must have the same length.')
        f, ax = plt.subplots(len(results), 2, gridspec_kw={'width_ratios': [1, 3]})
        for res_idx, (result, concentration, subpopulation_idx) in enumerate(
                zip(results, concentrations, subpopulation_indices)):
            ax1 = ax[res_idx, 0]
            ax2 = ax[res_idx, 1]
            if res_idx == 0:
                ax1.set_title('Estimated mixture')
                ax2.set_title('Estimated GR50 values')
            plot_gr50_subplot(ax1, ax2, result, concentration, subpopulation_idx)
    elif isinstance(results, Dict):
        f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
        plot_gr50_subplot(ax[0], ax[1], results, concentrations, subpopulation_indices)
    else:
        raise TypeError
    return f


def plot_gr50_subplot(ax1,
                      ax2,
                      result: Dict,
                      concentrations: Union[list, np.ndarray],
                      subpopulation_index: Union[int, str] = 'best'):
    default_colors = ['#2b1d72', '#b83d52', '#d2bc4b', '#aa4499', '#882255', '#88ccee', '#44aa99', '#999933', '#117733',
                      '#dddddd']
    if subpopulation_index == 'best':
        subpopulation_index = result['summary']['estimated_num_populations']
    mixture_params = list(result['summary']['final_parameters'][:subpopulation_index - 1])
    mixture_params.append(1 - np.sum(mixture_params))
    mixture_params = np.array(mixture_params)
    gr50 = np.array(result[f'{subpopulation_index}_subpopulations']['gr50'])
    gr50_ixs = np.argsort(gr50)
    gr50 = gr50[gr50_ixs]
    gr50 = list(gr50)
    mixture_params = mixture_params[gr50_ixs]

    concentration_ticks = copy.copy(concentrations)
    if concentration_ticks[0] == 0.0:
        concentration_ticks[0] = np.min([concentration_ticks[1], np.min(gr50)]) * 0.1
    xticks = list(dict.fromkeys(list(np.round(np.log10(concentration_ticks)))))
    if concentrations[0] == 0.0:
        xticks[0] = np.log10(concentration_ticks[0])

    ax1.pie(mixture_params, labels=[f'{np.round(mixture_params[idx] * 100)}%' for idx in range(len(mixture_params))],
            colors=default_colors, wedgeprops={'linewidth': 1, 'edgecolor': 'k'})

    [ax2.semilogx([concentration_ticks[i]] * 2, [0, 1], color='0.7') for i in range(len(concentration_ticks))]

    for gr_idx, gr in enumerate(gr50):
        gr_larger_conc = list(concentration_ticks < gr)
        if all(gr_larger_conc):
            lower_conc = np.max(concentration_ticks)
            upper_conc = lower_conc + (np.max(concentration_ticks) - np.min(concentration_ticks))
        else:
            upper_idx = gr_larger_conc.index(False)
            lower_conc = concentration_ticks[upper_idx - 1]
            upper_conc = concentration_ticks[upper_idx]
            ax2.plot(gr, [0.5], 'ko', markerfacecolor='w', markersize=5, markeredgewidth=1.5)
        ax2.fill_betweenx([0.25, 0.75], lower_conc, upper_conc, color=default_colors[gr_idx])

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_yaxis().set_visible(False)
    if concentrations[0] == 0.0:
        ticklabels = ['0'] + ['$10^{' + format(np.log10(elem), ".0f") + '}$' for elem in
                              concentration_ticks[1:]]
    else:
        ticklabels = ['$10^{' + format(np.log10(elem), ".0f") + '}$' for elem in
                      concentration_ticks]
    ticklabels = list(dict.fromkeys(ticklabels))
    ax2.set_xscale('log')
    ax2.set_xticks(10 ** np.array(xticks), ticklabels)

    ax2.minorticks_off()
    plt.xlabel('Drug concentration')
