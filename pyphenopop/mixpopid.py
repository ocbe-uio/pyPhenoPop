import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brentq
from typing import Union, Dict, Tuple
from tqdm import tqdm
import pandas as pd


def rate_expo(parameters: list,
              concentrations: np.ndarray) -> np.ndarray:
    """
    A function calculating a growth rate of a certain cell population with 
    parameters exposed to a certain drug with given concentrations according
    to the exponential cell population growth model.
    
    Arguments:
        * parameters: parameter vector containing 4 parameters (alpha, b, E, n)
        * concentrations: drug concentration vector
        
    """

    return parameters[0] + np.log(
        parameters[1] + (1 - parameters[1]) / (1 + (concentrations / parameters[2]) ** parameters[3]))


def pop_expo(parameters: list,
             concentrations: np.ndarray,
             timepoints: np.ndarray) -> np.ndarray:
    """
    A function calculating a certain cell population count using a function
    'rate' at given time points according to the exponential cell population 
    growth model.
    
    Arguments:
        * parameters: parameter vector containing 4 parameters (alpha, b, E, n)
        * concentrations: drug concentration vector
        * timepoints: time vector
        
    """
    # Reshaping the timepoints and the rates vectors to perform a multiplication:
    timepoints_rshpe = np.reshape(timepoints, (1, len(timepoints)))
    rates = np.reshape(rate_expo(parameters, concentrations), (len(concentrations), 1))
    # Returns a matrix with population counts where rows correspond to
    # concentrations and columns to time points:
    return np.exp(rates @ timepoints_rshpe)


def neg_log_likelihood(max_subpop: int,
                       parameters: np.ndarray,
                       measurements: np.ndarray,
                       concvec: np.ndarray,
                       timevec: np.ndarray,
                       num_replicates: int,
                       model: str,
                       num_timepoints_high: np.ndarray,
                       num_conc_high_noise: np.ndarray,
                       num_noise_high: int,
                       num_noise_low: int) -> float:
    """
    This function calculates the negative log-likelihood for a given
    model. Function's minimum is the optimal parameter estimate. 
    
    Arguments:
        * max_subpop: maximum number of cell subpopulations considered
        * parameters: Model parameters; for every model the first PopN-1 elements of
        Params must be mixture parameters, the last two are higher and lower 
        variance levels. Higher variance appears at concentration levels
        smaller than ConcT and time points after TimeT
        * measurements: Cell count at each independent variable condition. measurements should
        be structured as follows: DATA[k][j][i] ia cell count for the k-th 
        replicate at the j-th concentration at the i-th time point
        * concvec: List of concentrations considered measured in micromoles
        * timevec: List of time points measured in hours
        * num_replicates: number of replicates
        * model: A string representing a cell population growth model
        considered, for example 'expo' means exponential model. This variable
        determines a form of the parameter vector and a function calculating 
        cell population number.
        * num_timepoints_high: Number of time points with higher level of noise
        * num_conc_high_noise: Number of concentrations with higher level of noise
        * num_noise_high: Number of data points with high level of noise
        * num_noise_low: Number of data points with low level of noise
    """

    # Creating a mixture parameter vector:
    if max_subpop > 1:
        mixture_params = parameters[0:max_subpop - 1]
    else:
        mixture_params = [0, ]

    # Creating a model specific parameter vector based on the chosen model:
    p = parameters[max_subpop - 1:-2]
    if model == 'expo':
        if len(p) % 4 == 0:
            parameters_per_subpop = [[p[4 * j + i] for i in np.arange(4)] for j in np.arange(max_subpop)]
        else:
            raise KeyError('Error: parameter vector not suited for the chosen model')
        pop_model = pop_expo
    else:
        raise NotImplementedError

    # Higher variance level:
    sigma_high = parameters[5 * max_subpop - 1]

    # Lower variance level:
    sigma_low = parameters[5 * max_subpop]

    sum_resid = 0

    # Matrix with noise:
    sigma_matrix = np.ones((len(concvec), len(timevec))) / (2 * sigma_low ** 2)
    sigma_matrix[:num_conc_high_noise, len(timevec) - num_timepoints_high:] = 1 / (
            2 * sigma_high ** 2)  # higher noise is in the top right
    # corner (time bigger than time_threshold, concentration lower than conc_threshold)
    x_all = []
    for pop_idx in range(max_subpop - 1):
        x_all.append(mixture_params[pop_idx] * pop_model(parameters_per_subpop[pop_idx], concvec, timevec))
    x_all.append((1 - np.sum(mixture_params)) * pop_model(parameters_per_subpop[max_subpop - 1], concvec, timevec))
    for rep_idx in np.arange(num_replicates):
        simulated_counts = 0
        initial_counts = np.reshape(np.repeat(measurements[rep_idx, :, 0], len(timevec)), (len(concvec), len(timevec)))
        for pop_idx in range(max_subpop - 1):
            x = x_all[pop_idx]
            # We need to multiply measurements[k,:,0] (the initial cell count for all
            # concentrations for k-th replicate) by x, element by element
            # concentration wise, measurements[k,:,0] being the same for every time point
            # of x, so we form a matrix where measurements[k,:,0] is repeated the same
            # number of times as the number of time points, and than reshaped to
            # correspond to the dimensions of x for element wise multiplication:
            simulated_counts += initial_counts * x  # s is a matrix containing counts summed for
            # all subpopulation in a corresponding proportion for all time points and concentrations

        # Last subpopulation:
        x = x_all[-1]
        simulated_counts += initial_counts * x

        # Matrix containing residuals for all time points (except time 0 since
        # in this case the residual is zero by definition) and concentrations:
        resid = measurements[rep_idx, :, 1:] - simulated_counts

        # Adding noise by multiplying every residual by a corresponding noise
        # from the noise matrix, and calculating the sum for all time points
        # and concentrations:
        sum_resid += np.sum((resid ** 2) * sigma_matrix)
    return sum_resid + (num_noise_high / 2) * np.log(2 * np.pi * sigma_high ** 2) + (num_noise_low / 2) * np.log(
        2 * np.pi * sigma_low ** 2)


def mixture_id(max_subpop: int,
               data_file: str,
               timepoints: Union[list, np.ndarray],
               concentrations: Union[list, np.ndarray],
               num_replicates: int,
               model: str = 'expo',
               bounds_model: Dict = None,
               bounds_sigma_high: Tuple = (1e-05, 10000.0),
               bounds_sigma_low: Tuple = (1e-05, 5000.0),
               optimizer_options: Dict = None,
               num_optim: int = 200,
               selection_method: str = 'BIC') -> Dict:
    """
    This is a function that serves to determine the number of cell 
    subpopulations found in a given mixture with a maximum of PopN, and in what
    proportion these subpopulations are admixed, as well as model specific 
    subpopulation parameters based on the cell count data.
    
    The data provided by the user along with the independent variables vectors 
    (concentration levels, time points and number of replicates) is used to 
    find the parameter vector that better fits the data into the model, that is
    the one that minimizes the objective function that 
    calculates the negative log-likelihood for the chosen model. 
    The algorithm is based on the function 'minimize' from the optimization 
    package 'scipy.optimize'. User can specify a reference model, parameter 
    bounds used in optimization and a number of optimization attempts.
    
    This function performs this optimization procedure for all possible 
    population numbers ranging from 1 to PopN (the maximum number given by the 
    user), and selects N (estimated number of subpopulations) that the 
    corresponding model better fits the data using BIC (Bayesian information 
    criterion). The function returns a parameter vector in a form
    of an array. The first N-1 parameters are estimated proportions of N-1 
    subpopulations in the mixture (respectively, the N-th subpopulation's 
    proportion in the mixture is 1 minus their sum). The last two are estimated
    variance levels of the mixture (higher and lower depending on the 
    concentration level and on the time point considered). The rest are model 
    parameters; for example, in case of the exponential growth model every 
    cell line is characterized by 4 parameters (alpha, b: minimal drug response 
    value, E: IC50 and n: steepness parameter), therefore the middle part of 
    the vector is in the following form: 
    (alpha1, b1, E1, n1, ..., alpha_N, b_N, E_PopN, n_N).
    
    Additionally, this function produces graphs of the estimated growth rates 
    for all subpopulations considered in response to the increase in drug 
    concentration. Combined with the mixture parameter information, these 
    graphs help to determine to what extent every subpopulation in the mixture,
    major or minor, is responsive to the drug, which is useful information to
    design personalized treatment strategies. 
    
    Arguments:
        * max_subpop: maximum number of cell subpopulations considered
        * data_file: Name of the file containing the measured cell counts.
        * timepoints: list of time points measured in hours
        * concentrations: list of concentrations considered
        * num_replicates: number of replicates
        * model: cell population growth model considered, exponential ('expo')
        by default
        * bounds_model: bounds for parameter inference in a form of a
        dictionary where keys are model specific parameter names and values are tuples (lower and higher bounds);
        * bounds_sigma_high: higher sigma bounds used to infer the parameter in a
        form of a tuple (lower and higher bounds); default: bounds_sigma_high = (1e-05, 1000.0)
        * bounds_sigma_low: lower sigma bounds used to infer the parameter in a
        form of a tuple (lower and higher bounds); default: bounds_sigma_low = (1e-05, 1000.0)
        * optimizer_options: Dict with keys 'method' and 'options' that is passed to scipy.optimize.minimize to adapt
        the optimization algorithm and optimization options
        * num_optim: number of objective function optimization attempts; default: num_optim = 200
        * selection_method: Model selection method. Either 'AIC' or 'BIC'.
    """

    if bounds_model is None:
        bounds_model = {'alpha': (0.0, 0.1), 'b': (0.0, 1.0), 'E': (1e-06, 15), 'n': (0.01, 10)}
    elif set(bounds_model.keys()) != {'alpha', 'b', 'E', 'n'}:
        raise KeyError('Bounds should contain the keys "alpha", "b", "E" and "n"')

    if optimizer_options is None:
        optimizer_options = {'method': 'L-BFGS-B', 'options': {'disp': False, 'ftol': 1e-12}}

    num_timepoints = len(timepoints)
    num_concentrations = len(concentrations)

    measurements = np.array(pd.read_csv(data_file, header=None))
    measurements = measurements.reshape((num_timepoints, num_replicates, num_concentrations))
    measurements = measurements.transpose(1, 2, 0)

    # Fixed thresholds for concentration and time in order to choose either
    # higher or lower variance level (sigma_high and sigma_low):
    conc_threshold = 0.1
    time_threshold = 48

    num_datapoints = num_concentrations * num_timepoints * num_replicates

    # Initializing Bayesian information criterion:
    bic = np.inf
    aic = np.inf
    final_pop_idx = 1
    x_final_all = []
    fval_all = []

    # Comparing BIC for all models with a number of populations considered from
    # 1 to max_subpop in order to find the best fit:
    sorted_timepoints = np.sort(timepoints)[1:]  # time starts from the second value since the
    # first one is taken from the data
    sorted_concentrations = np.sort(concentrations)
    num_timepoints_high = np.sum(np.where(sorted_timepoints >= time_threshold, 1, 0))
    num_conc_high_noise = np.sum(np.where(sorted_concentrations <= conc_threshold, 1, 0))
    num_noise_high = num_timepoints_high * num_conc_high_noise * num_replicates
    num_noise_low = num_replicates * len(sorted_timepoints) * num_concentrations - num_noise_high
    results = {}
    for num_subpop in np.arange(1, max_subpop + 1):
        print(f'Optimizing for {num_subpop} subpopulation(s)')
        subpop_key = f'{num_subpop}_subpopulations'
        results[subpop_key] = {'fval': [], 'parameters': [], 'BIC': np.inf, 'AIC': np.inf}

        def obj(x):
            return neg_log_likelihood(num_subpop,
                                      x,
                                      measurements,
                                      sorted_concentrations,
                                      sorted_timepoints,
                                      num_replicates,
                                      model,
                                      num_timepoints_high,
                                      num_conc_high_noise,
                                      num_noise_high,
                                      num_noise_low)

        bnds, lb, ub = get_optimization_bounds(num_subpop, bounds_model, bounds_sigma_low, bounds_sigma_high)

        for _ in tqdm(np.arange(num_optim)):
            x0 = np.random.uniform(lb, ub)
            try:
                result = minimize(obj, x0, method=optimizer_options['method'], bounds=bnds,
                                  options=optimizer_options['options'])
                tmp_mix_params = list(result['x'][0:(num_subpop - 1)])
                tmp_mix_params.append(1 - np.sum(tmp_mix_params))
                tmp_mix_params = np.array(tmp_mix_params)
                if np.any(tmp_mix_params < 0):
                    results[subpop_key]['fval'].append(np.inf)
                    results[subpop_key]['parameters'].append(result['x'])
                else:
                    results[subpop_key]['fval'].append(result['fun'])
                    results[subpop_key]['parameters'].append(result['x'])
            except ValueError:
                pass
        final_idx = np.argmin(results[f'{num_subpop}_subpopulations']['fval'])
        fval = results[subpop_key]['fval'][final_idx]
        x_final = results[subpop_key]['parameters'][final_idx]
        results[subpop_key]['final_fval'] = fval
        results[subpop_key]['final_parameters'] = x_final
        results[subpop_key]['gr50'] = get_gr50(x_final, concentrations, num_subpop)
        x_final_all.append(x_final)
        fval_all.append(fval)
        bic_temp = len(bnds) * np.log(num_datapoints) + 2 * fval
        results[subpop_key]['BIC'] = bic_temp

        aic_temp = len(bnds) * 2 + 2 * fval
        results[subpop_key]['AIC'] = aic_temp

        if selection_method == 'AIC':
            if aic_temp < aic:
                aic = aic_temp
                final_pop_idx = num_subpop
            if bic_temp < bic:
                bic = bic_temp
        elif selection_method == 'BIC':
            if aic_temp < aic:
                aic = aic_temp

            if bic_temp < bic:
                bic = bic_temp
                final_pop_idx = num_subpop
        else:
            raise NotImplementedError(
                f'Selection method should be either "AIC" or "BIC". {selection_method} was provided'
            )

    results['summary'] = {'estimated_num_populations': final_pop_idx,
                          'final_neg_log_likelihood': fval_all[final_pop_idx - 1],
                          'best_optimization_idx': np.argmin(results[f'{final_pop_idx}_subpopulations']['fval']),
                          'final_parameters': x_final_all[final_pop_idx - 1]}

    print_results(x_final_all, fval_all, final_pop_idx, concentrations, model, selection_method)

    return results


def print_results(x_final_all: list,
                  fval_all: list,
                  final_pop_idx: int,
                  concentrations: Union[list, np.ndarray],
                  model: str,
                  selection_method: str):
    """
    Prints a summary of the results for a specific subpopulation model (usually the best model defined by the
    selection_method.
    Arguments:
        * x_final_all: List with optimized parameters for all subpopulation models.
        * fval_all: List with optimal neg. log-likelihood values for all subpopulation models.
        * final_pop_idx: Index of subpopulation model for which results should be printed.
        * concentrations: List of concentrations considered.
        * model: Cell population growth model considered, exponential. Currently only 'expo' is supported.
        * selection_method: Method used for model selection. Only used for printing purposes.
    """
    x_final = x_final_all[final_pop_idx - 1]
    fval = fval_all[final_pop_idx - 1]
    gr50 = get_gr50(x_final, concentrations, final_pop_idx)
    print(f'Estimated number of cell populations based on {selection_method}: {final_pop_idx}')
    print(f'Minimal negative log-likelihood value found: {fval}')
    mixture_parameters = list(x_final[0:(final_pop_idx - 1)])
    mixture_parameters.append(1 - np.sum(mixture_parameters))
    print('Mixture parameter(s): ', 1 if final_pop_idx == 1 else mixture_parameters)
    if model == 'expo':
        for idx in range(final_pop_idx):
            print(f'Model parameters for subpopulation {idx + 1}:')
            print(f'Estimated GR50: {gr50[idx]}')
            print('alpha : ', x_final[final_pop_idx - 1 + 4 * idx])
            print('b : ', x_final[final_pop_idx - 1 + 4 * idx + 1])
            print('E : ', x_final[final_pop_idx - 1 + 4 * idx + 2])
            print('n : ', x_final[final_pop_idx - 1 + 4 * idx + 3])
        print('Sigma high:', x_final[-2])
        print('Sigma low:', x_final[-1])
    else:
        raise NotImplementedError


def get_optimization_bounds(num_subpop: int,
                            bounds_model: Dict,
                            bounds_sigma_low: Tuple,
                            bounds_sigma_high: Tuple) -> Tuple[Tuple, list, list]:
    """
    Arguments:
        * num_subpop: Number of subpopulations.
        * bounds_model: Dictionary with parameter bounds for 'alpha', 'b', 'E' and 'n'.
        * bounds_sigma_low: Parameter bounds for sigma_low.
        * bounds_sigma_high: Parameter bounds for sigma_high.
    """
    # Changing the format of chosen bounds to fit the optimization procedure:
    if num_subpop > 1:
        bnds = [(0.0, 0.5) for _ in np.arange(num_subpop - 1)]
    else:
        bnds = []

    for _ in np.arange(num_subpop):
        bnds.append(bounds_model['alpha'])
        bnds.append(bounds_model['b'])
        bnds.append(bounds_model['E'])
        bnds.append(bounds_model['n'])

    bnds.append(bounds_sigma_high)
    bnds.append(bounds_sigma_low)
    bnds = tuple(bnds)

    lb = [bnds[i][0] for i in range(len(bnds))]
    ub = [bnds[i][1] for i in range(len(bnds))]
    return bnds, lb, ub


def get_gr50(parameters: list,
             concentrations: Union[list, np.ndarray],
             max_subpop: int) -> list:
    """
    Calculate GR50 values.

    Arguments:
        * parameters: Model parameters used for GR50 calculations.
        * concentrations: List of concentrations considered.
        * max_subpop: Maximal number of subpopulations.
    """
    p = parameters[max_subpop - 1:-2]
    parameters_per_subpop = [[p[4 * j + i] for i in np.arange(4)] for j in np.arange(max_subpop)]

    min_conc = np.min(concentrations)
    max_conc = np.max(concentrations)
    gr50_all = []
    for pop_idx in range(max_subpop):
        gr_min_conc = rate_expo(parameters_per_subpop[pop_idx], min_conc)
        gr_max_conc = rate_expo(parameters_per_subpop[pop_idx], max_conc)

        try:
            gr50 = brentq(
                lambda x: gr_max_conc + gr_min_conc - 2 * rate_expo(parameters_per_subpop[pop_idx], x),
                0.0000001,
                max_conc)
            gr50_all.append(gr50)
        except Exception as err:
            print(f'No GR50 value was found; Setting GR50 value to max_concentration. Error message: {err}')
            gr50_all.append(max_conc)

    return gr50_all
