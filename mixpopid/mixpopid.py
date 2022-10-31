import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def rate_expo(params, concentrations):
    """
    A function calculating a growth rate of a certain cell population with 
    parameters p exposed to a certain drug with given concentrations according
    to the exponential cell population growth model.
    
    Arguments:
        * params: parameter vector containing 4 parameters (alpha, b, E, n)
        * concentrations: drug concentration vector
        
    """

    return params[0] + np.log(params[1] + (1 - params[1]) / (1 + (concentrations / params[2]) ** params[3]))


def pop_expo(params, concentrations, timepoints):
    """
    A function calculating a certain cell population count using a function
    'rate' at given time points according to the exponential cell population 
    growth model.
    
    Arguments:
        * params: parameter vector containing 4 parameters (alpha, b, E, n)
        * concentrations: drug concentration vector
        * timepoints: time vector
        
    """
    # Reshaping the time points and the rates vectors to perform a
    # multiplication:
    T_ = np.reshape(timepoints, (1, len(timepoints)))
    rates = np.reshape(rate_expo(params, concentrations), (len(concentrations), 1))
    # Returns a matrix with population counts where lines correspond to 
    # concentrations and columns to time points:
    return np.exp(rates @ T_)


def objective(max_subpop, parameters, measurements, concvec, timevec, num_replicates, model, num_timepoints_high, num_conc_high_noise, num_noise_high, num_noise_low):
    """
    This function calculates the negative value of log-likelihood for a given
    model. Function's minimum is the optimal parameter estimate. 
    
    Arguments:
        * max_subpop: Number of cell lines considered
        * parameters: Model parameters; for every model the first PopN-1 elements of
        Params must be mixture parameters, the last two are higher and lower 
        variance levels. Higher variance appears at concentration levels
        smaller than ConcT and time points after TimeT
        * measurements: Cell count at each independent variable condition. DATA should
        be structured as follows: DATA[k][j][i] ia cell count for the k-th 
        replicate at the j-th concentration at the i-th time point
        * concvec: List of concentrations considered measured in micromoles
        * timevec: List of time points measured in hours
        * num_replicates: number of replicates
        * model: A string representing a cell population growth model
        considered, for example "expo" means exponential model. This variable 
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
    if model == "expo":
        if len(p) % 4 == 0:
            PMat = [[p[4 * j + i] for i in np.arange(4)] for j in np.arange(max_subpop)]
        else:
            raise KeyError("Error: parameter vector not suited for the chosen model")
        # Choosing a function calculating cell population number based on the
        # chosen model:
        pop_model = pop_expo
    else:
        raise NotImplementedError

    # Higher variance level:
    sigH = parameters[5 * max_subpop - 1]

    # Lower variance level:
    sigL = parameters[5 * max_subpop]

    sum_resid = 0

    # Matrix with noise:
    sigMat = np.ones((len(concvec), len(timevec))) / (2 * sigL ** 2)
    sigMat[:num_conc_high_noise, len(timevec) - num_timepoints_high:] = 1 / (2 * sigH ** 2)  # higher noise is in the top right
    # corner (time bigger than 48 hours, concentration lower than 0.1)
    x_all = []
    for pop_idx in range(max_subpop - 1):
        x_all.append(mixture_params[pop_idx] * pop_model(PMat[pop_idx], concvec, timevec))
    x_all.append((1 - np.sum(mixture_params)) * pop_model(PMat[max_subpop - 1], concvec, timevec))
    for rep_idx in np.arange(num_replicates):
        simulated_counts = 0
        initial_counts = np.reshape(np.repeat(measurements[rep_idx, :, 0], len(timevec)), (len(concvec), len(timevec)))
        for pop_idx in range(max_subpop - 1):
            X = x_all[pop_idx]
            # We need to multiply DATA[k,:,0] (the initial cell count for all
            # concentrations for k-th replicate) by X, element by element
            # concentration wise, DATA[k,:,0] being the same fot every time point
            # of X, so we form a matrix where DATA[k,:,0] is repeated the same
            # number of times as the number of time points, and than reshaped to
            # correspond to the dimensions of X for element wise multiplication:
            simulated_counts += initial_counts * X  # s is a matrix containing counts summed for
            # all subpopulation in a corresponding proportion for all time
            # points and concentrations

        # Last subpopulation:
        X = x_all[-1]
        simulated_counts += initial_counts * X

        # Matrix containing residuals for all time points (except time 0 since
        # in this case the residual is zero by definition) and concentrations:
        resid = measurements[rep_idx, :, 1:] - simulated_counts

        # Adding noise by multiplying every residual by a corresponding noise
        # from the noise matrix, and calculating the sum for all time points
        # and concentrations:
        sum_resid += np.sum((resid ** 2) * sigMat)
    return sum_resid + (num_noise_high / 2) * np.log(2 * np.pi * sigH ** 2) + (num_noise_low / 2) * np.log(2 * np.pi * sigL ** 2)


def mixtureID(max_subpop, measurements, timepoints, concentrations, num_replicates, model="expo",
              bounds_model=None, bounds_sigma_high=(1e-05, 10000.0),
              bounds_sigma_low=(1e-05, 5000.0), num_optim=200):
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
        * measurements: cell count at each independent variable condition. DATA should
        be structured as follows: DATA[k][j][i] ia a cell count for the k-th 
        replicate at the j-th concentration at the i-th time point
        * timepoints: list of time points measured in hours
        * concentrations: list of concentrations considered  measured in micromoles
        * num_replicates: number of replicates
        * model: cell population growth model considered, exponential ("expo") 
        by default
        * bounds_model: bounds for parameter inference in a form of a 
        dictionary where keys are model specific parameter names and values are
        tuples (lower and higher bounds); default: bounds_model = bounds_expo
        * bounds_sigma_high: higher sigma bounds used to infer the parameter in a
        form of a tuple (lower and higher bounds); default: bounds_sigma_high = (1e-05, 1000.0)
        * bounds_sigma_low: lower sigma bounds used to infer the parameter in a
        form of a tuple (lower and higher bounds); default: bounds_sigma_low = (1e-05, 1000.0)
        * num_optim: number of objective function optimization attempts; default: num_optim = 200
        
    """

    if bounds_model is None:
        bounds_model = {'alpha': (0.0, 0.1), 'b': (0.0, 1.0), 'E': (1e-06, 15), 'n': (0.01, 10)}
    elif set(bounds_model.keys()) != {'alpha', 'b', 'E', 'n'}:
        raise KeyError('Bounds should contain the keys "alpha", "b", "E" and "n"')

    # Fixed thresholds for concentration and time in order to choose either
    # higher or lower variance level (sigmaH and sigmaL):
    conc_threshold = 0.1
    time_threshold = 48

    num_datapoints = len(concentrations) * len(timepoints) * num_replicates

    # Initializing Bayesian information criterion:
    bic = np.inf
    final_pop_idx = 1
    x_final_all = []
    fval_all = []

    # Comparing BIC for all models with a number of populations considered from
    # 1 to PopN in order to find the best fit:
    sorted_timepoints = np.sort(timepoints)[1:]  # time starts from the second value since the
    # first one is taken from the data
    sorted_concentratiosn = np.sort(concentrations)
    num_timepoints_high = np.sum(np.where(sorted_timepoints >= time_threshold, 1, 0))
    num_conc_high_noise = np.sum(np.where(sorted_concentratiosn <= conc_threshold, 1, 0))
    num_noise_high = num_timepoints_high * num_conc_high_noise * num_replicates
    num_noise_low = num_replicates * len(sorted_timepoints) * len(sorted_concentratiosn) - num_noise_high
    for p in np.arange(1, max_subpop + 1):
        obj = lambda x: objective(p,
                                  x,
                                  measurements,
                                  sorted_concentratiosn,
                                  sorted_timepoints,
                                  num_replicates,
                                  model,
                                  num_timepoints_high,
                                  num_conc_high_noise,
                                  num_noise_high,
                                  num_noise_low)

        # Changing the format of chosen bounds to fit the optimization procedure:
        if p > 1:
            bnds = [(0.0, 0.5) for _ in np.arange(p - 1)]
        else:
            bnds = []
        if model == "expo":
            if bounds_model.keys() == bounds_model.keys():
                for _ in np.arange(p):
                    bnds.append(bounds_model['alpha'])
                    bnds.append(bounds_model['b'])
                    bnds.append(bounds_model['E'])
                    bnds.append(bounds_model['n'])
            else:
                print("Error: some parameters are missing in the bounds dictionary.")
                print("Consider updating the bounds or choosing a different model.")
                return -1

        bnds.append(bounds_sigma_high)
        bnds.append(bounds_sigma_low)
        bnds = tuple(bnds)

        # Optimization:
        fval = np.inf
        lb = [bnds[i][0] for i in range(len(bnds))]
        ub = [bnds[i][1] for i in range(len(bnds))]
        for n in np.arange(num_optim):
            print(f'Number of populations: {p}, opimization index: {n}')
            x0 = np.random.uniform(lb, ub)
            result = minimize(obj, x0, method='L-BFGS-B', bounds=bnds,
                              options={'disp': False, 'ftol': 1e-12})
            xx = result['x']
            ff = result['fun']

            if ff < fval:
                fval = ff
                x_final = xx
        x_final_all.append(x_final)
        fval_all.append(fval)
        bic_temp = len(bnds) * np.log(num_datapoints) + 2 * fval

        # Choosing the model with the smallest BIC:
        if bic_temp < bic:
            bic = bic_temp
            final_pop_idx = p

    # Results:
    x_final = x_final_all[final_pop_idx - 1]
    fval = fval_all[final_pop_idx - 1]
    print("Estimated number of cell populations: ", final_pop_idx)
    print("Minimal log-likelihood value found: ", fval)
    MixP = list(x_final[0:(final_pop_idx - 1)])
    MixP.append(1 - np.sum(MixP))
    print("Mixture parameter(s): ", 1 if final_pop_idx == 1 else MixP)
    if model == "expo":
        for i in range(final_pop_idx):
            print("Model parameters for subpopulation #%s:" % (i + 1))
            print("alpha : ", x_final[final_pop_idx - 1 + len(bounds_model) * i])
            print("b : ", x_final[final_pop_idx - 1 + len(bounds_model) * i + 1])
            print("E : ", x_final[final_pop_idx - 1 + len(bounds_model) * i + 2])
            print("n : ", x_final[final_pop_idx - 1 + len(bounds_model) * i + 3])
        print("Sigma high:", x_final[-2])
        print("Sigma low:", x_final[-1])
    else:
        raise NotImplementedError

    # Visualization:
    plt.figure(figsize=(10, 8))
    for i in range(final_pop_idx):
        param = x_final[4 * i + final_pop_idx - 1:4 * i + final_pop_idx + 3]
        plt.semilogx(concentrations, [rate_expo(param, x) for x in sorted(concentrations)], '-*', linewidth=3,
                     label="Subpopulation #%s" % (i + 1))

    plt.xlabel('Drug Concentration')
    plt.ylabel('Growth rate')
    plt.title('Estimated growth rates ')
    plt.legend()

    return x_final
