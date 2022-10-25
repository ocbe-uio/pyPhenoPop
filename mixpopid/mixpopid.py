import math
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def rateexpo(p, x):
    """
    A function calculating a growth rate of a certain cell population with 
    parameters p exposed to a certain drug with given concentrations according
    to the exponential cell population growth model.
    
    Arguments:
        * p: parameter vector containing 4 parameters (alpha, b, E, n)
        * x: drug concentration vector
        
    """
    
    return (p[0]+np.log(p[1]+ (1-p[1])/(1+(x/p[2])**p[3])))

def popexpo(params, C, T):
    """
    A function calculating a certain cell population count using a function
    'rate' at given time points according to the exponential cell population 
    growth model.
    
    Arguments:
        * p: parameter vector containing 4 parameters (alpha, b, E, n)
        * x: drug concentration vector
        * T: time vector
        
    """
    #Reshaping the time points and the rates vectors to perform a 
    #multiplication:
    T_=np.reshape(T,(1,len(T)))
    rates=np.reshape(rateexpo(params, C),(len(C),1))
    # Returns a matrix with population counts where lines correspond to 
    #concentrations and columns to time points:
    return np.exp(rates@T_)
    


def objective(PopN, Params, DATA, Conc, Time, NR, ConcT, TimeT, model):
    """
    This function calculates the negative value of log-likelihood for a given
    model. Function's minimum is the optimal parameter estimate. 
    
    Function arguments are the following:
        * PopN: number of cell lines considered 
        * Params: model parameters; for every model the first PopN-1 elements of 
        Params must be mixture parameters, the last two are higher and lower 
        variance levels. Higher variance appears at concentration levels
        smaller than ConcT and time points after TimeT
        * DATA: cell count at each independent variable condition. DATA should 
        be structured as follows: DATA[k][j][i] ia cell count for the k-th 
        replicate at the j-th concentration at the i-th time point
        * concvec: list of concentrations considered  measured in micromoles
        * timevec: list of time points measured in hours
        * NR: number of replicates
        * model: a string representing a cell population growth model 
        considered, for example "expo" means exponential model. This variable 
        determines a form of the parameter vector and a function calculating 
        cell population number.
    
    """
    
    #Creating a mixture parameter vector:
    if PopN>1:
        X1=Params[0:PopN-1]
    else:
        X1=[0,]

    #Creating a model specific parameter vector based on the chosen model:
    p=Params[PopN-1:-2]
    if model=="expo":
        if len(p)%4==0:
            PMat=[[p[4*j+i] for i in np.arange(4)] for j in np.arange(PopN)]
        else:
            print("Error: parameter vector not suited for the chosen model")
            return -1
        #Choosing a function calculating cell population number based on the
        #chosen model:
        f=popexpo
    
    #Higher variance level:
    sigH=Params[5*PopN-1]
    
    #Lower variance level:
    sigL=Params[5*PopN]
    
    sum_resid = 0
    
    #The vectors containing time points and concentration levels should be 
    #sorted in order to easily create a matrix with two levels of noise:
    timevec = np.sort(Time)[1:] #time starts from the second value since the 
                                #first one is taken from the data
    concvec = np.sort(Conc)
    
    #Calculation of noise:
    T = sum(np.where(timevec>=TimeT,1,0)) #number of time points with higher
                                        #level of noise
    C = sum(np.where(concvec<=ConcT,1,0)) #number of concentrations with higher
                                        #level of noise
    NH  = T*C*NR #total number of data points with higher level of noise
    NL = NR*len(timevec)*len(concvec)-NH #total number of data points with
                                        #lower level of noise
    #Matrix with noise:
    sigMat = np.ones((len(concvec), len(timevec)))/(2*sigL**2)
    sigMat[:C,len(timevec)-T:] =1/(2*sigH**2) #higher noise is in the top right
    #corner (time bigger than 48 hours, concentration lower than 0.1)
    
    for k in np.arange(NR):
        s=0
        for l in range(PopN-1):
            X=X1[l]*f(PMat[l],concvec,timevec)
            #We need to multiply DATA[k,:,0] (the initial cell count for all
            #concentrations for k-th replicate) by X, element by element 
            #cocentation wise, DATA[k,:,0] being the same fot every time point 
            #of X, so we form a matrix where DATA[k,:,0] is repeated the same
            #number of times as the number of time points, and than reshaped to
            #correspond to the dimensions of X for element wise multiplication:
            M=np.reshape(np.repeat(DATA[k,:,0],len(timevec)),(len(concvec),len(timevec)))
            s+=np.multiply(M,X) #s is a matrix containing counts summed for 
            #all subpopulation in a corresponding proportion for all time
            #points and concentrations
          
        #Last subpopulation:
        X=(1-sum(X1))*f(PMat[PopN-1],concvec,timevec)
        M=np.reshape(np.repeat(DATA[k,:,0],len(timevec)),(len(concvec),len(timevec)))
        s+=np.multiply(M,X)
        
        #Matrix containing residuals for all time points (except time 0 since 
        #in this case the residual is zero by deffinition) and concentrations:
        resid = DATA[k,:,1:] - s
        
        #Adding noise by multiplying every residual by a corresponding noise
        #from the noise matrix, and calculating the sum for all time points
        #and concentrations:
        sum_resid+=sum(sum(np.multiply(resid**2,sigMat)))
    
    return sum_resid+NH*math.log(sigH)+NL*math.log(sigL)

# Default bounds setting used in exponential model parameter inference :
bounds_expo = {'alpha': (1e-05, 0.05), 'b': (0.5, 1.0), 'E': (1e-05, 10),
               'n': (1e-05, 10)}

        
    


def mixtureID(PopN, DATA, Time, Conc, R, model = "expo", 
              bounds_model = bounds_expo, bounds_sigmaH = (1e-05, 1000.0),
              bounds_sigmaL = (1e-05, 500.0), num_optim = 200):
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
    
    Function arguments are the following:
        * PopN: maximum number of cell subpopulations considered 
        * DATA: cell count at each independent variable condition. DATA should 
        be structured as follows: DATA[k][j][i] ia a cell count for the k-th 
        replicate at the j-th concentration at the i-th time point
        * Time: list of time points measured in hours
        * Conc: list of concentrations considered  measured in micromoles
        * R: number of replicates
    Optional arguments are the following:
        * model: cell population growth model considered, exponential ("expo") 
        by default
        * bounds_model: bounds for parameter inference in a form of a 
        dictionary where keys are model specific parameter names and values are
        tuples (lower and higher bounds); default: bounds_model = bounds_expo
        * bounds_sigmaH: higher sigma bounds used to infer the parameter in a 
        form of a tuple (lower and higher bounds); 
        default: bounds_sigmaH = (1e-05, 1000.0)
        * bounds_sigmaL: lower sigma bounds used to infer the parameter in a 
        form of a tuple (lower and higher bounds); 
        default: bounds_sigmaH = (1e-05, 1000.0)
        * num_optim: number of objective function optimization attempts;
        default: num_optim = 200
        
    """


    #Fixed thresholds for concentration and time in order to choose either
    #higher or lower variance level (sigmaH and sigmaL):
    ConcT = 0.1
    TimeT = 48

    #Number of data points:
    points = len(Conc)*len(Time)*R

    #Initializing Bayesian information criterion:
    BIC = float("inf")

    X_FINAL = []
    FVAL = []

    #Comparing BIC for all models with a number of populations considered from 
    # 1 to PopN in order to find the best fit:
    for p in np.arange(1,PopN+1):

        #Function that will be optimized:
        f = lambda x: objective(p,x,DATA,Conc,Time,R,ConcT,TimeT,model)

        #Changing the format of chosen bounds to fit the optimization procedure:
        if p>1:
            bnds = [(1e-05, 0.5) for i in np.arange(p-1)]
        else:
            bnds = []
        if model=="expo":
            if bounds_model.keys()==bounds_expo.keys():
                for i in np.arange(p):
                    bnds.append(bounds_model['alpha'])
                    bnds.append(bounds_model['b'])
                    bnds.append(bounds_model['E'])
                    bnds.append(bounds_model['n'])
            else:
                print("Error: some parameters are missing in the bounds dictionary.")
                print("Consider updating the bounds or choosing a different model.")
                return -1


        bnds.append(bounds_sigmaH)
        bnds.append(bounds_sigmaL)
        bnds=tuple(bnds)

        
        #Optimization:
        fval=float("inf")

        for n in np.arange(num_optim) :
            x0=[random.random()*(bnds[i][1]-bnds[i][0]) + bnds[i][0] for i in np.arange(len(bnds))]
            F = minimize(f, x0, method='TNC', bounds=bnds, 
                         options={'accuracy':1e-07, 'eps': 1e-05, 
                                  'maxiter': 500, 'disp': True, 'ftol': 1e-03})

            xx=F.__getitem__('x')
            ff=F.__getitem__('fun')

            if ff<fval:
                fval = ff
                x_final = xx
        X_FINAL.append(x_final)
        FVAL.append(fval)
        BIC_temp=len(bnds)*math.log(points)+2*fval

        #Choosing the model with the smallest BIC:
        if BIC_temp<BIC:
            BIC = BIC_temp
            pfinal=p


    #Results:
    x_final = X_FINAL[pfinal-1]
    fval = FVAL[pfinal-1]
    print("Estimated number of cell populations: ", pfinal)
    print("Minimal log-likelihood value found: ", fval)
    MixP=list(x_final[0:(pfinal-1)])
    MixP.append(1-sum(MixP))
    print("Mixture parameter(s): ", 1 if pfinal==1 else MixP)
    if model=="expo":
        for i in range(pfinal):
            print("Model parameters for subpopulation #%s:" %(i+1))
            print("alpha : ", x_final[pfinal-1+len(bounds_model)*i])
            print("b : ", x_final[pfinal-1+len(bounds_model)*i+1])
            print("E : ", x_final[pfinal-1+len(bounds_model)*i+2])
            print("n : ", x_final[pfinal-1+len(bounds_model)*i+3])
            
    
    #Visualization:
    plt.figure(figsize=(10,8))
    for i in range(pfinal):
        param=x_final[4*i+pfinal-1:4*i+pfinal+3]
        plt.plot(range(len(Conc)),[rateexpo(param,x) for x in sorted(Conc)], '-*', linewidth=3, 
                           label="Subpopulation #%s" %(i+1))
    
    plt.xlabel('Log Drug Concentration')
    plt.ylabel('Growth rate')
    plt.title('Estimated growth rates ')
    plt.legend()

    return x_final