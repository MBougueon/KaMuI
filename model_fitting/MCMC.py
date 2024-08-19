#python 3.10
"""MCMC script with differents methods of parameters inferences and scoring methods"""

import distribution as dis
import numpy as np
import matplotlib.pylab as plt

from launch import *
from data_extraction import *
from scoring import *

def compute_conditional_params(mu, cov, idx, known_indices, known_values):
    """
    Compute the conditional mean and variance for the idx-th variable given known values.
    Parameters:
    -----------

    mu: numpy.ndarray
        Mean vector of the multivariate normal distribution.
    cov: numpy.ndarray
        Covariance matrix of the multivariate normal distribution.
    idx: int
        Index of the variable to sample.
    known_indices: list
        Indices of the known variables.
    known_values: list
        Values of the known variables.

    Returns:
    --------
    Conditional mean: float
        New conditional mean value
    Conditional variance: float
        New conditional variance value
    """
    # Extract the relevant submatrices and vectors
    mu_current = mu[idx]
    mu_other = mu[known_indices]


    sigma_curent = cov[idx, idx]
    sigma_other = cov[np.ix_(known_indices, known_indices)]
    sigma_current_other = cov[idx, known_indices]

    #Case with only 2 parameters equation must be adapted
    if len(sigma_other) == 1:
        sigma_other = sigma_other[0]
        sigma_current_other = sigma_current_other[0]
        known_values = known_values[0]
        # Compute the inverse of the covariance matrix of known variables
        sigma_other_inv = 1/sigma_other
        # Compute the conditional mean
        conditional_mean = int(mu_current + sigma_current_other * sigma_other_inv * (known_values - mu_other))
        # Compute the conditional variance
        conditional_variance = int(sigma_curent - sigma_current_other * sigma_other_inv * sigma_current_other)

    else:
        sigma_other_inv = np.linalg.inv(sigma_other)
        # Compute the conditional mean
        conditional_mean = mu_current + sigma_current_other @ sigma_other_inv @ (known_values - mu_other)

        # Compute the conditional variance
        conditional_variance = sigma_curent - sigma_current_other @ sigma_other_inv @ sigma_current_other

    return conditional_mean, conditional_variance

def gibbs_sampler(parameters,prior_distribution):
    """Gibbs sampler

    Parameters
    ----------
    parameter : dictionnary
        value of the paramters

    prior_distribution : string
        Target PDF (Probability Density Function)."""
    values = []
    mus = []
    data = []
    if prior_distribution == "normal":
        for p in parameters.keys():
            #list of parameter values
            values.append(parameters[p][0])
            #list of the paramters mean
            mus.append(parameters[p][1])
            #list of values generated for the COV
            if prior_distribution == "normal":
                data.append(np.random.normal(parameters[p][1], parameters[p][2], 10))
            if prior_distribution == "gamma":
                data.append(np.random.gamma(parameters[p][1], parameters[p][2], 10))
        mus = np.array(mus)
        #list of paramters idx, * for unpacking
        para_idx = [*range(0, len(parameters))]
        cov = np.cov(data)
        for idx in para_idx:
            #get a list of other parameters values
            other_values = values[:idx] + values[idx+1:]
            #get a list of other parameter ixd
            other_idx = para_idx[:idx] + para_idx[idx+1:]
            conditional_mean, conditional_variance = compute_conditional_params(mus, cov, idx, other_idx, np.array(other_values))
            current = np.random.normal(conditional_mean, np.sqrt(conditional_variance))
            #replace the value of the current parameter by its new one
            values[idx]= current
    else: 
        print(f"{prior_distribution} not implemented")

    return values

def metropolis_hasting(current, prior_distribution, mu, sigma):
    """
    Metropolis-Hastings algorithm

    Parameters
    ----------
    current : float
        value of the paramters estimated.

    prior_distribution : string
        Target PDF (Probability Density Function).

    N : int, optional
        Number of samples to generate. Default is 10000.

    burn_in : float, optional
        Proportion of proposal samples to ignore to allow the algorithm to converge.
        This fraction of the initial samples will be discarded. Default is 0.2.

    sigma : float, standard deviation for normal distribution or shape parameter for gamma distribution

    mu : float, mean for normal distribution or scale parameter for gamma distribution

    Returns:
    --------
    proposal: float
        proposed new value
    """


    # Propose a new sample based on the prior distribution
    proposal = -0.1
    u = np.log(np.random.uniform(0, 1))
    acceptance_crit = 1
    tries = 0
    # some time the value given can be negative, which is not support by Kappa
    # Loop ensure that the value remaine positive
    while proposal <= 0 & (u > acceptance_crit):
        proposal = dis.prior_distribution_MH(prior_distribution, current, sigma)
    # Calculate the acceptance criterion based on the prior distribution type
        if prior_distribution == "normal":
            acceptance_crit = dis.acceptance_criterion_norm(proposal,
                                                            current,
                                                            mu,
                                                            sigma)
        elif prior_distribution == "gamma":
            acceptance_crit = dis.acceptance_criterion_gamma(proposal,
                                                            current,
                                                            mu,
                                                            sigma)
        # Generate a random number between 0 and 1 on wich a log is applied
        u = np.log(np.random.uniform(0, 1))

    # Accept the proposal with probability equal to the acceptance criterion
    # if u < acceptance_crit:
    #     current = proposal
        tries +=1
    return(proposal, tries)

def mcmc(parameters, prior_distribution, approach, N = 1000, burn_in = 0.2):
    """
    MCMC algorithm to estimate parameters using either Metropolis-Hastings or Gibbs sampling.

    Parameters:
    -----------
    N : int
        Number of samples to generate.
    parameters : list
        List of the parameters to be estimated.
    data : numpy.ndarray
        Data used in the model.
    approach : str, optional
        MCMC approach to use: "metropolis_hasting" or "gibbs". Default is "metropolis_hasting".
    burn_in : float, optional
        Proportion of samples to ignore for the burn-in phase. Default is 0.2.
    **kwargs : dict
        Additional keyword arguments depending on the chosen approach.
        Possible keys include:
        - sigma : float, standard deviation for normal distribution
        - mu : float, mean for normal distribution
        - a : float, shape parameter for gamma distribution
        - b : float, scale parameter for gamma distribution

    Returns:
    --------
    list
        List of samples for each parameter.
    """
    idx_burn_in = int(burn_in * N)  # Index to start considering samples after burn-in period
    sample = {}
    tries = [0]*len(parameters)
    for p in parameters.keys():
        sample[p] = [parameters[p][0]]
    for i in range(N):  # Generate N samples
          # List to store samples for the current parameter
        t = 0
        if approach == "metropolis_hasting":
            for p in parameters.keys():  # Loop over each parameter
                # Use Metropolis-Hastings algorithm to sample the parameter
                parameters[p][0], tri = metropolis_hasting(parameters[p][0], prior_distribution, parameters[p][1], parameters[p][2])
                if i >= idx_burn_in:  # After the burn-in period
                    sample[p].append(parameters[p][0])   # Store the current parameter sample)
                    tries[t] = tries[t]+tri
                t +=1

        elif approach == "gibbs":
            samp = gibbs_sampler(parameters,prior_distribution)
            if i >= idx_burn_in:
                idx = 0
                for p in parameters.keys():
                    sample[p].append(samp[idx])
                    idx +=1

    return sample, tries

def para_inference(N=10, SOME_THRESHOLD = 5, **kwargs):
    """
    Inference of the parameters values by repeated generation of values, simulation and scoring
    to infer the best set of parameter values.

    Parameters
    ----------
    N: int
        Number of iterations.
    SOME_THRESHOLD: int
        Threshold for the number of meters for which an additional simulation may be unnecessary.
        
    **kwargs: dict
        Arbitrary keyword arguments that allow you to pass additional parameters to the function.
        The expected keys in kwargs include:
        - 'parameters': dict
            A dictionary containing the parameters to be inferred. Each key represents a parameter name,
            and its value is typically a list or range that will be refined during the iterations.
        - 'distribution': str
            The type of distribution to be used in the MCMC process (e.g., normal, uniform).
        - 'method': str
            The method to be used for parameter inference (e.g., 'metropolis_hasting').
        - 'Num': int
            Number of samples to be generated in each MCMC iteration.
        - 'burn_in': float
            The proportion of initial samples to discard during the MCMC process to allow the chain to stabilize.
        - 'output_file': str
            The path to the output file where data is stored or from which data is retrieved.
        - Additional keys (used in parallelized_lauch):
            - 'kasim': str
                A path or identifier needed for the simulation.
            - 'time': int
                Simulation time.
            - 'variables_units': dict
                A dictionary mapping variables to their respective units.
            - 'input': str
                Path to the input file for the simulation.
            - 'output': str
                Path to the output directory for the simulation results.
            - 'log': str
                Path to the log file where the simulation logs will be stored.
            - 'nb_jobs': int
                Number of parallel jobs to run during the simulation.
            - 'repetition': int
                Number of repetitions for the simulation.
    """
    # Initialize variables
    parameters = kwargs["parameters"]  
    best_aic = 10e6  # Set an initial high value for the best AIC score, to be minimized.
    limit_reach = 0  # Counter to track how many times the optimization limit has been reached.

    for i in range(N):
        generated_val, tries = mcmc(parameters, kwargs['distribution'], kwargs['method'], kwargs['Num'], kwargs['burn_in'])
        parallelized_launch(kwargs['kasim'],
                        kwargs['time'],
                        generated_val,
                        kwargs['input_file'],
                        kwargs['output_file'],
                        kwargs['log_folder'],
                        kwargs['nb_para_job'],
                        kwargs['repeat'])
        
        parameters_name = list(kwargs["parameters"].keys())
        df = get_data(kwargs['output_file'], parameters_name, [100])
        new_val, aic = score_calc(df, parameters, kwargs['exp_val'], kwargs['repeat'])
        
        # Calculate a weighted AIC score to compare the current and best AIC scores.
        b_aic = weighted_aic([best_aic, aic])
        # If the new AIC score is better, update the parameters with the new values.
        if b_aic[1] == max(b_aic):
            for key in new_val.keys():
                parameters[key][0] = new_val[key]
        else:
            # If the optimal parameter set is reached, increment the limit_reach counter.
            limit_reach +=1

        # If the limit_reach counter exceeds a certain threshold (not provided), further simulation may be unnecessary.
        if limit_reach > SOME_THRESHOLD:  # Note: SOME_THRESHOLD should be defined based on specific requirements.
            break  # Exit the loop if the limit is reached.

if __name__ == '__main__':

    kwargs = {
            "parameters" : {'off_rate' : [4,9,0.5],'on_rate' : [1e-1, 1e-1, 3e-1]},
            "distribution": "normal",
            "method" : "metropolis_hasting",
            "Num" : 100,
            "burn_in" : 0.2,
            "exp_val" : {'AB': [900,905], 'Cpp':[9000,9050]},
            "kasim":"/Tools/KappaTools-master/bin/KaSim",
            "time" : 1000,
            "input_file":"/home/palantir/Post_doc/KaMuI/model_fitting/toy_model.ka",
            "output_file": "/home/palantir/Post_doc/KaMuI/model_fitting/tests/",
            "log_folder": "/home/palantir/Post_doc/KaMuI/model_fitting/tests/logs",
            "nb_para_job":2,
            "repeat":2 
            }
    
    para_inference(**kwargs)
