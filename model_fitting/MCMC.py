#python 3.10
"""MCMC script with differents methods of parameters inferences and scoring methods"""

import distribution as dis
import numpy as np
from scipy.stats import gamma
from launch import *
import matplotlib.pylab as plt

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
    
    # Compute the inverse of the covariance matrix of known variables
    sigma_other_inv = np.linalg.inv(sigma_other))

    
    # Compute the conditional mean
    conditional_mean = mu_current + sigma_sigma_current_otherx_yz @ sigma_other_inv @ (known_values - mu_other)
    
    # Compute the conditional variance
    conditional_variance = sigma_curent - sigma_current_other @ sigma_other_inv @ sigma_x_yz.T
    
    return conditional_mean, conditional_variance
 
def gibbs_sampler(parameters,prior_distribution):
    #list of parameter values
    values = [parameters[p][0] for p in parameters.keys()]
    #list of the paramters mean
    mu = [parameters[p][1] for p in parameters.keys()]
    #list of paramters idx
    para_idx = range(len(parameters))
    #TO DO COV 

    for idx in para_idx: 
        #get a list of other parameters values
        other_values = values[:idx] + values[idx+1:]
        #get a list of other parameter ixd
        other_idx = para_idx[:idx] + mylist[idx+1:]
        conditional_mean, conditional_variance = compute_conditional_params(mu, cov, idx, other_idx, np.array(other_values))
        current = np.random.normal(conditional_mean, np.sqrt(conditional_variance))
        #replace the value of the current parameter by its new one      
        values[p]= current
    
    return np.array(values)

def metropolis_hasting(current, prior_distribution, mu, sigma):
    """
    Metropolis-Hastings algorithm

    Parameters
    ----------
    currnet : float
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
    count = 0
    u = np.log(np.random.uniform(0, 1))
    acceptance_crit = 1
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
    return(proposal)

def mcmc(parameters, data, observations, prior_distribution, approach, N = 1000, burn_in = 0.2, **kwargs):
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
    for p in parameters.keys():
        sample[p] = [parameters[p][0]]
    for i in range(N):  # Generate N samples
          # List to store samples for the current parameter
        if approach == "metropolis_hasting":
            for p in parameters.keys():  # Loop over each parameter
                # Use Metropolis-Hastings algorithm to sample the parameter
                parameters[p][0] = metropolis_hasting(parameters[p][0], prior_distribution, parameters[p][1], parameters[p][2])
        elif approach == "gibbs":

        if i >= idx_burn_in:  # After the burn-in period
            sample[p].append(parameters[p][0])   # Store the current parameter sample)
    return sample



if __name__ == '__main__':
    
    kwargs = {"kasim":"/Tools/KappaTools-master/bin/KaSim",
            "time" : 1000,
            "input_file":"/home/palantir/Post_doc/KaMuI/model_fitting/toy_model.ka",
            "output_file": "/home/palantir/Post_doc/KaMuI/model_fitting/tests/",
            "log_folder": "/home/palantir/Post_doc/KaMuI/model_fitting/tests/logs",
            "nb_para_job":4,
            "repeat":2 }
            # %var: 'on_rate' 1.0E-3 // per molecule per second
            # %var: 'off_rate' 0.1 // per second
            # %var: 'mod_rate' 1 // per second
    parameters = {'off_rate' : [9,9,0.5],
                  'on_rate' : [1e-3, 1e-3, 3e-3],
                  'mod_rate' :[70,70,7]}
    data = []
    observations = []
    test1 = mcmc(parameters, data, observations, "gamma", "metropolis_hasting", 100, 0.2, **kwargs)
    print(test1)
    for key in parameters.keys():
        t =  [x for x in range(1, len(test1[key])+1)]
        plt.plot(t,test1[key])
        plt.show()

        plt.close()
                    # parameter = {p : parameters[p][0]}
                # parallelized_launch(kwargs['kasim'], 
                                    # kwargs['time'], 
                                    # parameter, 
                                    # kwargs['input_file'],
                                    # kwargs['output_file'],
                                    # kwargs['log_folder'],
                                    # kwargs['nb_para_job'],
                                    # kwargs['repeat'])