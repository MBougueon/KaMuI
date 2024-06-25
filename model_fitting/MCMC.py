#python 3.10
"""MCMC script with differents methods of parameters inferences and scoring methods"""

import distribution as dis
import numpy as np
# from launch import *
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
        #TODO Search for Gamma distribution
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

def mcmc(parameters, prior_distribution, approach, N = 1000, burn_in = 0.2, **kwargs):
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
    # WARNING IF PARAMETER TO LOW, COV = 0
    parameters = {'off_rate' : [4,9,0.5],
                  'on_rate' : [1e-1, 1e-1, 3e-1],
                   'mod_rate' :[10,70,7],
                   'bidule' : [10,10,1]}
    test1, tries = mcmc(parameters, "gamma", "metropolis_hasting", 10000, 0.2, **kwargs)
    print(tries)
    for key in parameters.keys():
        t =  [x for x in range(1, len(test1[key])+1)]
        plt.plot(t,test1[key])
        plt.title(key)
        plt.show()

    #     plt.close()
                    # parameter = {p : parameters[p][0]}
                # parallelized_launch(kwargs['kasim'],
                                    # kwargs['time'],
                                    # parameter,
                                    # kwargs['input_file'],
                                    # kwargs['output_file'],
                                    # kwargs['log_folder'],
                                    # kwargs['nb_para_job'],
                                    # kwargs['repeat'])