#python 3.10
"""MCMC script with differents methods of parameters inferences and scoring methods"""

import distribution as dis
import numpy as np
import pandas as pd



    

def sampling(theta, other_theta, model_distri):
    """Sampling of parameter for Gibbs sampler"""
    # mean_other_theta = np.mean(other_theta)
    # var_other_theta = np.var(other_theta)
    # precision_other_theta = 1 / var_other_theta

    #new_theta = np.random.normal(mean_other_theta + precision_other_theta * (y - 2 * x), np.sqrt(1 / (len(y) * precision_other_theta + 1)))


def gibbs_sampler(parameters, model_distri):
    """Gibbs sampler
    
    Parameter
    ---------
    
    Parameters: list, list of the parameters estimated
    
    Model_distri: XXX, distribution of the model used to infer parameters
    
    """
    for p in range (0,len(parameters)):
        if p == 0:
            other_p = parameters[:p]
        if p == len(parameters):
            other_p = parameters[p:]
        else:
            other_p = parameters[p+1:]+parameters[:p]    
        p = sampling(parameters[p], other_p, model_distri)
                      
#estimer chaque paramètre en fonction des autres
    
#Définir équation d'échantillonage

def metropolis_hasting(parameters, prior_distribution, N=10000, burn_in=0.2, **kwargs):
    """
    Metropolis-Hastings algorithm
    
    Parameters
    ----------
    parameters : list
        List of the parameters to be estimated.
    
    prior_distribution : string
        Target PDF (Probability Density Function).
    
    N : int, optional
        Number of samples to generate. Default is 10000.
    
    burn_in : float, optional
        Proportion of proposal samples to ignore to allow the algorithm to converge.
        This fraction of the initial samples will be discarded. Default is 0.2.
    
    **kwargs : dict
        Additional keyword arguments for specifying parameters of the prior distribution.
        Possible keys include:
        - sigma : float, standard deviation for normal distribution
        - mu : float, mean for normal distribution
        - k : float, shape parameter for gamma distribution
        - theta : float, scale parameter for gamma distribution
    """

    
        
    samples = []  # List to store samples for the current parameter
    idx_burn_in = int(burn_in * N)  # Index to start considering samples after burn-in period
    current = p # Initialize the current value of the parameter

    for i in range(N):  # Generate N samples
        
        # Propose a new sample based on the prior distribution
        proposal = dis.prior_distribution_MH(prior_distribution, current, kwargs['sigma'])
        
        # Calculate the acceptance criterion based on the prior distribution type
        if prior_distribution == "normal":
            acceptance_crit = dis.acceptance_criterion_norm(proposal, current, kwargs['mu'], kwargs['sigma'])
        elif prior_distribution == "gamma":
            acceptance_crit = dis.acceptance_criterion_gamma(proposal, current, kwargs['k'], kwargs['theta'])
        
        # Generate a random number between 0 and 1
        u = np.random.uniform(0, 1)
        
        # Accept the proposal with probability equal to the acceptance criterion
        if u < acceptance_crit:
            current = proposal
        
        # Store the sample after burn-in period
        if i >= idx_burn_in:
            samples.append(current)
    
    # Store the samples for the current parameter
    all_samples.append(samples)
    


def mcmc(n, parameter, data, approach = "metropolis_hasting", **kwargs):
    """MCMC algorithm
    
    Parameter
    ---------
    
    Parameter: list, list of the parameters estimated"""

    all_samples = []  # List to store all samples for each parameter
    
    for p in parameters:  # Loop over each parameter

        if approach == "metropolis_hasting":
            samples = metropolis_hasting(parameters, prior_distribution, N=10000, burn_in=0.2, **kwargs)
    return all_samples
