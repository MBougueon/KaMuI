#python 3.10
"""MCMC script with differents methods of parameters inferences and scoring methods"""

import distribution as dis
import numpy as np
from scipy.stats import gamma
from launch import *


# def gibbs_sampler_V1(X, y, theta_other, p, distribution, **kwargs):
#     """
#     Samples a parameter theta_i from its conditional distribution given other parameters.

#     Parameters:
#     -----------
#     X : numpy.ndarray
#         Design matrix.
#     y : numpy.ndarray
#         Target values vector.
#     theta_other : numpy.ndarray
#         Vector of other parameters.
#     p : int
#         Index of the parameter to sample.
#     distribution : str
#         Type of conditional distribution.
#     **kwargs : dict
#         Additional keyword arguments based on the chosen distribution.
#         - a_0 : float, shape parameter for gamma distribution
#         - b_0 : float, scale parameter for gamma distribution

#     Returns:
#     --------
#     numpy.ndarray
#         A new sample of theta_i.
#     """

#     # Select the p-th column of the design matrix
#     X_p = X[:, p:p+1]

#     # Get the other parameters
#     X_other = np.delete(X, p, axis=1)

#     # Delete parameter p from the vector of other parameters
#     theta_other = np.delete(theta_other, p)

#     if distribution == "normal_multi":
#         # Compute the conditional mean of theta_p
#         mu_p = np.linalg.solve(X_p.T @ X_p, X_p.T @ (y - X_other @ theta_other))

#         # Compute the conditional covariance of theta_p
#         sigma_p = np.linalg.inv(X_p.T @ X_p)

#         # Sample theta_p from the multivariate normal distribution
#         proposed_p =  np.random.multivariate_normal(mu_p.flatten(), sigma_p)

#     elif distribution == "normal":
#         mu_p = np.linalg.solve(X_p.T @ X_p, X_p.T @ (y - X_other @ theta_other))
#         sigma_p = 1 / (X_p.T @ X_p)
#         proposed_p =  np.random.normal(mu_p, np.sqrt(sigma_p))

#     elif distribution == "gamma":
#         # Compute the predicted values using other parameters
#         y_hat = np.exp(X_other @ theta_other)
#         # Compute the shape parameter of the gamma distribution
#         a_p =  kwargs['a0'] + np.sum(y)
#         # Compute the rate parameter of the gamma distribution
#         b_p =  kwargs['b0'] + np.sum(y_hat)
#         proposed_p = gamma.rvs(a_p, scale=1/b_p)
        
#     return(proposed_p)


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
    """


    # Propose a new sample based on the prior distribution
    proposal = -0.1
    # some time the value given can be negative, which is not support by Kappa
    # Loop ensure that the value remaine positive
    while proposal < 0:
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
    if u < acceptance_crit:
        current = proposal

    return(current)

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
    all_samples = []  # List to store all samples for each parameter
    idx_burn_in = int(burn_in * N)  # Index to start considering samples after burn-in period

    for i in range(N):  # Generate N samples
        for p in parameters.keys():  # Loop over each parameter
            sample = []  # List to store samples for the current parameter
            parameter = {p : parameters[p][0]}
            # parallelized_launch(kwargs['kasim'], 
                                # kwargs['time'], 
                                # parameter, 
                                # kwargs['input_file'],
                                # kwargs['output_file'],
                                # kwargs['log_folder'],
                                # kwargs['nb_para_job'],
                                # kwargs['repeat'])

            if approach == "metropolis_hasting":
                # Use Metropolis-Hastings algorithm to sample the parameter

                parameters[p][0] = metropolis_hasting(parameters[p][0], prior_distribution, parameters[p][1], parameters[p][2])

            elif approach == "gibbs":
                len_ob = len(observations)  # Number of observations
                len_p = len(parameters)  # Number of parameters
                X = np.random.randn(len_ob, len_p)  # Design matrix
                y = X @ data  # Simulate data for Gibbs sampling
                parameters[p] = gibbs_sampler_V1(X, y, parameters, p, prior_distribution)  # Gibbs sampling step

            if i >= idx_burn_in:  # After the burn-in period
                sample.append(parameters)  # Store the current parameter sample

        # Store the samples for the current parameter
        all_samples.append(sample)
        print(all_samples)

    return all_samples



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
    parameters = {'off_rate' : [0.1,0.1,0.05],
                  'on_rate' : [1e-3, 1e-3, 5e-4],
                  'mod_rate' :[1,1,0.5]}
    data = []
    observations = []
    test1 = mcmc(parameters, data, observations, "gamma", "metropolis_hasting", 10, 0.2, **kwargs)
    print(test1)