#python 3.10
"""MCMC script with differents methods of parameters inferences and scoring methods"""

import distribution as dis
import numpy as np


def gibbs_sampler(X, y, theta_other, p, distribution):
    """
    Samples a parameter theta_i from its conditional distribution given other parameters.

    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix.
    y : numpy.ndarray
        Target values vector.
    theta_other : numpy.ndarray
        Vector of other parameters.
    p : int
        Index of the parameter to sample.

    Returns:
    --------
    numpy.ndarray
        A new sample of theta_i.
    """
    # Select the p-th column of the design matrix
    X_p = X[:, p:p+1]
    # Get the other parameters
    X_other = np.delete(X, p, axis=1)
    # Delete parameter p from the vector of other parameters
    theta_other = np.delete(theta_other, p)

    if distribution == "normal_multi":
        # Compute the conditional mean of theta_p
        mu_p = np.linalg.solve(X_p.T @ X_p, X_p.T @ (y - X_other @ theta_other))
        # Compute the conditional covariance of theta_p
        sigma_p = np.linalg.inv(X_p.T @ X_p)

        # Sample theta_p from the multivariate normal distribution
        return (np.random.multivariate_normal(mu_p.flatten(), sigma_p))

    # elif distribution == "gamma":


def metropolis_hasting(current, prior_distribution, **kwargs):
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

    **kwargs : dict
        Additional keyword arguments for specifying parameters of the prior distribution.
        Possible keys include:
        - sigma : float, standard deviation for normal distribution
        - mu : float, mean for normal distribution
        - k : float, shape parameter for gamma distribution
        - theta : float, scale parameter for gamma distribution
    """


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

    return(current)

def mcmc(parameters, data, observations, prior_distribution, approach="metropolis_hasting", N = 1000, burn_in=0.2, **kwargs):
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
        - k : float, shape parameter for gamma distribution
        - theta : float, scale parameter for gamma distribution

    Returns:
    --------
    list
        List of samples for each parameter.
    """
    all_samples = []  # List to store all samples for each parameter
    idx_burn_in = int(burn_in * N)  # Index to start considering samples after burn-in period



    for i in range(N):  # Generate N samples
        for p in range(parameters):  # Loop over each parameter
            sample = []  # List to store samples for the current parameter

            if approach == "metropolis_hasting":
                # Use Metropolis-Hastings algorithm to sample the parameter
                parameters[p] = metropolis_hasting(p, prior_distribution, **kwargs)

            elif approach == "gibbs":
                # Use Gibbs sampling algorithm to sample the parameter
                len_ob = len(observations)  # Number of observations
                len_p = len(parameters)  # Number of parameters
                X = np.random.randn(len_ob, len_p)  # Design matrix
                theta_current = parameters
                y = X @ data  # Simulate data for Gibbs sampling
                parameters[p] = gibbs_sampler(X, y, theta_current, p, prior_distribution)  # Gibbs sampling step

            if i >= idx_burn_in:  # After the burn-in period
                sample.append(parameters)  # Store the current parameter sample

        # Store the samples for the current parameter
        all_samples.append(sample)

    return all_samples
