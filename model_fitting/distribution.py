#python 3.10
"""Mathematical equations and function for different distrubtion"""


import numpy as np



    
# def prior_distribution_gibbs(parameter, mu, sigma):
#     """
#     Propose a new value for a parameter following a uniforme distribution

#     Parameter
#     ---------

#     theta: parameter estimate

#     mu: mean valeur of theta

#     sigma: std of theta

#     Return
#     ------

#     proposed_theta: new value of theta
#     """

#     proposed_theta = np.random.uniform(max(0, mu - sigma), min(mu + sigma, y))

#     return proposed_theta

def prior_distribution_MH(distribution,theta, sigma=1,):
    """
    Propose a new value for a parameter following the wanted distribution

    Parameter
    ---------

    theta: float, parameter estimate

    sigma: float, std of theta

    distribution: string, prior distribution

    Return
    ------

    proposed_theta: float,new value of theta
    """
    if distribution == "normal":
        return np.random.normal(theta, sigma)
    elif distribution == "gamma":
        return np.random.gamma(theta, sigma)

def acceptance_criterion_norm(proposed, current, mu, sigma=1):
    """Acceptance criterion for Metropolis-hasitng define by the normal distribution 
    
    Parameter
    ---------
    
    proposed: float, proposed value for the estimated parameter

    current: float, current value for the estimated paramter

    mu: float, mean of the wanted distribution

    sigma: float, std of the wanted distribution
    """

    return (np.exp(-0.5 * ((proposed - mu)**2 - (current - mu)**2) / (sigma**2)))

def acceptance_criterion_gamma(proposed, current, k, theta):
    """Acceptance criterion for Metropolis-hasitng define by the gamma distribution 
    
    Parameter
    ---------
    
    proposed: float, proposed value for the estimated parameter

    current: float, current value for the estimated paramter

    k: float, shape parameter of the wanted distribution

    theta: float, scale parameter of the wanted distribution

    """  
    return ((proposed / current)**(k - 1) * np.exp(-(proposed - current) / theta))

    


