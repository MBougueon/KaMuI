#python 3.10
"""Mathematical equations and function for different distrubtion"""


import numpy as np



    

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
        return round(np.random.normal(theta, sigma), 4)
    elif distribution == "gamma":
        return round(np.random.gamma(theta, sigma), 4)

def acceptance_criterion_norm(proposed, current, mu, sigma=1):
    """Acceptance criterion for Metropolis-hasitng define by the normal distribution 
    
    Parameter
    ---------
    
    proposed: float, proposed value for the estimated parameter

    current: float, current value for the estimated paramter

    mu: float, mean of the wanted distribution

    sigma: float, std of the wanted distribution
    """

    return (np.log(np.exp(-0.5 * ((proposed - mu)**2 - (current - mu)**2) / (sigma**2))))

def acceptance_criterion_gamma(proposed, current, a, b):
    """Acceptance criterion for Metropolis-hasitng define by the gamma distribution 
    
    Parameter
    ---------
    
    proposed: float, proposed value for the estimated parameter

    current: float, current value for the estimated paramter

    k: float, shape parameter of the wanted distribution

    theta: float, scale parameter of the wanted distribution

    """  
    return (np.log((proposed / current)**(a - 1) * np.exp(-(proposed - current) / b)))

    


