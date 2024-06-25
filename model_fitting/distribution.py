#python 3.10
"""Mathematical equations and function for different distrubtion"""


import numpy as np
from scipy.stats import gamma



    

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
        return round(np.random.normal(theta, sigma), 2)
    elif distribution == "gamma":
        return round(np.random.gamma(theta, sigma), 2)

def acceptance_criterion_norm(proposed, current, mu, sigma=1):
    """Acceptance criterion for Metropolis-hasting define by the normal distribution
    
    Parameter
    ---------
    
    proposed: float, proposed value for the estimated parameter

    current: float, current value for the estimated paramter

    mu: float, mean of the wanted distribution

    sigma: float, std of the wanted distribution
    """

    return (np.log(np.exp(-0.5 * ((proposed - mu)**2 - (current - mu)**2) / (sigma**2))))

def acceptance_criterion_gamma(proposed, current, k, theta):
    """Acceptance criterion for Metropolis-hasting define by the gamma distribution
    #TODO NOT FINISH, REACH EASILY FLOAT MAX VALUE, NEED CORRECTION
    Parameter
    ---------
    
    proposed: float, proposed value for the estimated parameter

    current: float, current value for the estimated paramter

    k : float, shape parameter of the wanted distribution

    theta: float, scale parameter of the wanted distribution

    """  
    if ((k  != 0) & (proposed != 0))):

        # a = (proposed**(k-1)*np.exp((-proposed)/theta))/(current**(k-1)*np.exp((-current)/theta))
        # return min(1,np.log(a))
        #return min(1,(np.log(round(np.exp(-(proposed / theta) - (current / theta)) * proposed**(-1 + k) * current**(1 - k)))))
        alpha = (gamma.logpdf(proposed, k, scale=1.0/theta) /
                 gamma.logpdf(current, k, scale=1.0/theta))
        return min(1,alpha)
        
    else:
        return 0
    


