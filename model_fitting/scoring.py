import numpy as np

def aic_c(log_likelihood, k, n):
    """Corrected Akaike information criterion
    
    Parameters:
    -----------
    log_likelihhod: float, 
        log of the likelihood of the model
    k: int, 
        number of parameters being estimated
    n: int,
        number of samples
    Returns:
    --------
    aic_c: float,
        AICc values of the model
    """
    aic_c = log_likelihood + ((2*k*n)/(n-k-1))

    return(aic_c)

def likelihood(exp_data, sim_data, m, n):
    """Calculate the likelihood of the model given its parameters and the experimental data
    Parameters:
    -----------
    exp_data: list,
        list of the experimental data points
    sim_data: list,
        list of the simulation data points
    m: int, 
        number of measurement points for each samples
    n: int,
        number of samples

    Returns:
    --------
    """

    term1 = n * m * np.log(2 * np.pi)
    
    # Term: sum over i and j
    term2 = 0
    # each validation point
    for i in range(n):
        #distance between exp and simul data
        prediction_error = [exp_data[i] - sim_data[l] for l  in range(m)]
        #each sample
        for j in range(m):        
            error_variance = np.std(prediction_error)
            term2 += (prediction_error[j] ** 2) / (error_variance ** 2) + 2 * np.log(error_variance)
    
    # Combine the terms to get -2 log(L)
    log_likelihood_value = term1 + term2
    return log_likelihood_value

def weighted_aic (aic_values):
    """Weight the AICc value of each model
    
    Parameters:
    -----------
    aic_values: list, 
        List of AICc values of each model
        
    Returns:
    --------
    w: list,
        list of the weighted AICc value of each model
    """
    min_aicc = min(aic_values)
    exp_terms = np.exp([-(aicc - min_aicc) / 2 for aicc in aic_values])

    w = []

    for aic in aic_values:
        w.append(np.exp(-(aic - min_aicc) / 2) / np.sum(exp_terms))
    return w

