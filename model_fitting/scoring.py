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
    aic = log_likelihood + ((2*k*n)/(n-k-1))
    return(aic)

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

    for j in range(m): 
        #distance between exp and simul data
        prediction_error = [exp_data[j] - sim_data[j][l] for l  in range(len(sim_data[j]))]

        error_variance = np.std(prediction_error)
        if error_variance == 0:
            error_variance = 10e-6
        term2 += (prediction_error[j] / error_variance )**2 + 2 * np.log(error_variance)
    
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
    d_aicc = np.asarray([-(aicc - min_aicc) for aicc in aic_values])
    w_aicc = np.exp(-0.5*d_aicc)/np.sum(np.exp(-0.5*d_aicc))
    return(w_aicc)

def score_calc(df, parameters, exp_data, observations ):
    """"""
    """Calculate the score for each paramters
    Parameters:
    -----------
    df: panda dataframe,
        df with sim values, paremters values
    parameters: list,
        name of the parameters
    exp_data: dictionnary
        name and value of the experimental observations
    Returns:
    ------
    """
    aic = {}
    for id in enumerate(df['exp_sim']):
        for time in enumerate(df.loc[df['exp_sim'] == id[1], '[T]']):
            sim_exp = []
            for obs in observations:
                sim_exp.append(df.loc[(df['exp_sim'] == id[1]) & (df['[T]'] == time[1]), obs].tolist())
        L = likelihood(exp_data, sim_exp, len(sim_exp[0]), len(observations))     
        aic[id[1]]= aic_c(L,len(sim_exp[0]), len(parameters))
    print(aic)
    b_aic = weighted_aic (aic.values())
    print(b_aic)



    

    