#python 3.10
"""scoring script with likelihood, aic, weighted aic calculation"""

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

def likelihood(exp_data, sim_data):
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

    # term1 = n * m * np.log(2 * np.pi)
    # Term: sum over i and j
    # each validation point
    log_likelihood_value = 0
    for name in sim_data.keys():
        # prediction_error = [i -j for i, j in zip(exp_data[name], sim_data[name])]
        prediction_error = np.array(exp_data[name]) - np.array(sim_data[name])
        error_variance = np.std(prediction_error)
        mean_error = np.mean(prediction_error)
        if error_variance == 0:
            error_variance = 10e-6
        term1 = len(exp_data) * len(exp_data[name]) * np.log(2 * np.pi)
        term2 = (mean_error / error_variance )**2 + 2 * np.log(error_variance)
        # Combine the terms to get -2 log(L)
        # get a -2log(L) for each exp observation to normalized it by the number of replicats
        log_likelihood_value += ((term1 + term2)/len(exp_data[name]))
    return log_likelihood_value

def weighted_aic (aic_values):
    """Weight the AICc value of each model

    Parameters:
    -----------
    aic_values: list,
        List of AICc values of each model

    Returns:
    --------
    w_aicc: list,
        list of the weighted AICc value of each model
    """
    d_aicc = np.array(aic_values) - np.min(aic_values)
    w_aicc = np.exp(-0.5*d_aicc)/np.sum(np.exp(-0.5*d_aicc))
    return(w_aicc)

def score_calc(df, parameters, replicat ):
    """
    Calculate the score for each set of parameters based on the simulated data, 
    comparing it to experimental observations and determining the best parameters using AIC.

    Parameters:
    -----------
    df: pandas DataFrame
        A dataframe containing simulated experiment results, time points, and parameter values.
    parameters: dict of dict
        A dictionary containing experimental observations (keys) and their respective parameter 
        values (sub-dictionaries) that were used in the simulation.
    replicat: int
        The number of replicates used in the simulation, which will be needed for calculating the AIC.

    Returns:
    --------
    best_val: dict
        A dictionary containing the parameter names and their respective values for the model 
        that has the best (lowest) weighted AIC score.
    best_aic: float
        The AIC score of the best model.
    """

    aic = {}
    exp_data = {}
    # Loop over all the experiment IDs in the 'exp_sim' column of the dataframe
    for exp_id in enumerate(df['exp_sim']):
        # Loop over the time points associated with each experiment in the dataframe
        for time in enumerate(df.loc[df['exp_sim'] == exp_id[1], '[T]']):
            sim_exp = {}
            for obs in parameters.keys():
                # Extract the list of simulated values for the current observable and time point
                sim_exp[obs[0]] = df.loc[(df['exp_sim'] == exp_id[1]) & (df['[T]'] == time[1]), obs[0]].tolist()
                # Store the corresponding experimental data for the current observable
                exp_data[obs[0]] = list(obs[1])
        ll = likelihood(exp_data, sim_exp)
        aic[exp_id[1]]= aic_c(ll,replicat, len(parameters))
        
    # Find the experiment ID with the best (lowest) weighted AIC score
    b_aic = weighted_aic(list(aic.values()))
    w_aic = dict(zip(list(aic.keys()), b_aic))

    best_id = max(w_aic, key=w_aic.get)
    best_aic = aic[best_id]
    best_val = {}
    # Loop through the observables and parameters to store the best parameter values
    for obs in parameters:
        for para in parameters[obs]:
            best_val[para] = df.loc[df['exp_sim'] == best_id, para].tolist()[0]
    return(best_val,best_aic)