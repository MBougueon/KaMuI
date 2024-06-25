import numpy as np

def compute_conditional_params(mu, cov, idx, known_indices, known_values):
    """
    Compute the conditional mean and variance for the idx-th variable given known values.
    :param mu: Mean vector of the multivariate normal distribution.
    :param cov: Covariance matrix of the multivariate normal distribution.
    :param idx: Index of the variable to sample.
    :param known_indices: Indices of the known variables.
    :param known_values: Values of the known variables.
    :return: Conditional mean and variance.
    """
    # Extract the relevant submatrices and vectors
    mu_current = mu[idx]

    mu_other = mu[known_indices]
    sigma_x2 = cov[idx, idx]
    sigma_yz2 = cov[np.ix_(known_indices, known_indices)]
    sigma_x_yz = cov[idx, known_indices]

    
    # Compute the inverse of the covariance matrix of known variables
    sigma_yz2_inv = np.linalg.inv(sigma_yz2)

    
    # Compute the conditional mean
    conditional_mean = mu_current + sigma_x_yz @ sigma_yz2_inv @ (known_values - mu_other)
    
    # Compute the conditional variance
    conditional_variance = sigma_x2 - sigma_x_yz @ sigma_yz2_inv @ sigma_x_yz
    
    return conditional_mean, conditional_variance

def sample_given_other(mu, cov, y, z, idx):
    """
    Sample from the conditional distribution of X given Y and Z.
    """
    conditional_mean, conditional_variance = compute_conditional_params(mu, cov, idx, [1, 2], np.array([y, z]))
    return np.random.normal(conditional_mean, np.sqrt(conditional_variance))

def gibbs_sampler(num_iterations, initial_values, mu, cov):
    x, y, z = initial_values
    samples = []
    
    for i in range(num_iterations):
        x = sample_given_other(mu, cov, y, z, i)
        y = sample_given_other(mu, cov, x, z, i)
        z = sample_given_other(mu, cov, x, y, i)
        samples.append((x, y, z))
    
    return np.array(samples)

def main():
    num_iterations = 1
    initial_values = (0, 0, 0)  # Initial values for X, Y, Z
    
    # Mean vector
    mu = np.array([0, 0, 0])
    
    # Covariance matrix
    cov = np.array([[1, 0.5, 0.2],
                    [0.5, 2, 0.1],
                    [0.2, 0.1, 3]])
    
    samples = gibbs_sampler(num_iterations, initial_values, mu, cov)
    
    print("Mean of samples for X:", np.mean(samples[:, 0]))
    print("Mean of samples for Y:", np.mean(samples[:, 1]))
    print("Mean of samples for Z:", np.mean(samples[:, 2]))
    print("Std of samples for X:", np.std(samples[:, 0]))
    print("Std of samples for Y:", np.std(samples[:, 1]))
    print("Std of samples for Z:", np.std(samples[:, 2]))

    mylist = [0, 1, 2, 3,6, 4, 5]
    x = 4
    print(mylist[:x] + mylist[x+1:])

if __name__ == "__main__":
    main()
