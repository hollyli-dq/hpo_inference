#!/usr/bin/env python3
"""
Fix for sigma_beta parameter in mcmc_simulation_hpo_k

The mcmc_simulation_hpo_k function expects sigma_beta to be an array
with shape matching the number of covariates (X.shape[0]),
but it's often passed as a scalar value.

This script demonstrates the proper way to convert the scalar to an array.
"""

import numpy as np

# Original problematic code:
# mcmc_results_k = mcmc_simulation_hpo_k(
#     num_iterations=num_iterations,
#     M0=items,
#     assessors=assessors,
#     M_a_dict=M_a_dict,
#     O_a_i_dict=O_a_i_dict,
#     observed_orders=y_a_i_dict,
#     sigma_beta=sigma_beta,    # This is a scalar!
#     X=X,  
#     dr=dr,
#     drrt=drrt,
#     ...
# )

# Solution:
# When sigma_beta is a scalar, convert it to an array with the right dimensions
# X.shape[0] is the number of covariates (p)

def fix_sigma_beta_parameter(sigma_beta, X):
    """
    Convert scalar sigma_beta to appropriate array if needed
    
    Parameters:
    -----------
    sigma_beta : float or ndarray
        Standard deviation for beta prior
    X : ndarray
        Covariate matrix with shape (p, n)
    
    Returns:
    --------
    ndarray
        sigma_beta as a properly shaped array
    """
    # If sigma_beta is already an array of the right shape, return it
    if isinstance(sigma_beta, np.ndarray) and sigma_beta.shape[0] == X.shape[0]:
        return sigma_beta
    
    # If it's a scalar or wrong shape, create array of the right size
    p = X.shape[0]  # Number of covariates
    return np.ones(p) * sigma_beta

# Example usage in notebook:
# 
# # Define scalar sigma_beta
# sigma_beta = 0.5
# 
# # Convert to proper array before passing to function
# sigma_beta_array = fix_sigma_beta_parameter(sigma_beta, X)
# 
# # Use the array in the function call
# mcmc_results_k = mcmc_simulation_hpo_k(
#     num_iterations=num_iterations,
#     M0=items,
#     assessors=assessors,
#     M_a_dict=M_a_dict,
#     O_a_i_dict=O_a_i_dict,
#     observed_orders=y_a_i_dict,
#     sigma_beta=sigma_beta_array,  # Now correctly shaped array
#     X=X,
#     ...
# )

print("Add this function to your notebook and use it to convert scalar sigma_beta to an array")
print("sigma_beta_array = fix_sigma_beta_parameter(sigma_beta, X)") 