#!/usr/bin/env python
"""
Basic example of Bayesian Partial Order Planning.

This example demonstrates how to use the Bayesian POP implementation
to infer partial orders from observed execution sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the Bayesian POP implementation
from src.bayesian_pop.bayesian_pop import (
    generate_synthetic_observations,
    mcmc_simulation_bayesian_pop,
    visualize_partial_order
)

def run_example():
    """Run a basic example of Bayesian POP."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define the number of actions in our domain
    n_actions = 5
    action_names = ["Check patient", "Administer medication", 
                   "Take vital signs", "Request lab tests", "Document findings"]
    
    # Define a true partial order as an adjacency matrix
    # H[i,j] = 1 means action i must be performed before action j
    true_H = np.zeros((n_actions, n_actions))
    
    # Define the partial order constraints
    # - Check patient (0) before Administer medication (1)
    # - Check patient (0) before Request lab tests (3)
    # - Take vital signs (2) before Administer medication (1)
    true_H[0, 1] = 1  # Check patient -> Administer medication
    true_H[0, 3] = 1  # Check patient -> Request lab tests
    true_H[2, 1] = 1  # Take vital signs -> Administer medication
    
    # Define features for each action
    # For simplicity, we'll use one-hot encoding for the actions
    # plus a constant term for the intercept
    X = np.zeros((n_actions, n_actions + 1))
    X[:, 0] = 1  # Intercept term
    for i in range(n_actions):
        X[i, i+1] = 1  # One-hot encoding for each action
    
    # Define true alpha parameters (for generating success probabilities)
    # Higher values mean higher success probability
    true_alpha = np.array([1.0, 0.5, 2.0, 1.5, 0.0, 0.8])
    
    # Generate synthetic observation data
    n_observations = 20
    observed_sequences, success_indicators = generate_synthetic_observations(
        true_H, true_alpha, X, n_observations, seed=42
    )
    
    # Print the observation data
    print("Generated observation sequences:")
    for i, seq in enumerate(observed_sequences):
        # Convert action indices to names
        seq_names = [action_names[action] for action in seq]
        print(f"  {i+1}: {seq_names}")
    
    # Run MCMC simulation to infer the partial order and success probabilities
    print("\nRunning MCMC simulation...")
    H_samples, beta_samples, H_mean, success_probs = mcmc_simulation_bayesian_pop(
        observed_sequences, X, n_actions, num_iterations=2000, K=5, sigma_beta=1.0, seed=42
    )
    
    # Print the inferred partial order
    print("\nInferred partial order (posterior mean):")
    for i in range(n_actions):
        for j in range(n_actions):
            if H_mean[i, j] > 0.5:
                print(f"  {action_names[i]} -> {action_names[j]} (prob: {H_mean[i, j]:.2f})")
    
    # Print the inferred success probabilities
    print("\nInferred success probabilities:")
    for i in range(n_actions):
        print(f"  {action_names[i]}: {success_probs[i]:.2f}")
    
    # Compare with true partial order
    print("\nComparison with true partial order:")
    print("  Edge    | True | Inferred (prob)")
    print("  --------|------|---------------")
    for i in range(n_actions):
        for j in range(n_actions):
            if i != j and (true_H[i, j] > 0 or H_mean[i, j] > 0.1):
                print(f"  {action_names[i]} -> {action_names[j]} | {true_H[i, j]:.0f}    | {H_mean[i, j]:.2f}")
    
    # Visualize the inferred partial order
    visualize_partial_order(
        H_mean, action_names, success_probs,
        title="Inferred Partial Order for Medical Protocol",
        filename="inferred_partial_order.png"
    )

if __name__ == "__main__":
    run_example() 