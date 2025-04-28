#!/usr/bin/env python
"""
Multi-agent example of Bayesian Partial Order Planning.

This example demonstrates how to use the multi-agent Bayesian POP implementation
to infer partial orders for multiple agents with cross-agent constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the Bayesian POP implementation
from src.bayesian_pop.bayesian_pop import (
    multi_agent_bayesian_pop,
    visualize_multi_agent_plan
)

def run_multi_agent_example():
    """Run a multi-agent example of Bayesian POP."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define the number of agents and actions per agent
    n_agents = 2
    n_actions_per_agent = [4, 3]  # Agent 1 has 4 actions, Agent 2 has 3 actions
    total_actions = sum(n_actions_per_agent)
    
    # Define action names for each agent
    action_names = [
        # Agent 1 (Nurse)
        ["Check vitals", "Draw blood", "Administer medication", "Update chart"],
        # Agent 2 (Doctor)
        ["Examine patient", "Review labs", "Prescribe treatment"]
    ]
    
    # Generate synthetic observed sequences for each agent
    # In a real application, these would come from real observations
    observed_sequences = [
        # Agent 1 sequences (10 observations)
        [
            [0, 1, 2, 3],  # Check vitals -> Draw blood -> Administer medication -> Update chart
            [0, 2, 1, 3],  # Check vitals -> Administer medication -> Draw blood -> Update chart
            [0, 1, 3, 2],  # Check vitals -> Draw blood -> Update chart -> Administer medication
            [0, 2, 3, 1],  # Check vitals -> Administer medication -> Update chart -> Draw blood
            [0, 1, 2, 3],  # Check vitals -> Draw blood -> Administer medication -> Update chart
            [0, 1, 2, 3],  # Check vitals -> Draw blood -> Administer medication -> Update chart
            [0, 1, 3, 2],  # Check vitals -> Draw blood -> Update chart -> Administer medication
            [0, 2, 3, 1],  # Check vitals -> Administer medication -> Update chart -> Draw blood
            [0, 1, 2, 3],  # Check vitals -> Draw blood -> Administer medication -> Update chart
            [0, 1, 2, 3],  # Check vitals -> Draw blood -> Administer medication -> Update chart
        ],
        # Agent 2 sequences (10 observations)
        [
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
            [0, 2, 1],  # Examine patient -> Prescribe treatment -> Review labs
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
            [0, 2, 1],  # Examine patient -> Prescribe treatment -> Review labs
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
            [0, 1, 2],  # Examine patient -> Review labs -> Prescribe treatment
        ]
    ]
    
    # Define features for each action
    # For simplicity, we'll use one-hot encoding for the actions
    # plus a constant term for the intercept
    X = np.zeros((total_actions, total_actions + 1))
    X[:, 0] = 1  # Intercept term
    for i in range(total_actions):
        X[i, i+1] = 1  # One-hot encoding for each action
    
    # Define cross-agent constraints 
    # These are known constraints between agents that must be respected
    # For example, the doctor must examine the patient (agent 2, action 0)
    # before the nurse can administer medication (agent 1, action 2)
    cross_agent_constraints = [
        {"source": (1, 0), "target": (0, 2)}  # Doctor's action 0 before Nurse's action 2
    ]
    
    # Convert the cross-agent constraints to a format the MCMC can understand
    # This maps the (agent_idx, action_idx) to a global action index
    action_mapping = {}
    current_idx = 0
    for agent_idx, n_actions in enumerate(n_actions_per_agent):
        for action_idx in range(n_actions):
            action_mapping[(agent_idx, action_idx)] = current_idx
            current_idx += 1
    
    # Run MCMC simulation for multi-agent Bayesian POP
    print("Running multi-agent Bayesian POP MCMC simulation...")
    H_list, beta_list, H_mean_list, success_probs_list = multi_agent_bayesian_pop(
        observed_sequences=observed_sequences,
        X=X,
        n_actions_per_agent=n_actions_per_agent,
        num_iterations=2000,
        K=5,
        sigma_beta=1.0,
        cross_agent_constraints=cross_agent_constraints,
        seed=42
    )
    
    # Print the inferred partial orders for each agent
    for agent_idx in range(n_agents):
        print(f"\nAgent {agent_idx+1} inferred partial order (posterior mean):")
        agent_actions = action_names[agent_idx]
        H_mean = H_mean_list[agent_idx]
        success_probs = success_probs_list[agent_idx]
        
        for i in range(n_actions_per_agent[agent_idx]):
            for j in range(n_actions_per_agent[agent_idx]):
                if H_mean[i, j] > 0.5 and i != j:
                    print(f"  {agent_actions[i]} -> {agent_actions[j]} (prob: {H_mean[i, j]:.2f})")
        
        print(f"\nAgent {agent_idx+1} inferred success probabilities:")
        for i in range(n_actions_per_agent[agent_idx]):
            print(f"  {agent_actions[i]}: {success_probs[i]:.2f}")
    
    # Print the inferred cross-agent constraints
    print("\nInferred cross-agent constraints:")
    for constraint in cross_agent_constraints:
        source_agent, source_action = constraint["source"]
        target_agent, target_action = constraint["target"]
        source_name = action_names[source_agent][source_action]
        target_name = action_names[target_agent][target_action]
        print(f"  Agent {source_agent+1}'s '{source_name}' must be performed before")
        print(f"  Agent {target_agent+1}'s '{target_name}'")
    
    # Visualize the multi-agent plan
    visualize_multi_agent_plan(
        H_mean_list, action_names, success_probs_list, 
        cross_agent_constraints,
        title="Multi-Agent Medical Protocol", 
        filename="multi_agent_plan.png"
    )

if __name__ == "__main__":
    run_multi_agent_example() 