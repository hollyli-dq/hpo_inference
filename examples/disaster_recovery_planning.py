#!/usr/bin/env python
"""
Example: Disaster Recovery Planning

This example demonstrates the use of Bayesian Partial Order Planning
for a disaster recovery scenario involving multiple agents.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Bayesian POP modules
from src.bayesian_pop.core import (
    mcmc_simulation_bayesian_pop,
    multi_agent_bayesian_pop,
    generate_synthetic_observations,
    visualize_partial_order,
    visualize_multi_agent_plan
)

def single_agent_example():
    """
    Example of single-agent Bayesian Partial Order Planning
    for a disaster recovery scenario.
    """
    print("\n=== Single-Agent Disaster Recovery Planning ===")
    
    # Define actions for a medical response team
    actions = [
        "assess_casualties",
        "establish_triage",
        "treat_critical_patients",
        "request_medical_supplies",
        "coordinate_evacuations",
        "set_up_field_hospital",
        "treat_moderate_injuries",
        "document_patients"
    ]
    n = len(actions)
    
    # Define known mandatory constraints
    mandatory_arcs = [
        (0, 1),  # assess_casualties -> establish_triage
        (1, 2),  # establish_triage -> treat_critical_patients
        (3, 6),  # request_medical_supplies -> treat_moderate_injuries
    ]
    
    # Create a simple feature matrix for each action
    # Features: [resource_intensive, time_critical]
    X = np.array([
        [0, 1],  # assess_casualties: not resource intensive, time critical
        [1, 1],  # establish_triage: resource intensive, time critical
        [1, 1],  # treat_critical_patients: resource intensive, time critical
        [0, 0],  # request_medical_supplies: not resource intensive, not time critical
        [0, 1],  # coordinate_evacuations: not resource intensive, time critical
        [1, 0],  # set_up_field_hospital: resource intensive, not time critical
        [1, 0],  # treat_moderate_injuries: resource intensive, not time critical
        [0, 0],  # document_patients: not resource intensive, not time critical
    ])
    
    # Define true partial order for generating synthetic data
    H_true = np.zeros((n, n))
    
    # Add mandatory arcs to the true partial order
    for i, j in mandatory_arcs:
        H_true[i, j] = 1
    
    # Add additional constraints to the true partial order
    additional_arcs = [
        (1, 5),  # establish_triage -> set_up_field_hospital
        (2, 7),  # treat_critical_patients -> document_patients
        (5, 6),  # set_up_field_hospital -> treat_moderate_injuries
        (6, 7),  # treat_moderate_injuries -> document_patients
    ]
    
    for i, j in additional_arcs:
        H_true[i, j] = 1
    
    # Define success probabilities for each action
    success_probs = {
        "assess_casualties": 0.95,
        "establish_triage": 0.85,
        "treat_critical_patients": 0.75,
        "request_medical_supplies": 0.90,
        "coordinate_evacuations": 0.80,
        "set_up_field_hospital": 0.70,
        "treat_moderate_injuries": 0.85,
        "document_patients": 0.95
    }
    
    # Generate synthetic observations
    num_observations = 10
    observed_orders, success_outcomes = generate_synthetic_observations(
        H_true, actions, success_probs, num_observations, random_seed=42
    )
    
    print(f"Generated {num_observations} synthetic observations")
    print(f"Example sequence: {[actions[i] for i in observed_orders[0]]}")
    print(f"Example outcomes: {success_outcomes[0]}")
    
    # Run MCMC simulation for Bayesian POP
    num_iterations = 5000
    results = mcmc_simulation_bayesian_pop(
        num_iterations=num_iterations,
        actions=actions,
        mandatory_arcs=mandatory_arcs,
        observed_orders=observed_orders,
        success_outcomes=success_outcomes,
        X=X,
        K=2,
        sigma_beta=1.0,
        random_seed=42
    )
    
    # Extract and display results
    inferred_constraints = results["inferred_constraints"]
    success_probs = results["action_success_probs"]
    
    print("\nInferred constraints:")
    for i, j in inferred_constraints:
        print(f"  {actions[i]} -> {actions[j]}")
    
    print("\nInferred success probabilities:")
    for action, prob in success_probs.items():
        print(f"  {action}: {prob:.2f}")
    
    # Visualize the inferred partial order
    visualize_partial_order(actions, results["H_final"], "Inferred Medical Response Plan")
    plt.savefig("medical_response_plan.png")
    print("Visualization saved as 'medical_response_plan.png'")

def multi_agent_example():
    """
    Example of multi-agent Bayesian Partial Order Planning
    for a disaster recovery scenario.
    """
    print("\n=== Multi-Agent Disaster Recovery Planning ===")
    
    # Define agents
    agents = ["medical_team", "fire_department", "police"]
    
    # Define actions for each agent
    agent_actions = {
        "medical_team": [
            "assess_casualties",
            "establish_triage",
            "treat_critical_patients",
            "set_up_field_hospital"
        ],
        "fire_department": [
            "assess_fire_damage",
            "extinguish_fires",
            "search_for_survivors",
            "clear_debris"
        ],
        "police": [
            "secure_perimeter",
            "direct_traffic",
            "evacuate_civilians",
            "establish_command_post"
        ]
    }
    
    # Define mandatory constraints for each agent
    agent_mandatory_arcs = {
        "medical_team": [
            (0, 1),  # assess_casualties -> establish_triage
            (1, 2),  # establish_triage -> treat_critical_patients
        ],
        "fire_department": [
            (0, 1),  # assess_fire_damage -> extinguish_fires
            (1, 2),  # extinguish_fires -> search_for_survivors
        ],
        "police": [
            (0, 2),  # secure_perimeter -> evacuate_civilians
            (3, 2),  # establish_command_post -> evacuate_civilians
        ]
    }
    
    # Define cross-agent constraints
    cross_agent_arcs = [
        # fire_department:search_for_survivors -> medical_team:assess_casualties
        ("fire_department", 2, "medical_team", 0),
        
        # police:evacuate_civilians -> medical_team:establish_triage
        ("police", 2, "medical_team", 1),
        
        # police:secure_perimeter -> fire_department:assess_fire_damage
        ("police", 0, "fire_department", 0)
    ]
    
    # Create feature matrices for each agent's actions
    # Features: [resource_intensive, time_critical]
    X = {
        "medical_team": np.array([
            [0, 1],  # assess_casualties
            [1, 1],  # establish_triage
            [1, 1],  # treat_critical_patients
            [1, 0],  # set_up_field_hospital
        ]),
        "fire_department": np.array([
            [0, 1],  # assess_fire_damage
            [1, 1],  # extinguish_fires
            [1, 1],  # search_for_survivors
            [1, 0],  # clear_debris
        ]),
        "police": np.array([
            [0, 1],  # secure_perimeter
            [0, 0],  # direct_traffic
            [0, 1],  # evacuate_civilians
            [0, 0],  # establish_command_post
        ])
    }
    
    # Run multi-agent MCMC simulation for Bayesian POP
    num_iterations = 3000
    results = multi_agent_bayesian_pop(
        num_iterations=num_iterations,
        agents=agents,
        agent_actions=agent_actions,
        agent_mandatory_arcs=agent_mandatory_arcs,
        cross_agent_arcs=cross_agent_arcs,
        X=X,
        K=2,
        sigma_beta=1.0,
        random_seed=42
    )
    
    # Display results
    print("\nInferred constraints for each agent:")
    for agent in agents:
        print(f"\n{agent}:")
        for i, j in results["agent_inferred_constraints"][agent]:
            print(f"  {agent_actions[agent][i]} -> {agent_actions[agent][j]}")
    
    print("\nCross-agent constraints:")
    for agent1, action1_idx, agent2, action2_idx in results["cross_agent_constraints"]:
        action1 = agent_actions[agent1][action1_idx]
        action2 = agent_actions[agent2][action2_idx]
        print(f"  {agent1}:{action1} -> {agent2}:{action2}")
    
    print("\nInferred success probabilities:")
    for agent in agents:
        print(f"\n{agent}:")
        for action, prob in results["agent_success_probs"][agent].items():
            print(f"  {action}: {prob:.2f}")
    
    # Visualize the multi-agent plan
    visualize_multi_agent_plan(
        agents=agents,
        agent_actions=agent_actions,
        agent_partial_orders=results["agent_partial_orders"],
        cross_agent_constraints=results["cross_agent_constraints"],
        title="Multi-Agent Disaster Recovery Plan"
    )
    plt.savefig("disaster_recovery_plan.png")
    print("Visualization saved as 'disaster_recovery_plan.png'")

if __name__ == "__main__":
    # Run the examples
    single_agent_example()
    plt.show()
    
    multi_agent_example()
    plt.show() 