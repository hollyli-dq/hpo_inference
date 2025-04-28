#!/usr/bin/env python
"""
Core functionality for Bayesian Partial Order Planning.
Implements single-agent and multi-agent partial order planning
using hierarchical partial orders and Bayesian inference.
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional, Union, Any

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import utility functions from existing modules
from src.utils.po_fun import ConversionUtils, GenerationUtils, BasicUtils, StatisticalUtils
from src.utils.po_accelerator_nle import LogLikelihoodCache, HPO_LogLikelihoodCache
from src.mcmc.mcmc_simulation import mcmc_simulation_hpo_k

def mcmc_simulation_bayesian_pop(
    num_iterations: int,
    actions: List[str],
    mandatory_arcs: List[Tuple[int, int]],
    observed_orders: List[List[int]],
    success_outcomes: List[List[bool]],
    X: Optional[np.ndarray] = None,
    K: int = 2,
    noise_option: str = "queue_jump",
    sigma_beta: Union[float, np.ndarray] = 1.0,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run MCMC simulation for Bayesian Partial Order Planning.
    
    Args:
        num_iterations: Number of MCMC iterations
        actions: List of action names
        mandatory_arcs: List of tuples (i, j) representing mandatory ordering constraints
        observed_orders: List of execution sequences (each a list of action indices)
        success_outcomes: List of boolean lists indicating success/failure for each action
        X: Optional feature matrix for actions (n_actions x n_features)
        K: Number of latent dimensions
        noise_option: Type of noise to use in MCMC
        sigma_beta: Standard deviation for beta parameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing inferred partial order, constraints, and success probabilities
    """
    n = len(actions)
    
    # Prepare feature matrix if provided
    if X is None:
        X = np.ones((n, 1))  # Default to intercept-only model
    
    # Initialize mandatory constraint matrix
    H_mandatory = np.zeros((n, n))
    for i, j in mandatory_arcs:
        H_mandatory[i, j] = 1
    H_mandatory = BasicUtils.transitive_closure(H_mandatory)
    
    # Prepare observed data for MCMC
    all_flat_orders = []
    all_outcomes = []
    for seq, outcomes in zip(observed_orders, success_outcomes):
        all_flat_orders.append(seq)
        all_outcomes.append(outcomes)
    
    # Run MCMC simulation using the existing function
    mcmc_results = mcmc_simulation_hpo_k(
        Y=all_flat_orders,
        outcomes=all_outcomes,
        X=X,
        num_iter=num_iterations,
        n=n,
        K=K,
        H_mandatory=H_mandatory,
        sigma_beta=sigma_beta,
        noise_option=noise_option,
        random_seed=random_seed
    )
    
    # Extract results
    H_final = mcmc_results["H_samples"][-1]
    H_reduced = BasicUtils.transitive_reduction(H_final)
    
    # Extract inferred constraints
    inferred_constraints = []
    for i in range(n):
        for j in range(n):
            if H_reduced[i, j] == 1:
                inferred_constraints.append((i, j))
    
    # Extract alpha and beta
    alpha = mcmc_results["alpha_samples"][-1]
    beta = mcmc_results["beta_samples"][-1]
    
    # Calculate success probabilities for each action
    action_success_probs = {}
    for i, action in enumerate(actions):
        logit = alpha + np.dot(X[i], beta)
        prob = 1 / (1 + np.exp(-logit))
        action_success_probs[action] = prob
    
    return {
        "H_final": H_final,
        "inferred_constraints": inferred_constraints,
        "action_success_probs": action_success_probs,
        "alpha": alpha,
        "beta": beta
    }

def multi_agent_bayesian_pop(
    num_iterations: int,
    agents: List[str],
    agent_actions: Dict[str, List[str]],
    agent_mandatory_arcs: Dict[str, List[Tuple[int, int]]],
    cross_agent_arcs: Optional[List[Tuple[str, int, str, int]]] = None,
    observed_orders: Optional[Dict[str, List[List[int]]]] = None,
    success_outcomes: Optional[Dict[str, List[List[bool]]]] = None,
    X: Optional[Dict[str, np.ndarray]] = None,
    K: int = 2,
    noise_option: str = "queue_jump",
    sigma_beta: Union[float, np.ndarray] = 1.0,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run MCMC simulation for Multi-Agent Bayesian Partial Order Planning.
    
    Args:
        num_iterations: Number of MCMC iterations
        agents: List of agent names
        agent_actions: Dictionary mapping agent names to their action lists
        agent_mandatory_arcs: Dictionary mapping agent names to their mandatory constraint lists
        cross_agent_arcs: List of tuples (agent1, action1_idx, agent2, action2_idx) for cross-agent constraints
        observed_orders: Dictionary mapping agent names to their observed execution sequences
        success_outcomes: Dictionary mapping agent names to their action success/failure outcomes
        X: Dictionary mapping agent names to their feature matrices
        K: Number of latent dimensions
        noise_option: Type of noise to use in MCMC
        sigma_beta: Standard deviation for beta parameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing inferred partial orders, cross-agent constraints, and success probabilities
    """
    if cross_agent_arcs is None:
        cross_agent_arcs = []
    
    if observed_orders is None:
        observed_orders = {agent: [] for agent in agents}
    
    if success_outcomes is None:
        success_outcomes = {agent: [] for agent in agents}
    
    if X is None:
        X = {agent: np.ones((len(agent_actions[agent]), 1)) for agent in agents}
    
    # Run MCMC for each agent separately
    agent_results = {}
    for agent in agents:
        agent_results[agent] = mcmc_simulation_bayesian_pop(
            num_iterations=num_iterations,
            actions=agent_actions[agent],
            mandatory_arcs=agent_mandatory_arcs[agent],
            observed_orders=observed_orders[agent],
            success_outcomes=success_outcomes[agent],
            X=X[agent],
            K=K,
            noise_option=noise_option,
            sigma_beta=sigma_beta,
            random_seed=random_seed
        )
    
    # Process cross-agent constraints
    inferred_cross_agent_constraints = []
    for agent1, action1_idx, agent2, action2_idx in cross_agent_arcs:
        # For now, we just keep the mandatory cross-agent constraints
        # A more sophisticated approach would infer these from data
        inferred_cross_agent_constraints.append((agent1, action1_idx, agent2, action2_idx))
    
    # Compile results
    return {
        "agent_partial_orders": {agent: result["H_final"] for agent, result in agent_results.items()},
        "agent_inferred_constraints": {agent: result["inferred_constraints"] for agent, result in agent_results.items()},
        "cross_agent_constraints": inferred_cross_agent_constraints,
        "agent_success_probs": {agent: result["action_success_probs"] for agent, result in agent_results.items()},
        "agent_alpha": {agent: result["alpha"] for agent, result in agent_results.items()},
        "agent_beta": {agent: result["beta"] for agent, result in agent_results.items()}
    }

def generate_synthetic_observations(
    H: np.ndarray,
    actions: List[str],
    success_probs: Dict[str, float],
    num_observations: int,
    random_seed: Optional[int] = None
) -> Tuple[List[List[int]], List[List[bool]]]:
    """
    Generate synthetic observations of execution sequences and outcomes.
    
    Args:
        H: Partial order matrix
        actions: List of action names
        success_probs: Dictionary mapping action names to success probabilities
        num_observations: Number of observations to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (observed_orders, success_outcomes)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(actions)
    observed_orders = []
    success_outcomes = []
    
    for _ in range(num_observations):
        # Generate a valid linear extension
        linear_ext = StatisticalUtils.sample_linear_extension(H)
        
        # Generate success outcomes
        outcomes = []
        for action_idx in linear_ext:
            action = actions[action_idx]
            success = np.random.random() < success_probs[action]
            outcomes.append(success)
        
        observed_orders.append(linear_ext)
        success_outcomes.append(outcomes)
    
    return observed_orders, success_outcomes

def visualize_partial_order(
    actions: List[str],
    partial_order_matrix: np.ndarray,
    title: str = "Inferred Partial Order"
) -> None:
    """
    Visualize a partial order as a directed graph.
    
    Args:
        actions: List of action names
        partial_order_matrix: Partial order matrix
        title: Title for the plot
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: networkx and matplotlib are required for visualization.")
        return
    
    n = len(actions)
    H_reduced = BasicUtils.transitive_reduction(partial_order_matrix)
    
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, label=actions[i])
    
    for i in range(n):
        for j in range(n):
            if H_reduced[i, j] == 1:
                G.add_edge(i, j)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20)
    
    # Draw labels
    labels = {i: actions[i] for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
def visualize_multi_agent_plan(
    agents: List[str],
    agent_actions: Dict[str, List[str]],
    agent_partial_orders: Dict[str, np.ndarray],
    cross_agent_constraints: List[Tuple[str, int, str, int]],
    title: str = "Multi-Agent Partial Order Plan"
) -> None:
    """
    Visualize a multi-agent plan as a directed graph.
    
    Args:
        agents: List of agent names
        agent_actions: Dictionary mapping agent names to their action lists
        agent_partial_orders: Dictionary mapping agent names to their partial order matrices
        cross_agent_constraints: List of tuples (agent1, action1_idx, agent2, action2_idx)
        title: Title for the plot
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: networkx and matplotlib are required for visualization.")
        return
    
    G = nx.DiGraph()
    
    # Define a color map for agents
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
    agent_colors = {agent: colors[i % len(colors)] for i, agent in enumerate(agents)}
    
    # Add nodes for each agent's actions
    node_mapping = {}  # Maps (agent, action_idx) to node id in the graph
    node_id = 0
    
    for agent in agents:
        for i, action in enumerate(agent_actions[agent]):
            node_mapping[(agent, i)] = node_id
            G.add_node(node_id, label=f"{agent}: {action}", agent=agent)
            node_id += 1
    
    # Add edges for each agent's constraints
    for agent in agents:
        H_reduced = BasicUtils.transitive_reduction(agent_partial_orders[agent])
        n = len(agent_actions[agent])
        
        for i in range(n):
            for j in range(n):
                if H_reduced[i, j] == 1:
                    G.add_edge(
                        node_mapping[(agent, i)],
                        node_mapping[(agent, j)],
                        constraint_type='within_agent'
                    )
    
    # Add edges for cross-agent constraints
    for agent1, action1_idx, agent2, action2_idx in cross_agent_constraints:
        G.add_edge(
            node_mapping[(agent1, action1_idx)],
            node_mapping[(agent2, action2_idx)],
            constraint_type='cross_agent'
        )
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes colored by agent
    for agent in agents:
        agent_nodes = [n for n, attr in G.nodes(data=True) if attr.get('agent') == agent]
        nx.draw_networkx_nodes(G, pos, nodelist=agent_nodes, node_size=700, 
                               node_color=agent_colors[agent], label=agent)
    
    # Draw different edge types
    within_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('constraint_type') == 'within_agent']
    cross_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('constraint_type') == 'cross_agent']
    
    nx.draw_networkx_edges(G, pos, edgelist=within_edges, width=1.5, arrowsize=20)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, width=2.0, arrowsize=20, 
                          edge_color='red', style='dashed')
    
    # Draw labels
    labels = {n: attr['label'] for n, attr in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout() 