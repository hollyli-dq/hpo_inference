#!/usr/bin/env python3
"""
Bayesian Partial Order Planning Module

This module provides functions for Bayesian inference over partial order plans,
leveraging the existing MCMC implementation for hierarchical partial orders.
It supports both single-agent and multi-agent planning scenarios.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any

# Import existing utility functions
from src.utils.po_fun import ConversionUtils, BasicUtils
from src.mcmc.hpo_po_hm_mcmc_k import mcmc_simulation_hpo_k

def mcmc_simulation_bayesian_pop(
    num_iterations: int,
    actions: List[str],
    mandatory_arcs: List[Tuple[int, int]],
    observed_orders: List[List[int]],
    success_outcomes: Optional[List[List[bool]]] = None,
    X: Optional[np.ndarray] = None,
    K: int = 2,
    noise_option: str = "queue_jump",
    sigma_beta: Optional[Union[float, np.ndarray]] = 1.0,
    update_probs: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run MCMC simulation for Bayesian Partial Order Planning.
    
    Args:
        num_iterations: Number of MCMC iterations
        actions: List of action names
        mandatory_arcs: List of (source, target) pairs representing mandatory constraints
        observed_orders: List of observed execution orders (each a list of action indices)
        success_outcomes: Optional list of success/failure outcomes for each action in each observed order
        X: Optional feature matrix for actions (num_actions × num_features)
        K: Latent space dimension
        noise_option: Type of noise model ("queue_jump" or "latent_space")
        sigma_beta: Prior standard deviation for beta coefficients
        update_probs: Dictionary of update probabilities for MCMC parameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing MCMC results and inferred partial order
    """
    # Set default update probabilities if not provided
    if update_probs is None:
        update_probs = {
            "H": 0.3,       # Partial order matrix
            "Z": 0.3,       # Latent positions
            "beta": 0.2,    # Regression coefficients
            "kappa": 0.1,   # Noise parameter
            "alpha": 0.1    # Intercept parameter
        }
    
    n_actions = len(actions)
    
    # Create a list of item indices
    item_idx = [{"i": i} for i in range(n_actions)]
    
    # For the basic model, we have a single assessor
    assessor_idx = [{"s": 0}]
    
    # Convert observed orders and outcomes to the expected format
    formatted_orders = []
    formatted_outcomes = []
    
    for i, order in enumerate(observed_orders):
        formatted_order = [{"i": idx, "s": 0} for idx in order]
        formatted_orders.append(formatted_order)
        
        if success_outcomes is not None:
            # Convert boolean outcomes to 1/0
            outcomes = [int(outcome) for outcome in success_outcomes[i]]
            formatted_outcomes.append(outcomes)
    
    # Set up features if provided
    X_formatted = X if X is not None else np.ones((n_actions, 1))
    
    # Set up the mandatory arcs as an adjacency matrix
    H_mandatory = np.zeros((n_actions, n_actions))
    for src, dst in mandatory_arcs:
        H_mandatory[src, dst] = 1
    
    # Apply transitive closure to ensure the constraints are consistent
    H_mandatory = BasicUtils.transitive_closure(H_mandatory)
    
    # Call the existing MCMC function
    mcmc_result = mcmc_simulation_hpo_k(
        S=1,                       # Single assessor
        N=n_actions,               # Number of actions
        n_orders=len(observed_orders),
        orders=formatted_orders,
        outcomes=formatted_outcomes if success_outcomes is not None else None,
        item_idx=item_idx,
        assessor_idx=assessor_idx,
        n_iterations=num_iterations,
        update_probs=update_probs,
        H_mandatory=H_mandatory,
        X=X_formatted,
        K=K,
        noise_option=noise_option,
        sigma_beta=sigma_beta,
        random_seed=random_seed
    )
    
    # Extract the inferred partial order
    H_final = mcmc_result.get("H_final", [np.zeros((n_actions, n_actions))])[0]
    
    # Convert the adjacency matrix to a list of (source, target) pairs
    inferred_constraints = []
    for i in range(n_actions):
        for j in range(n_actions):
            if H_final[i, j] > 0.5 and not H_mandatory[i, j]:  # Only include non-mandatory inferred constraints
                inferred_constraints.append((i, j))
    
    # Extract action-specific results
    action_success_probs = {}
    if 'alpha_final' in mcmc_result and X is not None:
        alpha_final = mcmc_result['alpha_final'][0]
        beta_final = mcmc_result['beta_final'][0]
        
        # Calculate success probability for each action
        logits = alpha_final + X_formatted @ beta_final
        probs = 1 / (1 + np.exp(-logits))
        
        for i, action in enumerate(actions):
            action_success_probs[action] = probs[i]
    
    # Prepare the final result
    result = {
        **mcmc_result,
        'actions': actions,
        'inferred_constraints': inferred_constraints,
        'action_success_probs': action_success_probs
    }
    
    return result


def multi_agent_bayesian_pop(
    num_iterations: int,
    agents: List[str],
    agent_actions: Dict[str, List[str]],
    agent_mandatory_arcs: Dict[str, List[Tuple[int, int]]],
    cross_agent_arcs: List[Tuple[str, int, str, int]],
    observed_orders: Dict[str, List[List[int]]],
    success_outcomes: Optional[Dict[str, List[List[bool]]]] = None,
    K: int = 2,
    noise_option: str = "queue_jump",
    sigma_beta: Optional[Union[float, np.ndarray]] = 1.0,
    update_probs: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run MCMC simulation for Multi-Agent Bayesian Partial Order Planning.
    
    Args:
        num_iterations: Number of MCMC iterations
        agents: List of agent names
        agent_actions: Dictionary mapping agent names to their action lists
        agent_mandatory_arcs: Dictionary mapping agent names to their mandatory constraints
        cross_agent_arcs: List of (agent1, action1_idx, agent2, action2_idx) tuples for cross-agent constraints
        observed_orders: Dictionary mapping agent names to lists of observed execution orders
        success_outcomes: Optional dictionary mapping agent names to lists of success/failure outcomes
        K: Latent space dimension
        noise_option: Type of noise model ("queue_jump" or "latent_space")
        sigma_beta: Prior standard deviation for beta coefficients
        update_probs: Dictionary of update probabilities for MCMC parameters
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing MCMC results and inferred partial orders
    """
    # Create a global list of all actions
    all_actions = []
    agent_to_global_idx = {}
    global_to_agent_idx = {}
    
    # Map local indices to global indices
    current_idx = 0
    for agent in agents:
        agent_to_global_idx[agent] = {}
        for local_idx, action in enumerate(agent_actions[agent]):
            all_actions.append(f"{agent}_{action}")
            agent_to_global_idx[agent][local_idx] = current_idx
            global_to_agent_idx[current_idx] = (agent, local_idx)
            current_idx += 1
    
    # Convert agent-specific mandatory arcs to global indices
    global_mandatory_arcs = []
    for agent, arcs in agent_mandatory_arcs.items():
        for src, dst in arcs:
            global_src = agent_to_global_idx[agent][src]
            global_dst = agent_to_global_idx[agent][dst]
            global_mandatory_arcs.append((global_src, global_dst))
    
    # Convert cross-agent arcs to global indices
    for agent1, action1, agent2, action2 in cross_agent_arcs:
        global_src = agent_to_global_idx[agent1][action1]
        global_dst = agent_to_global_idx[agent2][action2]
        global_mandatory_arcs.append((global_src, global_dst))
    
    # Convert observed orders to global indices
    global_observed_orders = []
    global_success_outcomes = []
    
    # For each execution
    execution_count = len(next(iter(observed_orders.values())))
    for execution_idx in range(execution_count):
        # Combine all agent orders for this execution
        global_order = []
        global_outcomes = []
        
        # Add each agent's actions in their specified order
        for agent in agents:
            local_order = observed_orders[agent][execution_idx]
            for local_idx in local_order:
                global_idx = agent_to_global_idx[agent][local_idx]
                global_order.append(global_idx)
                
                # Add outcomes if available
                if success_outcomes is not None:
                    outcome_idx = local_order.index(local_idx)
                    outcome = success_outcomes[agent][execution_idx][outcome_idx]
                    global_outcomes.append(outcome)
        
        global_observed_orders.append(global_order)
        if success_outcomes is not None:
            global_success_outcomes.append(global_outcomes)
    
    # Run the single-agent MCMC simulation with the combined data
    result = mcmc_simulation_bayesian_pop(
        num_iterations=num_iterations,
        actions=all_actions,
        mandatory_arcs=global_mandatory_arcs,
        observed_orders=global_observed_orders,
        success_outcomes=global_success_outcomes if success_outcomes is not None else None,
        K=K,
        noise_option=noise_option,
        sigma_beta=sigma_beta,
        update_probs=update_probs,
        random_seed=random_seed
    )
    
    # Extract agent-specific partial orders
    H_final = result['H_final'][0]
    
    # Convert back to agent-specific constraints
    agent_partial_orders = {agent: [] for agent in agents}
    cross_agent_constraints = []
    
    # Check each pair in the inferred H matrix
    for i in range(len(all_actions)):
        for j in range(len(all_actions)):
            if H_final[i, j] > 0.5:  # If there's an inferred constraint
                agent_i, local_i = global_to_agent_idx[i]
                agent_j, local_j = global_to_agent_idx[j]
                
                if agent_i == agent_j:  # Within-agent constraint
                    # Only add if it's not already in the mandatory constraints
                    if (local_i, local_j) not in agent_mandatory_arcs.get(agent_i, []):
                        agent_partial_orders[agent_i].append((local_i, local_j))
                else:  # Cross-agent constraint
                    # Only add if it's not already in the cross-agent constraints
                    cross_constraint = (agent_i, local_i, agent_j, local_j)
                    if cross_constraint not in cross_agent_arcs:
                        cross_agent_constraints.append(cross_constraint)
    
    # Extract agent-specific success probabilities
    agent_success_probs = {}
    if 'action_success_probs' in result:
        for agent in agents:
            agent_success_probs[agent] = {}
            for local_idx, action in enumerate(agent_actions[agent]):
                global_idx = agent_to_global_idx[agent][local_idx]
                global_action = all_actions[global_idx]
                if global_action in result['action_success_probs']:
                    agent_success_probs[agent][action] = result['action_success_probs'][global_action]
    
    # Prepare the final result
    multi_agent_result = {
        **result,
        'agents': agents,
        'agent_actions': agent_actions,
        'agent_partial_orders': agent_partial_orders,
        'cross_agent_constraints': cross_agent_constraints,
        'agent_success_probs': agent_success_probs
    }
    
    return multi_agent_result


def visualize_partial_order(
    actions: List[str],
    adjacency_matrix: np.ndarray,
    title: str = "Partial Order Plan"
) -> plt.Figure:
    """
    Visualize a partial order as a directed graph.
    
    Args:
        actions: List of action names
        adjacency_matrix: Binary adjacency matrix representing the partial order
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, action in enumerate(actions):
        G.add_node(i, label=action)
    
    # Add edges from the adjacency matrix
    for i in range(len(actions)):
        for j in range(len(actions)):
            if adjacency_matrix[i, j] > 0:
                G.add_edge(i, j)
    
    # Reduce to transitive reduction for cleaner visualization
    G = nx.transitive_reduction(G)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, arrows=True, arrowsize=20)
    
    # Add labels
    labels = {i: actions[i] for i in range(len(actions))}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    return plt


def visualize_multi_agent_plan(
    agents: List[str],
    agent_actions: Dict[str, List[str]],
    agent_partial_orders: Dict[str, List[Tuple[int, int]]],
    cross_agent_constraints: List[Tuple[str, int, str, int]],
    title: str = "Multi-Agent Partial Order Plan"
) -> plt.Figure:
    """
    Visualize a multi-agent partial order plan.
    
    Args:
        agents: List of agent names
        agent_actions: Dictionary mapping agent names to their action lists
        agent_partial_orders: Dictionary mapping agent names to their inferred constraints
        cross_agent_constraints: List of cross-agent constraints
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each agent's actions
    node_colors = {}
    agent_colors = {
        agents[i]: plt.cm.tab10(i) for i in range(min(len(agents), 10))
    }
    
    # Create node mapping
    node_mapping = {}  # Maps (agent, action_idx) to node_id
    node_labels = {}  # Maps node_id to display label
    
    node_id = 0
    for agent in agents:
        for action_idx, action in enumerate(agent_actions[agent]):
            node_mapping[(agent, action_idx)] = node_id
            node_labels[node_id] = f"{agent}:\n{action}"
            node_colors[node_id] = agent_colors[agent]
            G.add_node(node_id)
            node_id += 1
    
    # Add edges for within-agent constraints
    within_edges = []
    for agent, constraints in agent_partial_orders.items():
        for src, dst in constraints:
            src_id = node_mapping[(agent, src)]
            dst_id = node_mapping[(agent, dst)]
            within_edges.append((src_id, dst_id))
            G.add_edge(src_id, dst_id)
    
    # Add edges for cross-agent constraints
    cross_edges = []
    for agent1, action1, agent2, action2 in cross_agent_constraints:
        src_id = node_mapping[(agent1, action1)]
        dst_id = node_mapping[(agent2, action2)]
        cross_edges.append((src_id, dst_id))
        G.add_edge(src_id, dst_id)
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Use a hierarchical layout
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_size=3000, 
                          node_color=list(node_colors[node] for node in G.nodes()),
                          alpha=0.8)
    
    # Draw within-agent edges
    nx.draw_networkx_edges(G, pos,
                          edgelist=within_edges,
                          width=2,
                          alpha=0.7,
                          arrows=True,
                          arrowsize=20,
                          edge_color='blue')
    
    # Draw cross-agent edges
    nx.draw_networkx_edges(G, pos,
                          edgelist=cross_edges,
                          width=2.5,
                          alpha=0.9,
                          arrows=True,
                          arrowsize=25,
                          edge_color='red',
                          style='dashed')
    
    # Add labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=12)
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Within-Agent Constraint'),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Cross-Agent Constraint')
    ]
    
    # Add agent color legend
    for agent, color in agent_colors.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=agent,
                   markerfacecolor=color, markersize=15)
        )
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    return plt 