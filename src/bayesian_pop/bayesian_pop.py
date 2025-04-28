#!/usr/bin/env python
"""
Bayesian Partial Order Planning (POP) implementation.

This module provides functions for inferring partial orders and success probabilities
from observed execution sequences using Bayesian inference and MCMC methods.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

# Import existing utility functions
from src.utils.po_fun import BasicUtils, ConversionUtils, GenerationUtils, StatisticalUtils
from src.utils.po_accelerator_nle import LogLikelihoodCache, HPO_LogLikelihoodCache

def fix_sigma_beta_parameter(sigma_beta: Union[float, np.ndarray], X: np.ndarray) -> np.ndarray:
    """
    Ensure sigma_beta is properly formatted as an array that matches the number of covariates.
    
    Args:
        sigma_beta: A scalar or array representing the prior standard deviation for beta
        X: Covariate matrix of shape (n_observations, n_covariates)
    
    Returns:
        sigma_beta as an array with length matching the number of covariates in X
    """
    n_covariates = X.shape[1]
    
    # If sigma_beta is already an array with correct shape, return it
    if isinstance(sigma_beta, np.ndarray) and sigma_beta.size == n_covariates:
        return sigma_beta
    
    # If sigma_beta is a scalar, convert to array with same value for all covariates
    if np.isscalar(sigma_beta):
        return np.ones(n_covariates) * sigma_beta
    
    # If sigma_beta is an array but wrong size, raise an error
    if isinstance(sigma_beta, np.ndarray) and sigma_beta.size != n_covariates:
        raise ValueError(f"sigma_beta has size {sigma_beta.size} but should have size {n_covariates}")
    
    return sigma_beta

def generate_synthetic_observations(
    true_H: np.ndarray,
    true_alpha: np.ndarray,
    X: np.ndarray,
    n_observations: int,
    seed: Optional[int] = None
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generate synthetic observation data based on a true partial order and success probabilities.
    
    Args:
        true_H: True adjacency matrix representing the partial order
        true_alpha: True alpha parameters for action success probabilities
        X: Covariate matrix for features associated with each action
        n_observations: Number of observation sequences to generate
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (observed_sequences, success_indicators)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_actions = true_H.shape[0]
    
    # Compute success probabilities from alpha and X
    success_probs = 1 / (1 + np.exp(-X @ true_alpha))
    
    observed_sequences = []
    success_indicators = np.zeros((n_observations, n_actions))
    
    for i in range(n_observations):
        # Sample a valid sequence from the partial order
        valid_seqs = ConversionUtils.po_to_seqs(true_H)
        seq_idx = np.random.choice(len(valid_seqs))
        seq = valid_seqs[seq_idx]
        
        # Determine success/failure for each action
        success = np.random.random(n_actions) < success_probs
        success_indicators[i] = success
        
        # Generate observed sequence based on success/failure
        observed_seq = []
        for action in seq:
            observed_seq.append(action)
            if not success[action]:
                break
        
        observed_sequences.append(observed_seq)
    
    return observed_sequences, success_indicators

def mcmc_simulation_bayesian_pop(
    observed_sequences: List[List[int]],
    X: np.ndarray,
    n_actions: int,
    num_iterations: int = 5000,
    K: int = 5,
    sigma_beta: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Run MCMC simulation for Bayesian Partial Order Planning.
    
    Args:
        observed_sequences: List of observed execution sequences
        X: Covariate matrix for features associated with each action
        n_actions: Number of actions in the domain
        num_iterations: Number of MCMC iterations
        K: Number of samples drawn for each observation
        sigma_beta: Prior standard deviation for beta coefficients
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (H_samples, beta_samples, H_mean, success_probs)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Ensure sigma_beta is properly formatted
    sigma_beta = fix_sigma_beta_parameter(sigma_beta, X)
    
    # Initialize H from a weakly connected DAG
    H_curr = GenerationUtils.gen_random_DAG(n_actions)
    
    # Initialize beta from prior
    n_covariates = X.shape[1]
    beta_curr = np.random.normal(0, sigma_beta, n_covariates)
    
    # Initialize cache for likelihood computation
    cache = LogLikelihoodCache(n_actions)
    
    # Initialize storage for samples
    burnin = int(num_iterations * 0.2)  # 20% burn-in
    H_samples = []
    beta_samples = []
    
    # Run MCMC
    for iter in range(num_iterations):
        # Update H (partial order structure)
        # Randomly select edge to flip
        i, j = np.random.choice(n_actions, 2, replace=False)
        
        # Create proposed H by flipping edge (i,j)
        H_prop = H_curr.copy()
        H_prop[i, j] = 1 - H_prop[i, j]
        
        # Check if proposed H is still a DAG
        if H_prop[i, j] == 1 and H_prop[j, i] == 1:
            # Cannot have both (i,j) and (j,i)
            continue
        
        # Check for cycles after adding edge
        if H_prop[i, j] == 1:
            H_tc = BasicUtils.transitive_closure(H_prop)
            if H_tc[j, i] == 1:
                # Adding edge would create a cycle
                continue
        
        # Compute likelihood for current and proposed H
        alpha_curr = X @ beta_curr
        
        log_lik_curr = 0
        log_lik_prop = 0
        
        for seq in observed_sequences:
            log_lik_curr += cache.compute_log_likelihood(H_curr, seq, alpha_curr, K)
            log_lik_prop += cache.compute_log_likelihood(H_prop, seq, alpha_curr, K)
        
        # Compute acceptance probability
        log_accept_ratio = log_lik_prop - log_lik_curr
        
        # Accept or reject proposal
        if np.log(np.random.random()) < log_accept_ratio:
            H_curr = H_prop
        
        # Update beta (success probabilities)
        # Propose new beta using random walk
        beta_prop = beta_curr + np.random.normal(0, 0.1, n_covariates)
        
        # Compute alpha from proposed beta
        alpha_prop = X @ beta_prop
        
        # Compute prior for current and proposed beta
        log_prior_curr = np.sum(-0.5 * (beta_curr / sigma_beta)**2)
        log_prior_prop = np.sum(-0.5 * (beta_prop / sigma_beta)**2)
        
        # Compute likelihood for proposed beta
        log_lik_prop = 0
        for seq in observed_sequences:
            log_lik_prop += cache.compute_log_likelihood(H_curr, seq, alpha_prop, K)
        
        # Compute acceptance probability
        log_accept_ratio = log_lik_prop - log_lik_curr + log_prior_prop - log_prior_curr
        
        # Accept or reject proposal
        if np.log(np.random.random()) < log_accept_ratio:
            beta_curr = beta_prop
            alpha_curr = alpha_prop
            log_lik_curr = log_lik_prop
        
        # Store samples after burn-in
        if iter >= burnin:
            H_samples.append(H_curr.copy())
            beta_samples.append(beta_curr.copy())
    
    # Compute posterior mean for H
    H_mean = np.mean(H_samples, axis=0)
    
    # Compute success probabilities
    beta_mean = np.mean(beta_samples, axis=0)
    success_probs = 1 / (1 + np.exp(-X @ beta_mean))
    
    return H_samples, beta_samples, H_mean, success_probs

def multi_agent_bayesian_pop(
    observed_sequences_list: List[List[List[int]]],
    X_list: List[np.ndarray],
    n_actions_list: List[int],
    cross_agent_constraints: Optional[List[Tuple[int, int, int, int]]] = None,
    num_iterations: int = 5000,
    K: int = 5,
    sigma_beta: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None
) -> List[Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]]:
    """
    Run MCMC simulation for multi-agent Bayesian Partial Order Planning.
    
    Args:
        observed_sequences_list: List of observed execution sequences for each agent
        X_list: List of covariate matrices for each agent
        n_actions_list: List of number of actions for each agent
        cross_agent_constraints: List of tuples (agent_i, action_i, agent_j, action_j)
            representing cross-agent constraints
        num_iterations: Number of MCMC iterations
        K: Number of samples drawn for each observation
        sigma_beta: Prior standard deviation for beta coefficients
        seed: Random seed for reproducibility
        
    Returns:
        List of (H_samples, beta_samples, H_mean, success_probs) for each agent
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_agents = len(n_actions_list)
    
    # Initialize results for each agent
    results = []
    
    # Run MCMC simulation for each agent
    for agent_idx in range(n_agents):
        agent_results = mcmc_simulation_bayesian_pop(
            observed_sequences_list[agent_idx],
            X_list[agent_idx],
            n_actions_list[agent_idx],
            num_iterations=num_iterations,
            K=K,
            sigma_beta=sigma_beta,
            seed=seed+agent_idx if seed is not None else None
        )
        results.append(agent_results)
    
    # If cross-agent constraints are specified, incorporate them
    if cross_agent_constraints:
        # TODO: Implement cross-agent constraint handling
        pass
    
    return results

def visualize_partial_order(
    H: np.ndarray,
    action_names: List[str],
    success_probs: Optional[np.ndarray] = None,
    title: str = "Inferred Partial Order",
    filename: Optional[str] = None
) -> nx.DiGraph:
    """
    Visualize a partial order as a directed graph.
    
    Args:
        H: Adjacency matrix representing the partial order
        action_names: Names of actions
        success_probs: Success probabilities for each action
        title: Title for the plot
        filename: If provided, save the figure to this file
        
    Returns:
        NetworkX DiGraph object
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with labels
    for i, name in enumerate(action_names):
        label = name
        if success_probs is not None:
            label += f"\n({success_probs[i]:.2f})"
        
        G.add_node(i, label=label)
    
    # Add edges
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i, j] > 0.5:  # Use threshold for probabilistic H
                G.add_edge(i, j)
    
    # Create transitive reduction for cleaner visualization
    G_reduced = nx.transitive_reduction(G)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_reduced, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_reduced, pos, node_size=2000, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G_reduced, pos, arrowsize=20, width=2, alpha=0.7)
    
    # Draw labels
    labels = nx.get_node_attributes(G_reduced, 'label')
    nx.draw_networkx_labels(G_reduced, pos, labels=labels, font_size=12)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return G_reduced

def visualize_multi_agent_plan(
    H_list: List[np.ndarray],
    action_names_list: List[List[str]],
    success_probs_list: List[np.ndarray],
    cross_agent_constraints: Optional[List[Tuple[int, int, int, int]]] = None,
    title: str = "Multi-Agent Plan",
    filename: Optional[str] = None
) -> nx.DiGraph:
    """
    Visualize a multi-agent plan with cross-agent dependencies.
    
    Args:
        H_list: List of adjacency matrices for each agent
        action_names_list: List of action names for each agent
        success_probs_list: List of success probabilities for each agent
        cross_agent_constraints: List of tuples (agent_i, action_i, agent_j, action_j)
        title: Title for the plot
        filename: If provided, save the figure to this file
        
    Returns:
        NetworkX DiGraph object
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with labels
    agent_offset = 0
    for agent_idx, (action_names, success_probs) in enumerate(zip(action_names_list, success_probs_list)):
        for i, name in enumerate(action_names):
            node_id = agent_offset + i
            label = f"[{agent_idx}] {name}"
            if success_probs is not None:
                label += f"\n({success_probs[i]:.2f})"
            
            G.add_node(node_id, label=label, agent=agent_idx)
        
        # Add edges within each agent
        H = H_list[agent_idx]
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i, j] > 0.5:  # Use threshold for probabilistic H
                    G.add_edge(agent_offset + i, agent_offset + j)
        
        agent_offset += len(action_names)
    
    # Add cross-agent constraints
    if cross_agent_constraints:
        agent_offsets = [0]
        for i in range(len(action_names_list) - 1):
            agent_offsets.append(agent_offsets[-1] + len(action_names_list[i]))
        
        for agent_i, action_i, agent_j, action_j in cross_agent_constraints:
            node_i = agent_offsets[agent_i] + action_i
            node_j = agent_offsets[agent_j] + action_j
            G.add_edge(node_i, node_j, color='red', style='dashed')
    
    # Create transitive reduction for cleaner visualization
    G_reduced = nx.transitive_reduction(G)
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_reduced, seed=42)
    
    # Draw nodes
    agent_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    node_colors = [agent_colors[G.nodes[n]['agent'] % len(agent_colors)] for n in G_reduced.nodes()]
    nx.draw_networkx_nodes(G_reduced, pos, node_size=2000, node_color=node_colors)
    
    # Draw edges
    regular_edges = [(u, v) for u, v, d in G_reduced.edges(data=True) if 'color' not in d]
    cross_edges = [(u, v) for u, v, d in G_reduced.edges(data=True) if 'color' in d]
    
    nx.draw_networkx_edges(G_reduced, pos, edgelist=regular_edges, arrowsize=20, width=2, alpha=0.7)
    if cross_edges:
        nx.draw_networkx_edges(G_reduced, pos, edgelist=cross_edges, arrowsize=20, width=2, alpha=0.7, 
                              edge_color='red', style='dashed')
    
    # Draw labels
    labels = nx.get_node_attributes(G_reduced, 'label')
    nx.draw_networkx_labels(G_reduced, pos, labels=labels, font_size=12)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return G_reduced 