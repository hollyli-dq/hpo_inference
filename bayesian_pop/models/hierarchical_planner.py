"""
Hierarchical Bayesian Partial Order Planning Model

This module extends the BayesianPartialOrderPlanner to support multi-agent planning
scenarios using hierarchical partial orders.
"""

import numpy as np
import math
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import copy

from bayesian_pop.models.partial_order_planner import BayesianPartialOrderPlanner

class HierarchicalPartialOrderPlanner:
    def __init__(
        self,
        agents: List[str],
        agent_actions: Dict[str, List[str]],
        global_actions: Optional[List[str]] = None,
        mandatory_arcs: Dict[str, List[Tuple[int, int]]] = None,
        cross_agent_arcs: List[Tuple[str, int, str, int]] = None,
        K: int = 2,
        noise_option: str = "queue_jump",
        random_seed: int = 42
    ):
        """
        Initialize the Hierarchical Partial Order Planner.
        
        Parameters:
        -----------
        agents: List[str]
            List of agent names
        agent_actions: Dict[str, List[str]]
            Dictionary mapping each agent to their list of actions
        global_actions: List[str], optional
            List of actions that can be performed by any agent
        mandatory_arcs: Dict[str, List[Tuple[int, int]]], optional
            Dictionary mapping agent to their mandatory arcs
        cross_agent_arcs: List[Tuple[str, int, str, int]], optional
            List of tuples (agent1, action1_idx, agent2, action2_idx) indicating
            that agent1's action1 must precede agent2's action2
        K: int
            Dimension of latent space
        noise_option: str
            Noise model to use
        random_seed: int
            Random seed for reproducibility
        """
        self.agents = agents
        self.agent_actions = agent_actions
        self.global_actions = global_actions or []
        self.mandatory_arcs = mandatory_arcs or {agent: [] for agent in agents}
        self.cross_agent_arcs = cross_agent_arcs or []
        self.K = K
        self.noise_option = noise_option
        self.random_seed = random_seed
        
        # Dictionary mapping (agent, action_idx) to global action index
        self.agent_action_to_global = {}
        self.global_to_agent_action = {}
        
        # Create a global list of all actions
        self.all_actions = []
        self.agent_action_offsets = {}
        
        offset = 0
        for agent in self.agents:
            self.agent_action_offsets[agent] = offset
            for i, action in enumerate(self.agent_actions[agent]):
                global_idx = len(self.all_actions)
                self.all_actions.append(f"{agent}_{action}")
                self.agent_action_to_global[(agent, i)] = global_idx
                self.global_to_agent_action[global_idx] = (agent, i)
                offset += 1
        
        # Convert cross-agent arcs to global indices
        global_mandatory_arcs = []
        for agent, arcs in self.mandatory_arcs.items():
            for i, j in arcs:
                global_i = self.agent_action_to_global[(agent, i)]
                global_j = self.agent_action_to_global[(agent, j)]
                global_mandatory_arcs.append((global_i, global_j))
        
        # Add cross-agent arcs
        for agent1, action1, agent2, action2 in self.cross_agent_arcs:
            global_i = self.agent_action_to_global[(agent1, action1)]
            global_j = self.agent_action_to_global[(agent2, action2)]
            global_mandatory_arcs.append((global_i, global_j))
        
        # Create a single BayesianPartialOrderPlanner with all actions
        self.planner = BayesianPartialOrderPlanner(
            actions=self.all_actions,
            mandatory_arcs=global_mandatory_arcs,
            K=K,
            noise_option=noise_option,
            random_seed=random_seed
        )
        
        # Dictionary to store individual agent planners
        self.agent_planners = {}
        
    def initialize_parameters(
        self,
        X: Optional[np.ndarray] = None,
        sigma_beta: float = 1.0
    ):
        """
        Initialize model parameters.
        
        Parameters:
        -----------
        X: np.ndarray, optional
            Covariate matrix for all actions
        sigma_beta: float
            Prior standard deviation for beta coefficients
        """
        # Initialize the global planner
        if X is None:
            # Create a simple feature matrix if none provided
            X = np.eye(len(self.all_actions))
        
        self.planner.initialize_parameters(X=X, sigma_beta=sigma_beta)
        
        # Set up the assessors for the hierarchical model
        assessors = {i: agent for i, agent in enumerate(self.agents)}
        M_a_dict = {}
        
        for i, agent in assessors.items():
            # Get the global indices for this agent's actions
            M_a = [self.agent_action_to_global[(agent, j)] for j in range(len(self.agent_actions[agent]))]
            M_a_dict[i] = M_a
        
        # Initialize assessor parameters in the global planner
        self.planner.initialize_assessor_parameters(
            assessors=list(assessors.keys()),
            M_a_dict=M_a_dict
        )
        
        # Update the partial order
        self.planner.update_partial_order()
        
    def sample_multi_agent_execution(self, p_noise: Optional[float] = None):
        """
        Sample an execution sequence that respects agent assignments.
        
        Parameters:
        -----------
        p_noise: float, optional
            Probability of queue jump noise
            
        Returns:
        --------
        Dict[str, List[int]]
            Dictionary mapping agent to their execution sequence
        """
        # Sample a global execution order
        global_execution = self.planner.sample_execution_order(p_noise)
        
        # Split the execution by agent
        agent_executions = {agent: [] for agent in self.agents}
        
        for global_idx in global_execution:
            agent, local_idx = self.global_to_agent_action[global_idx]
            agent_executions[agent].append(local_idx)
        
        return agent_executions
    
    def sample_multi_agent_execution_with_outcomes(self, p_noise: Optional[float] = None):
        """
        Sample an execution sequence with outcomes that respects agent assignments.
        
        Parameters:
        -----------
        p_noise: float, optional
            Probability of queue jump noise
            
        Returns:
        --------
        Tuple[Dict[str, List[int]], Dict[str, List[bool]]]
            Tuple of dictionaries mapping agent to their execution sequence and outcomes
        """
        # Sample global execution order and outcomes
        global_execution, global_outcomes = self.planner.sample_execution_with_outcomes(p_noise)
        
        # Split by agent
        agent_executions = {agent: [] for agent in self.agents}
        agent_outcomes = {agent: [] for agent in self.agents}
        
        for i, global_idx in enumerate(global_execution):
            agent, local_idx = self.global_to_agent_action[global_idx]
            agent_executions[agent].append(local_idx)
            agent_outcomes[agent].append(global_outcomes[i])
        
        return agent_executions, agent_outcomes
    
    def compute_multi_agent_likelihood(
        self,
        observed_orders: Dict[str, List[int]],
        success_outcomes: Optional[Dict[str, List[bool]]] = None
    ) -> float:
        """
        Compute the likelihood of observed multi-agent executions.
        
        Parameters:
        -----------
        observed_orders: Dict[str, List[int]]
            Dictionary mapping agent to their observed execution sequence
        success_outcomes: Dict[str, List[bool]], optional
            Dictionary mapping agent to their observed success/failure outcomes
            
        Returns:
        --------
        float
            Log-likelihood of the observed orders
        """
        # Convert to global indices
        global_observed_order = []
        global_success_outcomes = [] if success_outcomes else None
        
        # Process each agent's observed order
        for agent, order in observed_orders.items():
            for i, local_idx in enumerate(order):
                global_idx = self.agent_action_to_global[(agent, local_idx)]
                global_observed_order.append(global_idx)
                
                if success_outcomes:
                    global_success_outcomes.append(success_outcomes[agent][i])
        
        # Compute likelihood using the global planner
        return self.planner.compute_likelihood(global_observed_order, global_success_outcomes)
    
    def mcmc_update(
        self,
        observed_orders: Dict[str, List[List[int]]],
        success_outcomes: Optional[Dict[str, List[List[bool]]]] = None,
        n_iterations: int = 1000,
        dr: float = 0.95,
        dtau: float = 0.95,
        drbeta: float = 0.1
    ) -> Dict[str, Any]:
        """
        Update model parameters using MCMC, with multiple observations per agent.
        
        Parameters:
        -----------
        observed_orders: Dict[str, List[List[int]]]
            Dictionary mapping agent to list of observed execution sequences
        success_outcomes: Dict[str, List[List[bool]]], optional
            Dictionary mapping agent to list of observed outcomes
        n_iterations: int
            Number of MCMC iterations
        dr, dtau, drbeta: float
            Step size parameters
            
        Returns:
        --------
        Dict
            Dictionary of MCMC results
        """
        # Convert to global indices
        all_global_orders = []
        all_global_outcomes = [] if success_outcomes else None
        
        # Process each agent's observations
        for agent, orders in observed_orders.items():
            for i, order in enumerate(orders):
                global_order = []
                global_outcomes = [] if success_outcomes else None
                
                for j, local_idx in enumerate(order):
                    global_idx = self.agent_action_to_global[(agent, local_idx)]
                    global_order.append(global_idx)
                    
                    if success_outcomes:
                        global_outcomes.append(success_outcomes[agent][i][j])
                
                all_global_orders.append(global_order)
                if success_outcomes:
                    all_global_outcomes.append(global_outcomes)
        
        # Run MCMC with the global planner
        trace = self.planner.mcmc_update(
            observed_orders=all_global_orders,
            success_outcomes=all_global_outcomes,
            n_iterations=n_iterations,
            dr=dr,
            dtau=dtau,
            drbeta=drbeta
        )
        
        return trace
    
    def get_agent_partial_orders(self) -> Dict[str, Dict[int, Set[int]]]:
        """
        Get the partial orders for each agent.
        
        Returns:
        --------
        Dict[str, Dict[int, Set[int]]]
            Dictionary mapping agent to their partial order
        """
        agent_partial_orders = {}
        
        for agent in self.agents:
            # Extract the partial order for this agent's actions
            po = {}
            for local_i in range(len(self.agent_actions[agent])):
                global_i = self.agent_action_to_global[(agent, local_i)]
                
                successors = set()
                for local_j in range(len(self.agent_actions[agent])):
                    global_j = self.agent_action_to_global[(agent, local_j)]
                    if global_j in self.planner.h_U.get(global_i, set()):
                        successors.add(local_j)
                
                po[local_i] = successors
            
            agent_partial_orders[agent] = po
        
        return agent_partial_orders
    
    def get_cross_agent_constraints(self) -> List[Tuple[str, int, str, int]]:
        """
        Get the cross-agent constraints inferred from the model.
        
        Returns:
        --------
        List[Tuple[str, int, str, int]]
            List of tuples (agent1, action1_idx, agent2, action2_idx) representing
            that agent1's action1 must precede agent2's action2
        """
        cross_agent_constraints = []
        
        # Check for edges between actions of different agents
        for global_i in range(len(self.all_actions)):
            agent_i, local_i = self.global_to_agent_action[global_i]
            
            for global_j in self.planner.h_U.get(global_i, set()):
                agent_j, local_j = self.global_to_agent_action[global_j]
                
                if agent_i != agent_j:
                    cross_agent_constraints.append((agent_i, local_i, agent_j, local_j))
        
        return cross_agent_constraints
    
    def get_max_posterior_plan(self) -> Dict[str, Any]:
        """
        Get the maximum posterior probability plan.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all the model information
        """
        return {
            'global_partial_order': self.planner.h_U,
            'agent_partial_orders': self.get_agent_partial_orders(),
            'cross_agent_constraints': self.get_cross_agent_constraints(),
            'rho': self.planner.rho,
            'tau': self.planner.tau,
            'prob_noise': self.planner.prob_noise,
            'pi': self.planner.pi
        } 