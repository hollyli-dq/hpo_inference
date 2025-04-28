"""
Bayesian Partial Order Planning Model

This module implements Bayesian Partial Order Planning as described in the paper.
It utilizes hierarchical partial orders to represent concurrent actions in planning
problems, and uses a Bayesian approach to update beliefs about which partial orders
are plausible based on observed action sequences.
"""

import numpy as np
import math
from scipy.stats import multivariate_normal, norm, beta
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import copy

class BayesianPartialOrderPlanner:
    def __init__(
        self, 
        actions: List[str],
        mandatory_arcs: List[Tuple[int, int]],
        K: int = 2,  # Dimension of latent space
        noise_option: str = "queue_jump",
        random_seed: int = 42
    ):
        """
        Initialize the Bayesian Partial Order Planner.
        
        Parameters:
        -----------
        actions: List[str]
            List of action names
        mandatory_arcs: List[Tuple[int, int]]
            List of tuples (i, j) indicating that action i must precede action j
        K: int
            Dimension of latent space (default: 2)
        noise_option: str
            Noise model to use ("queue_jump" or "mallows_noise")
        random_seed: int
            Random seed for reproducibility
        """
        self.actions = actions
        self.action_to_idx = {action: idx for idx, action in enumerate(actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(actions)}
        self.n_actions = len(actions)
        self.mandatory_arcs = mandatory_arcs
        self.K = K
        self.noise_option = noise_option
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize parameters
        self.rho = 0.7  # Initial correlation parameter
        self.tau = 0.5  # Initial hierarchical parameter
        self.prob_noise = 0.1  # Initial noise parameter for queue_jump
        self.mallow_theta = 0.5  # Initial theta for Mallows noise
        self.Sigma_rho = None
        self.U0 = None
        self.U_a_dict = {}
        self.h_U = None
        
        # Initialize covariate-related parameters
        self.X = None
        self.beta = None
        self.alpha = None
        
        # Success/failure parameters
        self.pi = None  # Success probabilities for actions
        
    def initialize_parameters(
        self,
        X: Optional[np.ndarray] = None,
        assessors: Optional[List[int]] = None,
        M_a_dict: Optional[Dict[int, List[int]]] = None,
        sigma_beta: float = 1.0
    ):
        """
        Initialize model parameters.
        
        Parameters:
        -----------
        X: np.ndarray, optional
            Covariate matrix (n_actions x p)
        assessors: List[int], optional
            List of assessor IDs
        M_a_dict: Dict[int, List[int]], optional
            Dictionary mapping assessor ID to their action subset
        sigma_beta: float
            Prior standard deviation for beta coefficients
        """
        # Initialize Sigma_rho (correlation matrix)
        self.Sigma_rho = self._build_Sigma_rho(self.K, self.rho)
        
        # Initialize U0 (global latent variables)
        self.U0 = self.rng.multivariate_normal(
            mean=np.zeros(self.K), 
            cov=self.Sigma_rho, 
            size=self.n_actions
        )
        
        # Initialize success probabilities
        self.pi = np.full(self.n_actions, 0.8)  # Default 80% success rate
        
        # Initialize covariate-related parameters if X is provided
        if X is not None:
            self.X = X
            p = X.shape[1]  # Number of covariates
            self.beta = self.rng.normal(loc=0.0, scale=sigma_beta, size=(p,))
            self.alpha = X.T @ self.beta
            
        # Initialize assessor-specific parameters if provided
        if assessors is not None and M_a_dict is not None:
            self.initialize_assessor_parameters(assessors, M_a_dict)
            
        # Initialize the partial order
        self.update_partial_order()
    
    def _build_Sigma_rho(self, K, rho):
        """Build the correlation matrix Sigma_rho with correlation parameter rho."""
        Sigma_rho = np.eye(K)
        for i in range(K):
            for j in range(K):
                if i != j:
                    Sigma_rho[i, j] = rho
        return Sigma_rho
    
    def initialize_assessor_parameters(self, assessors: List[int], M_a_dict: Dict[int, List[int]]):
        """
        Initialize assessor-specific parameters.
        
        Parameters:
        -----------
        assessors: List[int]
            List of assessor IDs
        M_a_dict: Dict[int, List[int]]
            Dictionary mapping assessor ID to their action subset
        """
        for a in assessors:
            M_a = M_a_dict.get(a, [])
            n_a = len(M_a)
            Ua = np.zeros((n_a, self.K), dtype=float)
            
            for i_loc, j_global in enumerate(M_a):
                mean_vec = self.tau * self.U0[j_global, :]
                cov_mat = (1.0 - self.tau**2) * self.Sigma_rho
                Ua[i_loc, :] = self.rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
                
            self.U_a_dict[a] = Ua
    
    def update_partial_order(self):
        """
        Update the partial order based on current U0, U_a_dict, and alpha.
        """
        # If not using the hierarchical model, just use the mandatory arcs
        self.h_U = self._build_partial_order_from_latents()
    
    def _build_partial_order_from_latents(self):
        """
        Build a partial order from the latent variables.
        
        Returns:
        --------
        Dict: The partial order represented as an adjacency dictionary
        """
        # Initialize with the mandatory arcs
        po = {i: set() for i in range(self.n_actions)}
        for i, j in self.mandatory_arcs:
            po[i].add(j)
        
        # Add edges based on the latent representations
        for i in range(self.n_actions):
            for j in range(self.n_actions):
                if i != j and (i, j) not in self.mandatory_arcs:
                    # Check if U0[i] "dominates" U0[j] in all dimensions
                    if all(self.U0[i, k] > self.U0[j, k] for k in range(self.K)):
                        po[i].add(j)
        
        # Return the partial order
        return po
    
    def sample_execution_order(self, p_noise: Optional[float] = None) -> List[int]:
        """
        Sample a total order execution sequence given the current partial order.
        
        Parameters:
        -----------
        p_noise: float, optional
            Probability of queue jump noise (overrides self.prob_noise if provided)
            
        Returns:
        --------
        List[int]
            A sampled total order as a list of action indices
        """
        if p_noise is None:
            p_noise = self.prob_noise
        
        # Initialize variables
        execution_order = []
        remaining_actions = set(range(self.n_actions))
        
        # Sample the execution order
        while remaining_actions:
            # Find the set of maximal feasible actions (no remaining predecessors)
            maximal_actions = self._get_maximal_actions(remaining_actions)
            
            # With probability p_noise, choose uniformly from remaining actions
            # With probability 1-p_noise, choose from maximal actions
            if random.random() < p_noise and remaining_actions:
                # Queue jump noise: choose any remaining action
                next_action = random.choice(list(remaining_actions))
            else:
                # Choose from maximal actions
                if maximal_actions:
                    next_action = random.choice(list(maximal_actions))
                else:
                    # If no maximal actions (shouldn't happen with a valid partial order)
                    next_action = random.choice(list(remaining_actions))
            
            execution_order.append(next_action)
            remaining_actions.remove(next_action)
        
        return execution_order
    
    def _get_maximal_actions(self, remaining_actions):
        """
        Get the maximal actions from the remaining actions.
        
        Parameters:
        -----------
        remaining_actions: Set[int]
            Set of remaining action indices
            
        Returns:
        --------
        List[int]
            List of maximal action indices
        """
        maximal_actions = []
        for action in remaining_actions:
            has_predecessor = False
            for pred in remaining_actions:
                if action in self.h_U.get(pred, set()):
                    has_predecessor = True
                    break
            if not has_predecessor:
                maximal_actions.append(action)
        return maximal_actions
    
    def sample_execution_with_outcomes(self, p_noise: Optional[float] = None) -> Tuple[List[int], List[bool]]:
        """
        Sample an execution sequence with success/failure outcomes.
        
        Parameters:
        -----------
        p_noise: float, optional
            Probability of queue jump noise
            
        Returns:
        --------
        Tuple[List[int], List[bool]]
            A tuple containing (execution_order, success_outcomes)
        """
        execution_order = self.sample_execution_order(p_noise)
        success_outcomes = [random.random() < self.pi[action] for action in execution_order]
        
        return execution_order, success_outcomes
    
    def compute_likelihood(
        self,
        observed_order: List[int],
        success_outcomes: Optional[List[bool]] = None
    ) -> float:
        """
        Compute the likelihood of an observed execution order.
        
        Parameters:
        -----------
        observed_order: List[int]
            The observed execution order as a list of action indices
        success_outcomes: List[bool], optional
            Success/failure outcomes for each action
            
        Returns:
        --------
        float
            The log-likelihood of the observed order
        """
        log_likelihood = 0.0
        remaining_actions = set(range(self.n_actions))
        
        # Process each action in the observed order
        for step, action in enumerate(observed_order):
            # Find maximal actions at this step
            maximal_actions = self._get_maximal_actions(remaining_actions)
            
            # Compute probability of choosing this action
            if action in maximal_actions:
                # Action is maximal (follows partial order)
                p_following_order = (1 - self.prob_noise) * (1.0 / len(maximal_actions))
            else:
                # Action is not maximal (queue jump)
                p_following_order = self.prob_noise * (1.0 / len(remaining_actions))
            
            # Add log probability for this action
            log_likelihood += math.log(p_following_order)
            
            # If we have success/failure outcomes, include those in likelihood
            if success_outcomes is not None:
                success = success_outcomes[step]
                p_success = self.pi[action] if success else (1 - self.pi[action])
                log_likelihood += math.log(p_success)
            
            # Remove the action from remaining actions
            remaining_actions.remove(action)
        
        return log_likelihood
    
    def mcmc_update(
        self,
        observed_orders: List[List[int]],
        success_outcomes: Optional[List[List[bool]]] = None,
        n_iterations: int = 1000,
        X: Optional[np.ndarray] = None,
        sigma_beta: float = 1.0,
        dr: float = 0.95,  # Step size for rho
        dtau: float = 0.95,  # Step size for tau
        drbeta: float = 0.1,  # Step size for beta
    ) -> Dict[str, Any]:
        """
        Update model parameters using MCMC.
        
        Parameters:
        -----------
        observed_orders: List[List[int]]
            List of observed execution orders
        success_outcomes: List[List[bool]], optional
            List of success/failure outcomes for each execution
        n_iterations: int
            Number of MCMC iterations
        X: np.ndarray, optional
            Covariate matrix (n_actions x p)
        sigma_beta: float
            Prior standard deviation for beta coefficients
        dr: float
            Step size parameter for rho updates
        dtau: float
            Step size parameter for tau updates
        drbeta: float
            Step size parameter for beta updates
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of MCMC results
        """
        # Initialize parameters if needed
        if self.U0 is None:
            self.initialize_parameters(X, sigma_beta=sigma_beta)
        
        # Initialize trace storage
        trace = {
            'rho': [], 'tau': [], 'prob_noise': [], 'pi': [], 
            'U0': [], 'U_a': [], 'beta': [], 'log_likelihood': []
        }
        
        # Calculate initial log-likelihood
        current_log_likelihood = 0.0
        for i, order in enumerate(observed_orders):
            if success_outcomes is not None:
                current_log_likelihood += self.compute_likelihood(order, success_outcomes[i])
            else:
                current_log_likelihood += self.compute_likelihood(order)
        
        # Main MCMC loop
        for iteration in range(n_iterations):
            # Choose update type randomly
            update_type = random.choice(['rho', 'tau', 'noise', 'U0', 'pi'])
            
            # Perform the chosen update
            if update_type == 'rho':
                # Update rho (correlation parameter)
                delta = random.uniform(dr, 1.0 / dr)
                rho_prime = 1.0 - (1.0 - self.rho) * delta
                if not (0.0 < rho_prime < 1.0):
                    rho_prime = self.rho
                
                # Propose new partial order with updated rho
                old_rho = self.rho
                old_Sigma_rho = self.Sigma_rho
                self.rho = rho_prime
                self.Sigma_rho = self._build_Sigma_rho(self.K, self.rho)
                
                # Compute new log-likelihood
                proposed_log_likelihood = 0.0
                for i, order in enumerate(observed_orders):
                    if success_outcomes is not None:
                        proposed_log_likelihood += self.compute_likelihood(order, success_outcomes[i])
                    else:
                        proposed_log_likelihood += self.compute_likelihood(order)
                
                # Compute log prior ratio
                log_prior_ratio = 0  # Assuming uniform prior on rho
                
                # Compute log acceptance ratio
                log_acceptance_ratio = (
                    proposed_log_likelihood - current_log_likelihood 
                    + log_prior_ratio 
                    - math.log(delta)  # Jacobian term for the transformation
                )
                
                # Accept or reject
                accept = random.random() < min(1.0, math.exp(min(log_acceptance_ratio, 700)))
                if accept:
                    current_log_likelihood = proposed_log_likelihood
                else:
                    # Revert to old values
                    self.rho = old_rho
                    self.Sigma_rho = old_Sigma_rho
            
            elif update_type == 'tau':
                # Update tau (hierarchical parameter)
                delta = random.uniform(dtau, 1.0 / dtau)
                tau_prime = 1.0 - (1.0 - self.tau) * delta
                if not (0.0 < tau_prime < 1.0):
                    tau_prime = self.tau
                
                # Propose new partial order with updated tau
                old_tau = self.tau
                self.tau = tau_prime
                self.update_partial_order()
                
                # Compute new log-likelihood
                proposed_log_likelihood = 0.0
                for i, order in enumerate(observed_orders):
                    if success_outcomes is not None:
                        proposed_log_likelihood += self.compute_likelihood(order, success_outcomes[i])
                    else:
                        proposed_log_likelihood += self.compute_likelihood(order)
                
                # Compute log prior ratio
                log_prior_ratio = 0  # Assuming uniform prior on tau
                
                # Compute log acceptance ratio
                log_acceptance_ratio = (
                    proposed_log_likelihood - current_log_likelihood 
                    + log_prior_ratio 
                    - math.log(delta)  # Jacobian term
                )
                
                # Accept or reject
                accept = random.random() < min(1.0, math.exp(min(log_acceptance_ratio, 700)))
                if accept:
                    current_log_likelihood = proposed_log_likelihood
                else:
                    # Revert to old values
                    self.tau = old_tau
                    self.update_partial_order()
            
            elif update_type == 'noise':
                # Update noise parameter
                if self.noise_option == 'queue_jump':
                    # Sample from Beta prior
                    old_prob_noise = self.prob_noise
                    prob_noise_prime = beta.rvs(1, 9)  # Example prior: Beta(1, 9)
                    self.prob_noise = prob_noise_prime
                    
                    # Compute new log-likelihood
                    proposed_log_likelihood = 0.0
                    for i, order in enumerate(observed_orders):
                        if success_outcomes is not None:
                            proposed_log_likelihood += self.compute_likelihood(order, success_outcomes[i])
                        else:
                            proposed_log_likelihood += self.compute_likelihood(order)
                    
                    # Compute log prior ratio
                    log_prior_ratio = (
                        beta.logpdf(prob_noise_prime, 1, 9) - beta.logpdf(old_prob_noise, 1, 9)
                    )
                    
                    # Compute log acceptance ratio
                    log_acceptance_ratio = proposed_log_likelihood - current_log_likelihood + log_prior_ratio
                    
                    # Accept or reject
                    accept = random.random() < min(1.0, math.exp(min(log_acceptance_ratio, 700)))
                    if accept:
                        current_log_likelihood = proposed_log_likelihood
                    else:
                        # Revert to old values
                        self.prob_noise = old_prob_noise
                else:
                    # Update Mallows noise parameter
                    pass
            
            elif update_type == 'U0':
                # Update one row of U0
                j = random.randint(0, self.n_actions - 1)
                
                # Propose new value
                old_U0 = self.U0.copy()
                self.U0[j, :] = self.rng.multivariate_normal(
                    mean=self.U0[j, :],
                    cov=0.1 * self.Sigma_rho
                )
                self.update_partial_order()
                
                # Compute new log-likelihood
                proposed_log_likelihood = 0.0
                for i, order in enumerate(observed_orders):
                    if success_outcomes is not None:
                        proposed_log_likelihood += self.compute_likelihood(order, success_outcomes[i])
                    else:
                        proposed_log_likelihood += self.compute_likelihood(order)
                
                # Compute log prior ratio (using multivariate normal density)
                old_log_prior = multivariate_normal.logpdf(old_U0[j, :], mean=np.zeros(self.K), cov=self.Sigma_rho)
                new_log_prior = multivariate_normal.logpdf(self.U0[j, :], mean=np.zeros(self.K), cov=self.Sigma_rho)
                log_prior_ratio = new_log_prior - old_log_prior
                
                # Compute log acceptance ratio
                log_acceptance_ratio = proposed_log_likelihood - current_log_likelihood + log_prior_ratio
                
                # Accept or reject
                accept = random.random() < min(1.0, math.exp(min(log_acceptance_ratio, 700)))
                if accept:
                    current_log_likelihood = proposed_log_likelihood
                else:
                    # Revert to old values
                    self.U0 = old_U0
                    self.update_partial_order()
            
            elif update_type == 'pi':
                # Update success probability for a random action
                if success_outcomes is not None:
                    j = random.randint(0, self.n_actions - 1)
                    
                    # Sample from Beta prior
                    old_pi_j = self.pi[j]
                    pi_prime = beta.rvs(1, 1)  # Uniform prior
                    self.pi[j] = pi_prime
                    
                    # Compute new log-likelihood
                    proposed_log_likelihood = 0.0
                    for i, order in enumerate(observed_orders):
                        proposed_log_likelihood += self.compute_likelihood(order, success_outcomes[i])
                    
                    # Compute log prior ratio (uniform prior, so this is 0)
                    log_prior_ratio = 0
                    
                    # Compute log acceptance ratio
                    log_acceptance_ratio = proposed_log_likelihood - current_log_likelihood + log_prior_ratio
                    
                    # Accept or reject
                    accept = random.random() < min(1.0, math.exp(min(log_acceptance_ratio, 700)))
                    if accept:
                        current_log_likelihood = proposed_log_likelihood
                    else:
                        # Revert to old values
                        self.pi[j] = old_pi_j
            
            # Store trace at regular intervals
            if iteration % 10 == 0:
                trace['rho'].append(self.rho)
                trace['tau'].append(self.tau)
                trace['prob_noise'].append(self.prob_noise)
                trace['pi'].append(self.pi.copy())
                trace['U0'].append(self.U0.copy())
                trace['log_likelihood'].append(current_log_likelihood)
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{n_iterations}, Log-Likelihood: {current_log_likelihood:.2f}")
        
        return trace
    
    def get_max_posterior_plan(self) -> Dict[str, Any]:
        """
        Get the maximum posterior probability plan.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the partial order and related information
        """
        # Return the current partial order
        return {
            'partial_order': self.h_U,
            'U0': self.U0,
            'rho': self.rho,
            'tau': self.tau,
            'prob_noise': self.prob_noise,
            'pi': self.pi
        } 