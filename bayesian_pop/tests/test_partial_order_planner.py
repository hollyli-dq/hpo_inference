"""
Tests for the BayesianPartialOrderPlanner class.
"""

import pytest
import numpy as np
from bayesian_pop.models.partial_order_planner import BayesianPartialOrderPlanner

def test_planner_initialization():
    """Test that the planner initializes correctly."""
    # Define a simple scenario
    actions = ["Action1", "Action2", "Action3"]
    mandatory_arcs = [(0, 1), (1, 2)]  # Action1 -> Action2 -> Action3
    
    # Create the planner
    planner = BayesianPartialOrderPlanner(
        actions=actions,
        mandatory_arcs=mandatory_arcs,
        K=2,
        noise_option="queue_jump",
        random_seed=42
    )
    
    # Check that the attributes were set correctly
    assert planner.actions == actions
    assert planner.n_actions == len(actions)
    assert planner.mandatory_arcs == mandatory_arcs
    assert planner.K == 2
    assert planner.noise_option == "queue_jump"

def test_parameter_initialization():
    """Test that the parameters initialize correctly."""
    actions = ["Action1", "Action2", "Action3"]
    mandatory_arcs = [(0, 1), (1, 2)]
    
    planner = BayesianPartialOrderPlanner(
        actions=actions,
        mandatory_arcs=mandatory_arcs,
        K=2
    )
    
    # Initialize parameters
    planner.initialize_parameters()
    
    # Check that the parameters were initialized
    assert planner.U0 is not None
    assert planner.Sigma_rho is not None
    assert planner.pi is not None
    assert planner.h_U is not None
    
    # Check dimensions
    assert planner.U0.shape == (len(actions), 2)
    assert planner.Sigma_rho.shape == (2, 2)
    assert len(planner.pi) == len(actions)

def test_sampling_execution_order():
    """Test sampling an execution order."""
    actions = ["Action1", "Action2", "Action3"]
    mandatory_arcs = [(0, 1), (1, 2)]  # Action1 -> Action2 -> Action3
    
    planner = BayesianPartialOrderPlanner(
        actions=actions,
        mandatory_arcs=mandatory_arcs,
        K=2
    )
    
    planner.initialize_parameters()
    
    # Sample an execution order with no noise
    execution_order = planner.sample_execution_order(p_noise=0)
    
    # With no noise, the execution should follow the mandatory arcs
    assert execution_order == [0, 1, 2]
    
    # Sample an execution with outcomes
    execution_order, success_outcomes = planner.sample_execution_with_outcomes(p_noise=0)
    
    # Check that the execution order still follows the mandatory arcs
    assert execution_order == [0, 1, 2]
    
    # Check that success outcomes were generated
    assert len(success_outcomes) == len(execution_order)
    assert all(isinstance(outcome, bool) for outcome in success_outcomes)

def test_likelihood_calculation():
    """Test calculation of likelihood for an execution order."""
    actions = ["Action1", "Action2", "Action3"]
    mandatory_arcs = [(0, 1), (1, 2)]  # Action1 -> Action2 -> Action3
    
    planner = BayesianPartialOrderPlanner(
        actions=actions,
        mandatory_arcs=mandatory_arcs,
        K=2
    )
    
    planner.initialize_parameters()
    
    # Calculate likelihood for a valid execution order
    log_likelihood = planner.compute_likelihood([0, 1, 2])
    
    # The log likelihood should be finite
    assert np.isfinite(log_likelihood)
    
    # Calculate likelihood with success outcomes
    log_likelihood = planner.compute_likelihood([0, 1, 2], [True, True, True])
    
    # The log likelihood should still be finite
    assert np.isfinite(log_likelihood)

def test_mcmc_update():
    """Test that MCMC updates run without errors."""
    actions = ["Action1", "Action2", "Action3"]
    mandatory_arcs = [(0, 1), (1, 2)]  # Action1 -> Action2 -> Action3
    
    planner = BayesianPartialOrderPlanner(
        actions=actions,
        mandatory_arcs=mandatory_arcs,
        K=2
    )
    
    planner.initialize_parameters()
    
    # Generate some observed data
    observed_orders = [[0, 1, 2], [0, 1, 2]]
    success_outcomes = [[True, True, True], [True, False, True]]
    
    # Run a very short MCMC update
    trace = planner.mcmc_update(
        observed_orders=observed_orders,
        success_outcomes=success_outcomes,
        n_iterations=10
    )
    
    # Check that the trace contains the expected keys
    assert "rho" in trace
    assert "tau" in trace
    assert "prob_noise" in trace
    assert "pi" in trace
    assert "U0" in trace
    assert "log_likelihood" in trace
    
    # Check that the trace has the expected length
    assert len(trace["rho"]) == 1  # Since we store every 10 iterations 