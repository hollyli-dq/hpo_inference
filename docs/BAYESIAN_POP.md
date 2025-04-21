# Bayesian Partial Order Planning

This document describes the Bayesian Partial Order Planning (BPOP) module and its capabilities for inferring partial order constraints and success probabilities from observed execution sequences.

## Overview

Bayesian Partial Order Planning is a framework for inferring constraints and success probabilities in planning domains using observations of execution sequences and their outcomes. The module implements:

1. **Single-Agent Planning**: Infer partial order constraints and success probabilities for a single agent's actions.
2. **Multi-Agent Planning**: Infer constraints within each agent's actions and cross-agent constraints between different agents.
3. **Action Success Prediction**: Learn the relationship between action features and success probabilities.

## Core Functions

### Single-Agent Planning

```python
mcmc_simulation_bayesian_pop(
    num_iterations,
    actions,
    mandatory_arcs,
    observed_orders,
    success_outcomes,
    X=None,
    K=2,
    noise_option="queue_jump",
    sigma_beta=1.0,
    random_seed=None
)
```

**Parameters:**
- `num_iterations`: Number of MCMC iterations to run
- `actions`: List of action names
- `mandatory_arcs`: List of tuples (i, j) representing mandatory ordering constraints
- `observed_orders`: List of execution sequences (each a list of action indices)
- `success_outcomes`: List of boolean lists indicating success/failure for each action in the execution sequence
- `X`: Optional feature matrix for actions, used for predicting success probabilities
- `K`: Number of latent dimensions for the embedding space
- `noise_option`: Type of noise to use in MCMC ("queue_jump" or "latent_position")
- `sigma_beta`: Standard deviation for beta parameters in the hierarchical model
- `random_seed`: Random seed for reproducibility

**Returns:**
A dictionary containing:
- `H_final`: Final partial order matrix
- `inferred_constraints`: List of inferred constraints (i, j) 
- `action_success_probs`: Dictionary mapping action names to success probabilities
- `alpha`: Inferred alpha values for success probability prediction
- `beta`: Inferred beta values for success probability prediction

### Multi-Agent Planning

```python
multi_agent_bayesian_pop(
    num_iterations,
    agents,
    agent_actions,
    agent_mandatory_arcs,
    cross_agent_arcs=None,
    observed_orders=None,
    success_outcomes=None,
    K=2,
    noise_option="queue_jump",
    sigma_beta=1.0,
    random_seed=None
)
```

**Parameters:**
- `num_iterations`: Number of MCMC iterations to run
- `agents`: List of agent names
- `agent_actions`: Dictionary mapping agent names to their action lists
- `agent_mandatory_arcs`: Dictionary mapping agent names to their mandatory constraint lists
- `cross_agent_arcs`: List of tuples (agent1, action1_idx, agent2, action2_idx) for cross-agent constraints
- `observed_orders`: Dictionary mapping agent names to their observed execution sequences
- `success_outcomes`: Dictionary mapping agent names to their action success/failure outcomes
- `K`: Number of latent dimensions for the embedding space
- `noise_option`: Type of noise to use in MCMC
- `sigma_beta`: Standard deviation for beta parameters
- `random_seed`: Random seed for reproducibility

**Returns:**
A dictionary containing:
- `agent_partial_orders`: Dictionary mapping agent names to their inferred constraints
- `cross_agent_constraints`: List of inferred cross-agent constraints
- `agent_success_probs`: Dictionary mapping agent names to their action success probabilities

## Visualization Functions

### Visualize Single-Agent Plan

```python
visualize_partial_order(
    actions,
    partial_order_matrix,
    title="Inferred Partial Order"
)
```

Visualizes a partial order as a directed graph.

### Visualize Multi-Agent Plan

```python
visualize_multi_agent_plan(
    agents,
    agent_actions,
    agent_partial_orders,
    cross_agent_constraints,
    title="Multi-Agent Partial Order Plan"
)
```

Visualizes a multi-agent plan as a directed graph with different colors for different agents and special edges for cross-agent constraints.

## Example Usage

See `examples/disaster_recovery_planning.py` for complete examples of:
1. Single-agent disaster recovery planning
2. Multi-agent disaster recovery planning

## Mathematical Background

The Bayesian Partial Order Planning framework uses:

1. **Hierarchical Partial Orders**: A representation of partial orders using a latent embedding space
2. **Bayesian Inference**: MCMC sampling to infer posterior distributions over partial orders
3. **Success Probability Estimation**: A logistic regression model to relate action features to success probabilities

For each action with features x, the success probability is modeled as:
```
logit(p) = α + x·β
```
where α is a base success rate and β represents the impact of each feature on success probability.

## Integration with Other Modules

The BPOP module builds on:
- `src/utils/po_fun.py`: Core partial order operations
- `src/utils/po_accelerator_nle.py`: Log-likelihood caching
- `src/mcmc/mcmc_simulation.py`: MCMC simulation for hierarchical partial orders 