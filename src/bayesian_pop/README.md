# Bayesian Partial Order Planning (POP)

## Overview

Bayesian Partial Order Planning is a framework for inferring action ordering constraints and success probabilities from observed execution sequences. The framework uses Markov Chain Monte Carlo (MCMC) methods to perform Bayesian inference on partial orders.

## Key Concepts

### Partial Orders

A partial order is a binary relation over a set that is reflexive, antisymmetric, and transitive. In the context of planning, a partial order represents constraints on the order in which actions must be executed.

For example, in a medical emergency response:
- We must assess casualties before establishing triage
- We must establish triage before treating critical patients
- We must request medical supplies before treating moderate injuries

These constraints form a partial order because not all actions need to be ordered relative to each other (e.g., requesting medical supplies can happen in parallel with assessing casualties).

### Representation

Partial orders are represented as directed acyclic graphs (DAGs) where:
- Nodes represent actions
- Edges represent ordering constraints (action i must occur before action j)

Mathematically, we use adjacency matrices H where:
- H[i,j] = 1 if action i must occur before action j
- H[i,j] = 0 otherwise

### Bayesian Inference

The goal is to infer:
1. The most likely partial order given observed execution sequences
2. The success probabilities of each action

We use Bayesian inference to:
- Start with prior distributions over partial orders and success probabilities
- Update these distributions based on observed execution sequences
- Obtain posterior distributions that capture our updated beliefs

### MCMC Simulation

Markov Chain Monte Carlo (MCMC) is used to approximate the posterior distributions:
- We iteratively propose changes to the partial order and success probabilities
- We accept or reject proposals based on how well they explain the observed data
- Over time, the MCMC chain converges to the posterior distribution

## Core Functions

The main functions in this module are:

### `mcmc_simulation_bayesian_pop`

Runs MCMC simulation for Bayesian Partial Order Planning for a single agent.

### `multi_agent_bayesian_pop`

Extends the framework to multiple agents, where each agent has its own set of actions and constraints, with possible cross-agent dependencies.

### `generate_synthetic_observations`

Generates synthetic observation data based on a true partial order and success probabilities.

### Visualization Functions

- `visualize_partial_order`: Visualizes a partial order as a directed graph.
- `visualize_multi_agent_plan`: Visualizes a multi-agent plan with cross-agent dependencies.

## Usage

See the `examples` directory for detailed examples of using Bayesian POP for disaster recovery planning, manufacturing process optimization, and other applications.

## References

- Chung, S. H., et al. (2014). Bayesian networks applied to process monitoring.
- Jeon, H., et al. (2022). Hierarchical Partial Order Planning.
- Shah, A., et al. (2020). Planning with uncertain specifications. 