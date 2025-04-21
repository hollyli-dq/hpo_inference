# Hierarchical Partial Orders (HPO) Inference

A Bayesian framework for inferring hierarchical partial orders from ranking data. This implementation is based on research from the paper "Partial Order Hierarchies and Rank-Order Data".

## Background

### Partial Order Hierarchies and Rank-Order Data

In rank-order data, assessors give preference orders over choice sets. An order is registered as a list giving the elements of its choice set in order from best to worst. Well known parametric models for list-data include the Mallows model and the Plackett-Luce model. These models seek a total order which is "central" to the lists provided by the assessors. Extensions model the list-data as realisations of a mixture of distributions each centred on a total order.

Recent work has relaxed the requirement that the centering order be a total order and instead centre the random lists on a partial order. Lists are random linear extensions of a partial order or linear extensions observed with noise. We give a new hierarchical model for partial orders to handle list data which come in labeled groups. The model reduces to a Plackett-Luce model when the partial order dimension is set equal one and can be used to cluster unlabeled list data.

We carry out Bayesian inference for the poset hierarchy using MCMC. Evaluation of the likelihood costs #P so applications are restricted to choice sets of up to 20 elements.

**Keywords**: Bayesian Inference, Partial Orders, Linear Extensions, Hierarchical Model, Clustering.

## Overview

This project implements a Bayesian approach to infer latent hierarchical partial orders from observed rankings/preferences. It uses Markov Chain Monte Carlo (MCMC) methods to perform posterior inference on model parameters.

### Theorem 9 — *Hierarchical Partial‑Order (HPO)*

Let $\alpha_M = X\beta$ and $\Sigma_\rho$ be as in Section 3; $0 < \tau \le 1$; $M_a \in \mathcal B_M$ for each assessor $a\in A$; and $M_0 = \displaystyle\bigcup_{a=1}^{A} M_a$.

#### Latent Gaussian hierarchy

$$
U_{j,:}^{(0)} \sim \mathcal N\bigl(\mathbf 0,\Sigma_\rho\bigr), \quad j\in M_0,
$$

$$
U_{j,:}^{(a)} \mid U_{j,:}^{(0)} \sim \mathcal N\!\Bigl(\tau\,U_{j,:}^{(0)},\,(1-\tau^{2})\Sigma_\rho\Bigr), \quad a\in A,\; j\in M_a,
$$

$$
\eta_{j,:}^{(a)} = G^{-1}\!\bigl(\Phi(U_{j,:}^{(a)})\bigr) + \alpha_j\,\mathbf 1_K^{\!\top}, \quad a = 0,\dots,A,\; j\in M_a,
$$

$$
h^{(a)} = h\bigl(\eta^{(a)}\bigr), \quad a = 0,\dots,A.
$$

Define $\eta(U,\beta)=\bigl(\eta^{(0)},\eta^{(1)},\dots,\eta^{(A)}\bigr)$ and $h = h\!\bigl(\eta(U,\beta)\bigr)$.

## Features

- **Hierarchical Partial Order Generation**: Create synthetic hierarchical partial order structures with global and assessor-specific components
- **Data Input/Output**: Define assessor item sets and generate/load ranking data from various formats
- **Total Order Generation**: Generate observed total orders from hierarchical partial orders with configurable noise models
- **MCMC Inference**:
  - Fixed-dimension MCMC inference with specified latent dimension K
  - Reversible-jump MCMC for dimension inference (dynamic K)
- **Flexible Noise Models**: Support for different noise models (queue-jump, Mallows)
- **Posterior Analysis**: Analyze MCMC output for hyperparameter distributions and structural recovery
- **Configuration-based**: Easy parameter tuning via YAML configuration files

## Project Structure

```
hpo_inference/
├── src/
│   ├── mcmc/                 # MCMC implementation
│   │   ├── hpo_po_hm_mcmc.py # Fixed K implementation
│   │   └── hpo_po_hm_mcmc_k.py # Reversible-jump implementation
│   └── utils/                # Core utilities
│       ├── po_fun.py         # Partial order functions
│       ├── po_fun_plot.py    # Visualization utilities
│       └── po_accelerator_nle.py # Likelihood acceleration
├── tests/                    # Test cases
│   ├── hpo_debugger_test.py  # Parameter validation tests
│   └── extreme_case_test.py  # Tests for edge cases
├── config/                   # Configuration files
│   └── hpo_mcmc_configuration.yaml
├── notebooks/                # Example analyses
│   └── hpo_mcmc_simulation.ipynb
└── data/
    └── observed_rankings.csv # Generated rankings
```

## Installation

Create a conda environment with the required dependencies:

```bash
conda env create -f environment.yml
conda activate hpo_env
```

## Main Functions and Usage

### 1. Generating Hierarchical Partial Orders

```python
# Define basic parameters
K = 3                       # Latent dimension
rho_prior = 0.1667          # Prior parameter for correlation
n = len(items_df['item_id'].unique())      # Number of items in global set
items = list(range(n))      # Global item set

# Define assessor structure
assessors = assessors_df['assessor_id'].tolist()       # Assessor IDs
M_a_dict = {
    1: [0, 2, 4, 6, 8],     # Assessor 1 evaluates even-indexed items
    2: [1, 3, 5, 7, 9],     # Assessor 2 evaluates odd-indexed items
    3: [0, 1, 2, 3, 4]      # Assessor 3 evaluates first five items
}

# Generate the hierarchical partial orders
h_U_dict = StatisticalUtils.build_hierarchical_partial_orders(
    M0=items,  
    assessors=assessors,   
    M_a_dict=M_a_dict,   
    U0=U_global,           # Global latent positions
    U_a_dict=U_a_dict,     # Assessor-specific latent positions
    alpha=alpha            # Optional covariate effects
)
```

### 2. Generating Total Order Lists

```python
# Generate choice sets for assessors
O_a_i_dict = GenerationUtils.generate_choice_sets_for_assessors(
    M_a_dict,             # Item sets per assessor
    min_tasks=n*2,        # Number of tasks per assessor
    min_size=2            # Minimum items per choice set
)

# Generate observed total orders from the hierarchical partial orders
y_a_i_dict = GenerationUtils.generate_total_orders_for_assessor(
    h_U_dict,            # Hierarchical partial orders
    M_a_dict,            # Item sets per assessor
    O_a_i_dict,          # Choice sets per assessor
    prob_noise_true      # Noise probability
)

# Save rankings to CSV for later use
save_rankings_to_csv(y_a_i_dict, output_file='data/observed_rankings.csv')
```

### 3. MCMC Inference

#### Fixed K Inference

```python
# Define parameters from configuration
mcmc_pt = [
    config["mcmc"]["update_probabilities"]["rho"],
    config["mcmc"]["update_probabilities"]["tau"],
    config["mcmc"]["update_probabilities"]["rho_tau"],
    config["mcmc"]["update_probabilities"]["noise"],
    config["mcmc"]["update_probabilities"]["U_0"],
    config["mcmc"]["update_probabilities"]["U_a"],
    config["mcmc"]["update_probabilities"]["beta"]
]

# Run MCMC with fixed K
mcmc_results = mcmc_simulation_hpo(
    num_iterations=num_iterations,
    M0=items,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
    observed_orders=y_a_i_dict,
    sigma_beta=sigma_beta,  
    X=X,  
        K=K,    
        dr=dr,
        drrt=drrt,
    drbeta=drbeta,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
    mcmc_pt=mcmc_pt,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=42
    )
```

#### Reversible-Jump K Inference

```python
# Update probabilities including K moves
mcmc_pt = [
    config["mcmc"]["update_probabilities"]["rho"],
    config["mcmc"]["update_probabilities"]["tau"],
    config["mcmc"]["update_probabilities"]["rho_tau"],
    config["mcmc"]["update_probabilities"]["noise"],
    config["mcmc"]["update_probabilities"]["U_0"],
    config["mcmc"]["update_probabilities"]["U_a"],
    config["mcmc"]["update_probabilities"]["K"],
    config["mcmc"]["update_probabilities"]["beta"]
]

# Run MCMC with variable K (reversible jump)
mcmc_results_k = mcmc_simulation_hpo_k(
        num_iterations=num_iterations,
    M0=items,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
    observed_orders=y_a_i_dict,
    sigma_beta=sigma_beta,  
    X=X,  
        dr=dr,
        drrt=drrt,
    drbeta=drbeta,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
    mcmc_pt=mcmc_pt,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        K_prior=K_prior,
        rho_tau_update=rho_tau_update,
        random_seed=42
    )
```

## Test Cases

The package includes comprehensive test cases for validating the MCMC implementation. All tests output diagnostic plots to the `tests/hpo_test_output` directory.

### Running Tests

Tests are organized by parameter and edge case scenarios. To run specific tests, use:

```bash
# To run tests for specific parameters:
python -m tests.hpo_debugger_test rho       # Test rho parameter updates
python -m tests.hpo_debugger_test tau       # Test tau parameter updates
python -m tests.hpo_debugger_test rhotau    # Test joint rho-tau updates
python -m tests.hpo_debugger_test p         # Test probability of noise updates
python -m tests.hpo_debugger_test theta     # Test Mallows theta parameter
python -m tests.hpo_debugger_test k         # Test dimension parameter K (reversible jump)
python -m tests.hpo_debugger_test u0        # Test global latent positions
python -m tests.hpo_debugger_test ua        # Test assessor-specific latent positions
python -m tests.hpo_debugger_test beta      # Test covariate parameter updates

# To run extreme case tests:
python -m tests.extreme_case_test
```

### Parameter Tests (hpo_debugger_test.py)

The `hpo_debugger_test.py` script includes multiple functions for testing different MCMC parameter updates:

- **check_rho()**: Tests if the correlation parameter ρ follows its theoretical posterior distribution (truncated Beta). Outputs histograms and diagnostic plots.
- **check_tau()**: Tests if the hierarchical coupling parameter τ follows its theoretical distribution (Uniform). Outputs histograms with theoretical overlays.
- **check_rho_tau()**: Tests joint updates of ρ and τ parameters using reversible proposals. Checks acceptance rates and distribution accuracy.
- **check_P()**: Tests updates of probability noise parameter for queue-jump noise model. Validates against theoretical Beta distribution.
- **check_theta()**: Tests Mallows theta parameter updates for the Mallows noise model. Verifies exponential prior recovery.
- **check_U0()**: Tests global latent position updates. Checks for correct correlation structure and marginal distributions.
- **check_Ua()**: Tests assessor-specific latent positions. Verifies the hierarchical relationship: U_a ~ N(τU_0, (1-τ²)Σ).
- **check_K()**: Tests reversible jump dimension updates. Validates acceptance rates and dimension posterior probabilities.
- **check_beta()**: Tests covariate parameter updates. Tests with different step sizes to diagnose acceptance rate issues.

Each test function outputs:

- Distribution histograms with theoretical overlays
- Trace plots showing MCMC mixing
- Log-likelihood diagnostics
- Acceptance rate statistics

### Extreme Case Tests (extreme_case_test.py)

The `extreme_case_test.py` script tests edge cases:

- **TAU=0 Scenario**: When τ=0, assessor-specific latent positions become independent from global positions. This tests whether U_a ~ N(0, Σ) when τ=0.
- **TAU=1 Scenario**: When τ=1, assessor-specific latent positions exactly match global positions. Tests if U_a = U_0 when τ=1.

This test showcases how the hierarchical model handles these extreme parameter settings and validates the theoretical properties of the model.

## Configuration Options

The model parameters are specified in `config/hpo_mcmc_configuration.yaml`. Key parameters include:

```yaml
sampling:
 min_tasks_scaler: 2        # For generating data (tasks per assessor)
 min_size: 2                # Minimum size of choice sets

mcmc:
  num_iterations: 50000     # Total number of MCMC iterations
  K: 3                      # Latent dimension for fixed-K inference
  update_probabilities:
    rho: 0.1                # Proportion of iterations to update rho
    tau: 0.1                # Proportion of iterations to update tau
    rho_tau: 0.1            # Proportion of iterations to update rho and tau
    noise: 0.1              # Proportion of iterations to update noise parameters
    U_0: 0.2                # Proportion of iterations to update global latent variables
    U_a: 0.2                # Proportion of iterations to update assessor-specific latent variables
    K: 0.1                  # Proportion of iterations to update K (for reversible jump)
    beta: 0.1               # Proportion of iterations to update covariates
```

To enable/disable reversible jump K inference:

- For fixed K: Set `mcmc.update_probabilities.K` to 0 or exclude K from the mcmc_pt list
- For variable K: Include a positive value for `mcmc.update_probabilities.K` and include it in mcmc_pt

## Data Files

This section describes the data files structure for use with the Hierarchical Partial Order Inference package.

### File Structure

#### 1. `item_characteristics.csv`

This file contains information about the items, assessors, and selective sets:

- **Items Section**:

  - `item_id`: Unique identifier for each item (0-9)
  - `label`: Descriptive label for each item (e.g., "Product A")
  - `covariate_1` to `covariate_5`: Five-dimensional covariates for each item
- **Assessors Section**:

  - `assessor_id`: Unique identifier for each assessor (1-5)
  - `name`: Name of the assessor
- **Selective Sets Section**:

  - `selective_set_id`: Unique identifier for each selective set
  - `assessor_id`: ID of the assessor associated with this selective set
  - `items`: Semicolon-separated list of item IDs that the assessor evaluates

#### 2. `observed_rankings.csv`

This file contains the observed total order rankings:

- `ranking_id`: Unique identifier for each ranking
- `assessor_id`: ID of the assessor who provided this ranking
- `task_id`: Task identifier
- `choice_set`: Comma-separated list of item IDs in the choice set
- `observed_ranking`: Comma-separated list of item IDs in the observed ranking order (best to worst)

## Example Notebook

The `notebooks/hpo_mcmc_simulation.ipynb` notebook demonstrates a complete workflow:

1. Loading configuration and setting up parameters
2. Generating synthetic hierarchical partial orders
3. Creating observed total orders with noise
4. Running MCMC inference (both fixed K and reversible-jump)
5. Analyzing the posterior distributions
6. Comparing inferred vs. true partial orders

## Performance Notes

MCMC runtime varies based on problem size:

- ~50 minutes for 200,000 iterations with n=12 items, N=50 tasks
- ~40 minutes for 200,000 iterations with n=5 items
- ~46 minutes for 500,000 iterations with n=5 items, N=10 tasks

## References

1. Chuxuan Jiang and Geoff K. Nicholls. Bayesian inference for partial orders with ties from ranking data with a Placket-Luce distribution centred on random linear extensions, 2021.
2. Chuxuan Jiang, Geoff K. Nicholls, and Jeong Eun Lee. Bayesian inference for vertex-series-parallel partial orders. In Robin J. Evans and Ilya Shpitser, editors, *Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence*, volume 216 of Proceedings of Machine Learning Research, pages 995–1004. PMLR, 31 Jul–04 Aug 2023. URL https://proceedings.mlr.press/v216/jiang23b.html.
