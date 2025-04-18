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
U_{j,:}^{(0)} \sim \mathcal N\!\bigl(\mathbf 0,\Sigma_\rho\bigr), \quad j\in M_0,
$$

$$
U_{j,:}^{(a)} \mid U_{j,:}^{(0)} \sim \mathcal N\!\Bigl(\tau\,U_{j,:}^{(0)},\,(1-\tau^{2})\Sigma_\rho\Bigr), \quad a\in A,\; j\in M_a,
$$

$$
\eta_{j,:}^{(a)} = G^{-1}\!\bigl(\Phi(U_{j,:}^{(a)})\bigr) + \alpha_j\,\mathbf 1_K^{\!\top}, \quad a = 0,\dots,A,\; j\in M_a,
$$

$$
h^{(a)} = h\!\bigl(\eta^{(a)}\bigr), \quad a = 0,\dots,A.
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
│   └── hpo_debugger_test.py  # Validation tests
├── config/                   # Configuration files
│   └── hpo_mcmc_configuration.yaml
├── notebooks/                # Example analyses
│   └── hpo_mcmc_simulation_betak.ipynb
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
n = 10                      # Number of items in global set
items = list(range(n))      # Global item set

# Define assessor structure
assessors = [1, 2, 3]       # Assessor IDs
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

### 4. Analyzing MCMC Results

After running MCMC, the results can be analyzed for:

- Posterior distributions of hyperparameters (ρ, τ, K, noise parameters)
- Acceptance rates and mixing diagnostics
- Comparisons between true and inferred partial orders

## Configuration Options

The model parameters are specified in `config/hpo_mcmc_configuration.yaml`. Key parameters include:

```yaml
sampling:
 min_tasks_scaler: 2        # For generating data (tasks per assessor)
 min_size: 2                # Minimum size of choice sets

mcmc:
  num_iterations: 10000     # Total number of MCMC iterations
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


### Example Usage

The included `data/load_data_example.py` script demonstrates how to load and use these data files with the hierarchical partial order inference package:

```python
# Load data and configuration
data = prepare_mcmc_input_data()
config = load_config("config/hpo_mcmc_configuration.yaml")

# Choose inference type based on configuration

# Run MCMC inference with configuration parameters
results = run_mcmc_inference(data, config, use_reversible_jump=True)

# Analyze results
analyze_results(results, data, use_reversible_jump)
```

The script handles:

1. Loading data from CSV files into the required format
2. Reading parameters from the configuration file
3. Setting up MCMC parameters based on the configuration
4. Running the appropriate MCMC algorithm (fixed K or reversible-jump)
5. Analyzing and visualizing the results

### Data Format for Your Own Datasets

To use your own data:

1. Format your item characteristics in the same structure as `item_characteristics.csv`
2. Format your observed rankings in the same structure as `observed_rankings.csv`
3. Update your configuration in `config/hpo_mcmc_configuration.yaml` to match your dataset
4. Use the `prepare_mcmc_input_data()` function to convert these files into the format required by the MCMC functions

## Test Cases

The package includes test cases for validating the MCMC implementation:

### No Data Tests

Tests for when no actual ranking data is provided, verifying that the model recovers the prior distributions.

### Extreme Scenario Tests

Tests for edge cases such as:

- When τ is close to 0 (assessors independent from global consensus)
- When τ is close to 1 (assessors perfectly follow global consensus)
- When ρ has extreme values
- When the noise level is very high or low

To run tests:

```bash
python -m tests.hpo_debugger_test
```

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
