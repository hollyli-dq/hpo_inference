# HPO Inference: Function Reference

This document provides comprehensive documentation for the key functions in the Hierarchical Partial Orders (HPO) Inference package.

## Table of Contents

1. [Generation Utilities](#generation-utilities)
2. [Basic Utilities](#basic-utilities)
3. [Statistical Utilities](#statistical-utilities)
4. [Likelihood Acceleration](#likelihood-acceleration)
5. [Visualization Utilities](#visualization-utilities)
6. [MCMC Inference](#mcmc-inference)
7. [Data Processing](#data-processing)

## Generation Utilities

The `GenerationUtils` class provides functions for generating synthetic data for HPO inference.

### `generate_U`

```python
GenerationUtils.generate_U(n: int, K: int, rho_val: float) -> np.ndarray
```

Generates a latent variable matrix U of size n × K from a multivariate normal distribution with zero mean and correlation parameter rho_val.

**Parameters:**
- `n`: Number of observations (items)
- `K`: Number of latent dimensions
- `rho_val`: Correlation value for the covariance matrix

**Returns:**
- An n × K numpy array of latent positions

### `generate_choice_sets_for_assessors`

```python
GenerationUtils.generate_choice_sets_for_assessors(
    M_a_dict: Dict[int, List[int]],
    min_tasks: int = 1,
    min_size: int = 2
) -> Dict[int, List[List[int]]]
```

For each assessor, generates a set of choice sets (subsets of items to be ranked).

**Parameters:**
- `M_a_dict`: Dictionary mapping assessor IDs to their overall list of item IDs
- `min_tasks`: Minimum number of tasks (choice sets) per assessor
- `min_size`: Minimum number of items in each choice set

**Returns:**
- Dictionary mapping assessor IDs to lists of choice sets

### `generate_total_orders_for_assessor`

```python
GenerationUtils.generate_total_orders_for_assessor(
    h_dict: Dict[int, np.ndarray],
    M_a_dict: Dict[int, List[int]],
    O_a_i_dict: Dict[int, List[List[int]]],
    prob_noise: float
) -> Dict[int, List[List[int]]]
```

For each assessor, generates total orders (linear extensions) from their local partial order.

**Parameters:**
- `h_dict`: Dictionary mapping assessor IDs to local partial order matrices
- `M_a_dict`: Dictionary mapping assessor IDs to their ordered list of global item IDs
- `O_a_i_dict`: Dictionary mapping assessor IDs to a list of choice sets
- `prob_noise`: Noise (jump) probability for the queue-jump model

**Returns:**
- Dictionary mapping assessor IDs to lists of total orders

## Basic Utilities

The `BasicUtils` class provides fundamental operations for working with partial orders.

### `build_Sigma_rho`

```python
BasicUtils.build_Sigma_rho(K: int, rho_val: float) -> np.ndarray
```

Constructs a K × K correlation matrix with 1's on the diagonal and rho_val on all off-diagonal elements.

**Parameters:**
- `K`: Dimension of the matrix
- `rho_val`: Value for off-diagonal elements

**Returns:**
- A K × K correlation matrix

### `transitive_reduction`

```python
BasicUtils.transitive_reduction(adj_matrix: np.ndarray) -> np.ndarray
```

Computes the transitive reduction of a partial order (minimizes redundant edges).

**Parameters:**
- `adj_matrix`: An n × n numpy array representing the adjacency matrix of the partial order

**Returns:**
- The transitive reduction as an n × n numpy array

### `transitive_closure`

```python
BasicUtils.transitive_closure(adj_matrix: np.ndarray) -> np.ndarray
```

Computes the transitive closure of a relation (adds all implied edges).

**Parameters:**
- `adj_matrix`: An n × n numpy array representing the adjacency matrix of the relation

**Returns:**
- The transitive closure as an n × n numpy array

### `nle`

```python
BasicUtils.nle(tr: np.ndarray) -> int
```

Counts the number of linear extensions of a partial order.

**Parameters:**
- `tr`: An n × n numpy array representing the adjacency matrix of the transitive reduction

**Returns:**
- The number of linear extensions as an integer

### `num_extensions_with_first`

```python
BasicUtils.num_extensions_with_first(tr: np.ndarray, first_item_idx: int) -> int
```

Counts how many linear extensions of a partial order have a specific item as the first element.

**Parameters:**
- `tr`: Adjacency matrix of the partial order
- `first_item_idx`: Index of the item that should be first

**Returns:**
- The number of linear extensions with the specified first item

## Statistical Utilities

The `StatisticalUtils` class provides functions for statistical computations in HPO inference.

### `build_hierarchical_partial_orders`

```python
StatisticalUtils.build_hierarchical_partial_orders(
    M0: List[int],
    assessors: List[int],
    M_a_dict: Dict[int, List[int]],
    U0: np.ndarray,
    U_a_dict: Dict[int, np.ndarray],
    alpha: np.ndarray,
    link_inv: Optional[Callable] = None
) -> Dict[int, np.ndarray]
```

Constructs hierarchical partial orders from latent positions according to Theorem 9.

**Parameters:**
- `M0`: Global set of items
- `assessors`: List of assessor IDs
- `M_a_dict`: Dictionary mapping assessor IDs to their item sets
- `U0`: Global latent positions matrix (n_global × K)
- `U_a_dict`: Dictionary of assessor-specific latent matrices
- `alpha`: Item effects vector
- `link_inv`: Inverse link function (default: Gumbel quantile)

**Returns:**
- Dictionary mapping IDs (0 for global, other IDs for assessors) to partial order matrices

### Prior Sampling Functions

These functions sample from the prior distributions for various parameters:

```python
# Sample from prior for rho (correlation parameter)
StatisticalUtils.rRprior(fac: float = 1/6, tol: float = 1e-4) -> float

# Sample from prior for noise probability
StatisticalUtils.rPprior(noise_beta_prior: float) -> float

# Sample from prior for tau (hierarchy strength)
StatisticalUtils.rTauprior() -> float

# Sample from prior for K (latent dimension)
StatisticalUtils.rKprior(lam: float = 3.0) -> int

# Sample from prior for beta (covariate effects)
StatisticalUtils.rBetaPrior(sigma_beta: float, p: int) -> np.ndarray
```

### `generate_total_order_for_choice_set_with_queue_jump`

```python
StatisticalUtils.generate_total_order_for_choice_set_with_queue_jump(
    subset: List[int],
    M_a: List[int],
    h_local: np.ndarray,
    prob_noise: float
) -> List[int]
```

Generates a total order for a specific choice set using the queue-jump noise model.

**Parameters:**
- `subset`: List of global item IDs to order
- `M_a`: List of all global item IDs for the assessor
- `h_local`: Local partial order matrix for the assessor
- `prob_noise`: Probability of taking a random jump

**Returns:**
- A total order as a list of global item IDs

## Likelihood Acceleration

The `HPO_LogLikelihoodCache` class provides optimized likelihood calculations with caching.

### `calculate_log_likelihood_hpo`

```python
HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
    U: Dict[int, np.ndarray],
    h_U: Dict[int, np.ndarray],
    observed_orders: Dict[int, List[List[int]]],
    M_a_dict: Dict[int, List[int]],
    O_a_i_dict: Dict[int, List[List[int]]],
    item_to_index: Dict[int, int],
    prob_noise: float,
    mallow_theta: float,
    noise_option: str,
    alpha: np.ndarray
) -> float
```

Calculates the log-likelihood of observed rankings given the HPO model parameters.

**Parameters:**
- `U`: Dictionary of latent variables
- `h_U`: Dictionary of partial order matrices
- `observed_orders`: Dictionary of observed rankings
- `M_a_dict`: Dictionary mapping assessor IDs to their item sets
- `O_a_i_dict`: Dictionary mapping assessor IDs to choice sets
- `item_to_index`: Mapping from item ID to index
- `prob_noise`: Noise probability
- `mallow_theta`: Mallows model parameter
- `noise_option`: Noise model type ('queue_jump' or 'mallows_noise')
- `alpha`: Item effects vector

**Returns:**
- Log-likelihood value (float)

## Visualization Utilities

The `PO_plot` class provides functions for visualizing results.

### `save_rankings_to_csv`

```python
PO_plot.save_rankings_to_csv(
    y_a_i_dict: Dict[int, Union[List[List[int]], Dict[int, List[List[int]]]]],
    output_file: str = 'data/observed_rankings.csv'
) -> None
```

Saves observed rankings to a CSV file.

**Parameters:**
- `y_a_i_dict`: Dictionary of observed rankings
- `output_file`: Path to output CSV file

### `visualize_partial_order`

```python
PO_plot.visualize_partial_order(
    final_h: np.ndarray,
    Ma_list: list,
    title: str = None,
    ax: Optional[plt.Axes] = None
) -> None
```

Visualizes a partial order as a directed graph.

**Parameters:**
- `final_h`: Adjacency matrix for the partial order
- `Ma_list`: Labels for the nodes
- `title`: Title for the plot
- `ax`: Matplotlib axes to draw on (optional)

### `plot_inferred_variables`

```python
PO_plot.plot_inferred_variables(
    mcmc_results: Dict[str, Any],
    true_param: Dict[str, Any],
    config: Dict[str, Any],
    burn_in: int = 100,
    output_filename: str = "inferred_parameters.pdf",
    output_filepath: str = ".",
    assessors: Optional[List[int]] = None,
    M_a_dict: Optional[Dict[int, Any]] = None
) -> None
```

Plots MCMC traces and posterior densities for inferred parameters.

**Parameters:**
- `mcmc_results`: MCMC results dictionary
- `true_param`: Dictionary of true parameter values
- `config`: Configuration dictionary
- `burn_in`: Number of burn-in iterations to exclude
- `output_filename`: Output filename
- `output_filepath`: Output directory
- `assessors`: List of assessor IDs (optional)
- `M_a_dict`: Assessor-to-items mapping (optional)

### `compare_and_visualize_global`

```python
PO_plot.compare_and_visualize_global(
    h_true_global: np.ndarray,
    h_inferred_global: np.ndarray,
    index_to_item_global: Dict[int, int],
    global_Ma_list: List[str],
    do_transitive_reduction: bool = True
) -> None
```

Compares and visualizes true and inferred global partial orders.

**Parameters:**
- `h_true_global`: True global partial order matrix
- `h_inferred_global`: Inferred global partial order matrix
- `index_to_item_global`: Mapping from indices to items
- `global_Ma_list`: List of global item labels
- `do_transitive_reduction`: Whether to apply transitive reduction

### `plot_joint_parameters`

```python
PO_plot.plot_joint_parameters(mcmc_results: Dict[str, Any]) -> None
```

Creates scatter plots to visualize the joint behavior of parameters.

**Parameters:**
- `mcmc_results`: MCMC results dictionary

### `plot_update_acceptance_by_category`

```python
PO_plot.plot_update_acceptance_by_category(
    mcmc_results: Dict[str, Any],
    desired_order: Optional[List[str]] = None,
    jitter_strength: float = 0.08
) -> None
```

Plots the acceptance rates of different types of MCMC updates.

**Parameters:**
- `mcmc_results`: MCMC results dictionary
- `desired_order`: Order of update categories (optional)
- `jitter_strength`: Amount of jitter for visualization

## MCMC Inference

The main MCMC algorithms are implemented in separate modules.

### `mcmc_simulation_hpo`

```python
mcmc_simulation_hpo(
    num_iterations: int,
    M0: List[int],
    assessors: List[int],
    M_a_dict: Dict[int, List[int]],
    O_a_i_dict: Dict[int, List[List[int]]],
    observed_orders: Dict[int, List[List[int]]],
    sigma_beta: np.ndarray,
    X: np.ndarray,
    K: int,
    dr: float,
    drrt: float,
    drbeta: float,
    sigma_mallow: float,
    noise_option: str,
    mcmc_pt: List[float],
    rho_prior: float,
    noise_beta_prior: float,
    mallow_ua: float,
    rho_tau_update: bool = False,
    random_seed: int = 42
) -> Dict[str, Any]
```

Performs MCMC inference for the HPO model with fixed dimension K.

**Key Parameters:**
- `num_iterations`: Number of MCMC iterations
- `K`: Fixed latent dimension
- `noise_option`: Noise model type ('queue_jump' or 'mallows_noise')
- `mcmc_pt`: Update probabilities for different parameters

**Returns:**
- Dictionary with traces for all parameters, acceptance rates, and timing information

### `mcmc_simulation_hpo_k`

```python
mcmc_simulation_hpo_k(
    # Same parameters as mcmc_simulation_hpo, plus:
    K_prior: float,
    # mcmc_pt should include probability for K updates
) -> Dict[str, Any]
```

Performs reversible-jump MCMC inference with varying dimension K.

**Additional Parameters:**
- `K_prior`: Prior parameter for dimension K
- `mcmc_pt`: Should include probability for K dimension updates

**Returns:**
- Dictionary with traces for all parameters, including K

## Data Processing

Utilities for loading and processing data files.

### `prepare_mcmc_input_data`

```python
prepare_mcmc_input_data() -> Dict[str, Any]
```

Loads data from CSV files and prepares structures for MCMC inference.

**Returns:**
- Dictionary with `M0`, `assessors`, `M_a_dict`, `O_a_i_dict`, `observed_orders`, and `X`.

### `load_config`

```python
load_config(config_path: str) -> Dict[str, Any]
```

Loads YAML configuration file.

**Parameters:**
- `config_path`: Path to the configuration file

**Returns:**
- Configuration dictionary

### `run_mcmc_inference`

```python
run_mcmc_inference(
    data: Dict[str, Any],
    config: Dict[str, Any],
    use_reversible_jump: bool = False
) -> Dict[str, Any]
```

Runs MCMC inference with the provided data and configuration.

**Parameters:**
- `data`: Data dictionary from `prepare_mcmc_input_data`
- `config`: Configuration dictionary from `load_config`
- `use_reversible_jump`: Whether to use reversible-jump MCMC

**Returns:**
- MCMC results dictionary 