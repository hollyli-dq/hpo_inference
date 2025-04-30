#!/usr/bin/env python
import os, sys, pkgutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

# 1) paths
current_dir  = Path.cwd()           # /…/hpo_inference/hpo_inference/notebooks
project_root = current_dir.parents[0]   # /…/hpo_inference
src_dir      = project_root / "src"
utils_dir    = src_dir / "utils"
data_dir      = project_root / "data"

# Define the default configuration path
DEFAULT_CONFIG_PATH = project_root / "config" / "hpo_generator.yaml"

# 2) make sure __init__.py files exist
for d in (src_dir, utils_dir):
    d.mkdir(parents=True, exist_ok=True)
    init_file = d / "__init__.py"
    if not init_file.exists():
        init_file.touch()           # empty file is fine

# 3) add to sys.path once
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

print("✅ Current dir :", current_dir)
print("✅ Project root in PYTHONPATH :", project_root_str)
print("✅ src package found? ->", pkgutil.find_loader("src") is not None)

# 4) now the project‑specific imports
from src.utils.po_fun           import BasicUtils, StatisticalUtils, GenerationUtils
from src.utils.po_fun_plot      import PO_plot
from src.utils.po_accelerator_nle import LogLikelihoodCache, HPO_LogLikelihoodCache

# 5) load config
import yaml, pprint
config_path = project_root / "config" / "hpo_generator.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print("✅ Configuration loaded, top‑level keys:", list(config.keys())[:10])
# --- Utility Functions ---
def generate_subsets(num_assessors, num_items, min_size, max_size, rng):
    """
    Generate item subsets for each assessor.
    
    Args:
        num_assessors: Number of assessors
        num_items: Total number of items
        min_size: Minimum subset size
        max_size: Maximum subset size
        rng: Random number generator
        
    Returns:
        List of item index lists, one for each assessor
    """
    subsets = []
    all_items = list(range(num_items))
    
    for _ in range(num_assessors):
        subset_size = rng.integers(min_size, max_size + 1)
        subset = rng.choice(all_items, size=subset_size, replace=False).tolist()
        subsets.append(subset)
    
    return subsets


# --- Main Generation Function ---
def generate_hpo_data(config: Dict):
    """Generates data according to the Hierarchical Partial Order model based on config."""

    print("Starting data generation...")
    start_time = time.time()

    # 0. Extract Config Sections (with defaults for safety)
    # Use what's available in the config file
    simulation_config = config.get('simulation', {})
    sampling_config = config.get('sampling', {})
    mcmc_config = config.get('mcmc', {})
    noise_config = config.get('noise', {})
    prior_config = config.get('prior', {})
    
    # Construct output paths
    data_dir = project_root / "data"/ config["output_paths"]["dir"]
    plot_dir = os.path.join(data_dir, "plots")
    true_params_path = os.path.join(data_dir, "true_parameters.yaml")
    item_chars_path = os.path.join(data_dir, "item_characteristics.csv")
    rankings_path = os.path.join(data_dir, "observed_rankings.csv")

    # Extract specific parameters with checks/defaults
    K_true = mcmc_config.get('K', 3)
    rho_true = simulation_config.get('true_rho', 0.8)
    tau_true = simulation_config.get('true_tau', 0.7)
    sigma_beta_true = prior_config.get('sigma_beta', 0.5)
    noise_option = noise_config.get('noise_option', 'queue_jump')
    prob_noise_true = simulation_config.get('true_noise_prob', 0.1)
    
    n = simulation_config.get('num_items', 10)
    A = simulation_config.get('num_assessors', 5)
    p = simulation_config.get('num_covariates', 2)
    seed = config.get('random_seed', 42)

    min_tasks_scaler = sampling_config.get('min_tasks_scaler', 2)
    min_choice_set_size = sampling_config.get('min_size', 2)

    do_visualize = True  # Set to True to generate visualizations

    # Setup RNG
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # Ensure output directories exist
    os.makedirs(data_dir, exist_ok=True)
    if do_visualize:
         os.makedirs(plot_dir, exist_ok=True)

    print(f" Parameters: K={K_true}, rho={rho_true}, tau={tau_true}, noise={noise_option}")
    print(f" Setup: {n} items, {A} assessors, {p} covariates, seed={seed}")

    # 1. Generate Items, Assessors, Covariates, True Beta/Alpha
    items = list(range(n))  # Item IDs from 0 to n-1 (using 0-indexing)
    assessors = list(range(1, A + 1))  # Assessor IDs from 1 to A
    X = rng.normal(loc=0.0, scale=1.0, size=(p, n))  # Covariates (p x n)
    beta_true = rng.normal(loc=0.0, scale=sigma_beta_true, size=(p,))
    alpha_true = X.T @ beta_true  # (n x 1)

    # 2. Generate Assessor Subsets (M_a_dict)
    print(" Generating assessor subsets (Ma)...")
    min_subset_size = max(2, int(n * 0.3))
    max_subset_size = min(n, int(n * 0.8))
    
    # Generate subsets for each assessor
    M_a_dict_indices = generate_subsets(A, n, min_subset_size, max_subset_size, rng)
    M_a_dict = {assessor_id: [items[i] for i in indices] 
                for assessor_id, indices in zip(assessors, M_a_dict_indices)}

    # 3. Generate True Latent Variables (U0, U_a_dict)
    print(" Generating true latent variables (U0, Ua)...")
    Sigma_rho_true = BasicUtils.build_Sigma_rho(K_true, rho_true)
    U_global = rng.multivariate_normal(mean=np.zeros(K_true), cov=Sigma_rho_true, size=n)  # (n, K_true)

    U_a_dict = {}
    cov_mat_a = (1.0 - tau_true**2) * Sigma_rho_true
    for a_id in assessors:
        Ma = M_a_dict[a_id]
        n_a = len(Ma)
        Ua = np.zeros((n_a, K_true), dtype=float)
        for i_loc, item_id in enumerate(Ma):
            mean_vec = tau_true * U_global[item_id, :]
            Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat_a)
        U_a_dict[a_id] = Ua

    # 4. Build True Partial Orders (h_U_dict)
    print(" Building true partial orders (h_U)...")
    h_U_dict = StatisticalUtils.build_hierarchical_partial_orders(
        M0=items, assessors=assessors, M_a_dict=M_a_dict,
        U0=U_global, U_a_dict=U_a_dict, alpha=alpha_true, link_inv=None
    )

    # 5. (Optional) Visualize Partial Orders
    if do_visualize:
        print(" Visualizing partial orders...")
        try:
            if 0 in h_U_dict:
                global_po_reduced = BasicUtils.transitive_reduction(h_U_dict[0])
                PO_plot.visualize_partial_order(
                    global_po_reduced, items, 
                    title=f"Global True H (K={K_true})"
                )
                plt.savefig(os.path.join(plot_dir, "global_po.png"))
                plt.close()
                
            for a_id in assessors:
                if a_id in h_U_dict:
                    Ma = M_a_dict[a_id]
                    assessor_po_reduced = BasicUtils.transitive_reduction(h_U_dict[a_id])
                    PO_plot.visualize_partial_order(
                        assessor_po_reduced, Ma, 
                        title=f"Assessor {a_id} True H (K={K_true})"
                    )
                    plt.savefig(os.path.join(plot_dir, f"assessor_{a_id}_po.png"))
                    plt.close()
        except Exception as e:
            print(f" Warning: Plotting failed - {e}")

    # 6. Generate Choice Sets (O_a_i_dict)
    print(" Generating choice sets (Oai)...")
    try:
        O_a_i_dict = GenerationUtils.generate_choice_sets_for_assessors(
            M_a_dict, 
            min_tasks=min_tasks_scaler * n,
            min_size=min_choice_set_size
        )
    except Exception as e:
        print(f" Warning: Using fallback method for choice sets - {e}")
        # Fallback implementation for choice sets
        O_a_i_dict = {}
        for a_id, Ma in M_a_dict.items():
            num_tasks = max(min_tasks_scaler, int(min_tasks_scaler * len(Ma)))
            choice_sets = []
            for _ in range(num_tasks):
                size = rng.integers(min_choice_set_size, len(Ma))
                choice_set = rng.choice(Ma, size=size, replace=False).tolist()
                choice_sets.append(choice_set)
            O_a_i_dict[a_id] = choice_sets

    # 7. Generate Observed Rankings (y_a_i_dict)
    print(f" Generating observed rankings with noise option: {noise_option}...")
    try:
        y_a_i_dict = GenerationUtils.generate_total_orders_for_assessor(
            h_U_dict, M_a_dict, O_a_i_dict, prob_noise_true
        )
    except Exception as e:
        print(f" Error generating rankings: {e}")
        print(" Please check if the method exists in GenerationUtils or update the code.")
        sys.exit(1)

    # 8. Save Generated Data
    print(" Saving generated data...")
    # Save true parameters
    true_params_to_save = {
        'K': K_true,
        'rho': rho_true,
        'tau': tau_true,
        'sigma_beta': sigma_beta_true,
        'noise_option': noise_option,
        'prob_noise': prob_noise_true,
        'num_items': n,
        'num_assessors': A,
        'num_covariates': p,
        'random_seed': seed,
        'beta': beta_true.tolist()
    }
    with open(true_params_path, 'w') as f:
        yaml.dump(true_params_to_save, f, default_flow_style=False)

    # Save item characteristics file
    # Items Section
    item_char_df = pd.DataFrame({
        'item_id': items,
    })
    
    # Add covariates
    for p_idx in range(p):
        item_char_df[f'covariate_{p_idx+1}'] = X[p_idx]
    
    # Assessors Section
    assessors_df = pd.DataFrame({
        'assessor_id': assessors,
        'name': [f"Assessor_{a}" for a in assessors]
    })
    
    # Selective Sets Section
    selective_list = []
    for a_id, item_ids in M_a_dict.items():
        selective_list.append({
            'selective_set_id': a_id,
            'assessor_id': a_id,
            'items': ";".join(map(str, item_ids))
        })
    selective_sets_df = pd.DataFrame(selective_list)
    
    # Save as multi-section CSV
    with open(item_chars_path, 'w') as f:
        # Write items section
        item_char_df.to_csv(f, index=False)
        f.write('\n')  # Blank line
        
        # Write assessors section
        assessors_df.to_csv(f, index=False)
        f.write('\n')  # Blank line
        
        # Write selective sets section
        selective_sets_df.to_csv(f, index=False)

    # Save observed rankings
    try:
        print(rankings_df)
        PO_plot.save_rankings_to_csv(y_a_i_dict, output_file=rankings_path)
    except Exception as e:
        print(f" Warning: Could not save rankings with PO_plot - {e}")
        # Fallback method to save rankings
        rankings_list = []
        ranking_id = 1
        for a_id, rankings in y_a_i_dict.items():
            for task_id, (ranking, choice_set) in enumerate(zip(rankings, O_a_i_dict[a_id])):
                rankings_list.append({
                    'ranking_id': ranking_id,
                    'assessor_id': a_id,
                    'task_id': task_id + 1,
                    'choice_set': ','.join(map(str, choice_set)),
                    'observed_ranking': ','.join(map(str, ranking))
                })
                ranking_id += 1
        rankings_df = pd.DataFrame(rankings_list)

        rankings_df.to_csv(rankings_path, index=False)


    end_time = time.time()
    print(f"... Data generation finished in {end_time - start_time:.2f} seconds.")
    print(f" True parameters saved to: {true_params_path}")
    print(f" Item characteristics saved to: {item_chars_path}")
    print(f" Observed rankings saved to: {rankings_path}")
    if do_visualize:
         print(f" Plots saved to: {plot_dir}")


# --- Execution Guard ---
if __name__ == "__main__":
    # Determine config path (use default or command-line argument)
    config_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    
    print(f"Using configuration file: {config_file}")
    

    # Run generation with loaded config
    generate_hpo_data(config)