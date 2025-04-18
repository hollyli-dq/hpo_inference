import os
import sys
import pandas as pd
import numpy as np
import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mcmc.hpo_po_hm_mcmc import mcmc_simulation_hpo
from src.mcmc.hpo_po_hm_mcmc_k import mcmc_simulation_hpo_k
from src.utils.po_fun_plot import PO_plot

def load_item_characteristics(file_path="data/item_characteristics.csv"):
    """Load item characteristics from CSV file"""
    # Read the items section
    items_df = pd.read_csv(file_path, nrows=10)
    
    # Read the assessors section (skip the items and the blank line)
    assessors_df = pd.read_csv(file_path, skiprows=12, nrows=5)
    
    # Read the selective sets section (skip items, blank line, assessors, blank line)
    selective_sets_df = pd.read_csv(file_path, skiprows=19)
    
    return items_df, assessors_df, selective_sets_df

def load_observed_rankings(file_path="data/observed_rankings.csv"):
    """Load observed rankings from CSV file"""
    rankings_df = pd.read_csv(file_path)
    return rankings_df

def prepare_mcmc_input_data():
    """Prepare data structures for MCMC inference"""
    items_df, assessors_df, selective_sets_df = load_item_characteristics()
    rankings_df = load_observed_rankings()
    
    # Extract item IDs and covariates
    items = items_df['item_id'].tolist()
    
    # Create covariate matrix X (p × n)
    covariate_cols = [col for col in items_df.columns if col.startswith('covariate_')]
    X = items_df[covariate_cols].values.T  # Transpose to get (p × n)
    
    # Extract assessor IDs
    assessors = assessors_df['assessor_id'].tolist()
    
    # Create M_a_dict: assessor_id -> list of items assessed
    M_a_dict = {}
    for _, row in selective_sets_df.iterrows():
        assessor_id = row['assessor_id']
        items_str = row['items']
        items_list = [int(item) for item in items_str.split(';')]
        M_a_dict[assessor_id] = items_list
    
    # Create O_a_i_dict: assessor_id -> list of choice sets
    O_a_i_dict = {}
    for assessor_id in assessors:
        assessor_rankings = rankings_df[rankings_df['assessor_id'] == assessor_id]
        choice_sets = []
        for _, row in assessor_rankings.iterrows():
            choice_set_str = row['choice_set'].strip('"')
            choice_set = [int(item) for item in choice_set_str.split(',')]
            choice_sets.append(choice_set)
        O_a_i_dict[assessor_id] = choice_sets
    
    # Create observed_orders: assessor_id -> list of observed rankings
    observed_orders = {}
    for assessor_id in assessors:
        assessor_rankings = rankings_df[rankings_df['assessor_id'] == assessor_id]
        rankings = []
        for _, row in assessor_rankings.iterrows():
            ranking_str = row['observed_ranking'].strip('"')
            ranking = [int(item) for item in ranking_str.split(',')]
            rankings.append(ranking)
        observed_orders[assessor_id] = rankings
    
    return {
        'M0': items,
        'assessors': assessors,
        'M_a_dict': M_a_dict,
        'O_a_i_dict': O_a_i_dict,
        'observed_orders': observed_orders,
        'X': X
    }

def load_config(config_path="../hpo_inference/config/hpo_mcmc_configuration.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def run_mcmc_inference(data, config, use_reversible_jump=False):
    """Run MCMC inference using the provided data and configuration"""
    # Extract configuration parameters
    K = config["mcmc"]["K"]
    dr = config["rho"]["dr"]
    drrt = config["rhotau"]["drrt"]
    drbeta = 0.1  # Default value if not in config
    noise_option = config["noise"]["noise_option"]
    sigma_mallow = config["noise"]["sigma_mallow"]
    
    prior_config = config["prior"]
    rho_prior = prior_config["rho_prior"]
    noise_beta_prior = prior_config["noise_beta_prior"]
    mallow_ua = prior_config["mallow_ua"]
    K_prior = prior_config["k_prior"]
    
    rho_tau_update = config["reversible_two_factors"]["rho_tau_update"]
    
    # Set up update probabilities
    if use_reversible_jump:
        # For reversible jump MCMC (with K updates)
        mcmc_pt = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1]  # Include K dimension updates
        
        # Run reversible jump MCMC
        results = mcmc_simulation_hpo_k(
            num_iterations=config["mcmc"]["num_iterations_debug"],
            M0=data['M0'],
            assessors=data['assessors'],
            M_a_dict=data['M_a_dict'],
            O_a_i_dict=data['O_a_i_dict'],
            observed_orders=data['observed_orders'],
            sigma_beta=0.5,  # Standard deviation for beta prior
            X=data['X'],
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
    else:
        # For fixed K MCMC
        mcmc_pt = [0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.1]  # No K dimension updates
        
        # Run fixed K MCMC
        results = mcmc_simulation_hpo(
            num_iterations=config["mcmc"]["num_iterations_debug"],
            M0=data['M0'],
            assessors=data['assessors'],
            M_a_dict=data['M_a_dict'],
            O_a_i_dict=data['O_a_i_dict'],
            observed_orders=data['observed_orders'],
            sigma_beta=0.5,  # Standard deviation for beta prior
            X=data['X'],
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
    
    return results

def analyze_results(results, data, use_reversible_jump=False):
    """Analyze MCMC results"""
    # Extract basic statistics
    burn_in = int(len(results['rho_trace']) * 0.3)  # Use 30% as burn-in
    
    print("\n===== MCMC Results =====")
    print(f"Acceptance rate: {results['overall_acceptance_rate']:.2%}")
    print(f"Final rho: {results['rho_final']:.4f}")
    print(f"Final tau: {results['tau_final']:.4f}")
    
    if use_reversible_jump:
        print(f"Final K: {results['K_final']}")
        print(f"K posterior mode: {np.bincount(results['K_trace']).argmax()}")
    
    # Posterior statistics (after burn-in)
    rho_posterior = results['rho_trace'][burn_in:]
    tau_posterior = results['tau_trace'][burn_in:]
    
    print("\n===== Posterior Statistics =====")
    print(f"Rho: mean={np.mean(rho_posterior):.4f}, std={np.std(rho_posterior):.4f}")
    print(f"Tau: mean={np.mean(tau_posterior):.4f}, std={np.std(tau_posterior):.4f}")
    
    # Create visualizations
    # PO_plot.plot_joint_parameters(results)
    # PO_plot.plot_update_acceptance_by_category(results)
    
    # Display inferred partial orders
    h_global = results['H_final'][0]
    print("\n===== Inferred Global Partial Order =====")
    for (i, j) in h_global.items():
        item_i = data['M0'][i]
        item_j = data['M0'][j]
        print(f"Item {item_i} < Item {item_j}")
    
    return rho_posterior, tau_posterior

if __name__ == "__main__":
    # Load data
    data = prepare_mcmc_input_data()
    config = load_config()
    
    # Choose whether to use reversible jump MCMC
    use_reversible_jump = True
    
    # Run MCMC inference
    results = run_mcmc_inference(data, config, use_reversible_jump)
    
    # Analyze results
    analyze_results(results, data, use_reversible_jump) 