import cProfile
import pstats
from pstats import SortKey
import time
import numpy as np
import pandas as pd
from src.mcmc.hpo_po_hm_mcmc_cl import mcmc_simulation_hpo_cluster

def profile_mcmc_detailed(num_iterations=100, output_file="mcmc_profile.prof"):
    """
    Profile the MCMC simulation with detailed analysis
    """
    print(f"Starting MCMC profiling with {num_iterations} iterations...")
    
    # Set up your MCMC parameters (adjust these to match your actual usage)
    # You'll need to replace these with your actual parameter values
    M0 = list(range(10))  # Example: 10 items
    observed_orders = [[[1, 2, 3], [2, 3, 1]] for _ in range(50)]  # Example data
    O_a_i_dict = [[[1, 2, 3], [1, 2, 3]] for _ in range(50)]  # Example choice sets
    
    # Example parameters - adjust to your actual values
    mcmc_params = {
        'num_iterations': num_iterations,
        'M0': M0,
        'O_a_i_dict': O_a_i_dict,
        'observed_orders': observed_orders,
        'sigma_beta': 0.1,
        'X': np.random.randn(5, len(M0)),  # Example covariate matrix
        'dr': 0.1,
        'drrt': 0.1,
        'drbeta': 0.1,
        'sigma_mallow': 0.1,
        'noise_option': 'queue_jump',
        'mcmc_pt': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],  # Update frequencies
        'rho_prior': 0.5,
        'noise_beta_prior': 0.1,
        'mallow_ua': 1.0,
        'K_prior': 3,
        'Dri_alpha': 0.1,
        'Dri_theta': 1.0,
        'random_seed': 42
    }
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Start profiling
    start_time = time.time()
    profiler.enable()
    
    try:
        # Run your MCMC - replace with your actual parameters
        results = mcmc_simulation_hpo_cluster(**mcmc_params)
        
    except Exception as e:
        print(f"Error during MCMC: {e}")
        return None
    finally:
        profiler.disable()
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"MCMC completed in {total_time:.2f} seconds")
    print(f"Time per iteration: {total_time/num_iterations:.4f} seconds")
    
    # Save profile data
    profiler.dump_stats(output_file)
    
    # Analyze the profile
    analyze_profile(output_file, total_time)
    
    return results, profiler

def analyze_profile(profile_file="mcmc_profile.prof", total_time=None):
    """
    Analyze the profile results and identify bottlenecks
    """
    print("\n" + "="*60)
    print("MCMC PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Load profile data
    stats = pstats.Stats(profile_file)
    
    print("\n1. TOP 20 FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 50)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    
    print("\n2. TOP 20 FUNCTIONS BY INDIVIDUAL TIME:")
    print("-" * 50)
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(20)
    
    print("\n3. FUNCTIONS CALLED MOST FREQUENTLY:")
    print("-" * 50)
    stats.sort_stats(SortKey.CALLS)
    stats.print_stats(20)
    
    # Focus on your specific modules
    print("\n4. FOCUS ON YOUR MODULES:")
    print("-" * 50)
    stats.print_stats("po_accelerator_nle|hpo_po_hm_mcmc_cl|po_fun")
    
    # Get bottleneck functions
    print("\n5. BOTTLENECK ANALYSIS:")
    print("-" * 50)
    
    stats.sort_stats(SortKey.CUMULATIVE)
    bottlenecks = []
    
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, function_name = func
        if ct > 0.01 * (total_time or 1):  # Functions taking >1% of total time
            bottlenecks.append({
                'function': f"{filename}:{function_name}",
                'cumulative_time': ct,
                'total_calls': nc,
                'time_per_call': ct/nc if nc > 0 else 0,
                'percentage': (ct/(total_time or 1)) * 100 if total_time else 0
            })
    
    # Sort by cumulative time
    bottlenecks.sort(key=lambda x: x['cumulative_time'], reverse=True)
    
    print("Functions consuming >1% of total time:")
    for i, func in enumerate(bottlenecks[:10], 1):
        print(f"{i:2d}. {func['function']}")
        print(f"    Cumulative time: {func['cumulative_time']:.3f}s ({func['percentage']:.1f}%)")
        print(f"    Calls: {func['total_calls']}, Time per call: {func['time_per_call']:.6f}s")
        print()

def profile_specific_function(func, *args, **kwargs):
    """
    Profile a specific function
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(10)
    
    return result

def quick_profile_likelihood():
    """
    Quick profile of just the likelihood calculation
    """
    from src.utils.po_accelerator_nle import HPO_LogLikelihoodCache
    
    # Create dummy data for testing
    U0 = np.random.randn(10, 3)
    U_a_dict = {1: np.random.randn(5, 3)}
    h_U = {1: np.random.randint(0, 2, (5, 5))}
    observed_orders = {1: [[[1, 2, 3], [2, 1, 3]]]}
    M_a_dict = {1: [1, 2, 3, 4, 5]}
    O_a_i_dict = {1: [[1, 2, 3], [1, 2, 3]]}
    item_to_index = {i: i for i in range(10)}
    
    print("Profiling likelihood calculation...")
    
    result = profile_specific_function(
        HPO_LogLikelihoodCache.calculate_log_likelihood_hpo,
        U={"U0": U0, "U_a_dict": U_a_dict},
        h_U=h_U,
        observed_orders=observed_orders,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        item_to_index=item_to_index,
        prob_noise=0.1,
        mallow_theta=1.0,
        noise_option="queue_jump",
        alpha=np.zeros(10)
    )
    
    return result

if __name__ == "__main__":
    # Run profiling
    print("Choose profiling option:")
    print("1. Full MCMC profiling (slower, comprehensive)")
    print("2. Quick likelihood profiling (faster, focused)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        results, profiler = profile_mcmc_detailed(num_iterations=50)
    elif choice == "2":
        quick_profile_likelihood()
    else:
        print("Invalid choice. Running quick profile...")
        quick_profile_likelihood() 