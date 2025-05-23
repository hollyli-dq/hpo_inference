#!/usr/bin/env python3
"""
Simple script to profile your MCMC run
Usage: python profile_my_mcmc.py
"""

import cProfile
import pstats
from pstats import SortKey
import time
import sys
import os

# Add your source directory to path
sys.path.append('src')

from src.mcmc.hpo_po_hm_mcmc_cl import mcmc_simulation_hpo_cluster

def profile_your_mcmc():
    """
    Replace the parameters below with your actual MCMC parameters
    """
    
    # TODO: Replace these with your actual parameters from your notebook
    # Copy these values from where you currently call mcmc_simulation_hpo_cluster
    
    # Example parameters - REPLACE WITH YOUR ACTUAL VALUES
    num_iterations = 50  # Start small for profiling
    M0 = []  # Your global items
    O_a_i_dict = []  # Your choice sets
    observed_orders = []  # Your observed rankings
    sigma_beta = 0.1
    X = None  # Your covariate matrix
    dr = 0.1
    drrt = 0.1  
    drbeta = 0.1
    sigma_mallow = 0.1
    noise_option = "queue_jump"
    mcmc_pt = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    rho_prior = 0.5
    noise_beta_prior = 0.1
    mallow_ua = 1.0
    K_prior = 3
    Dri_alpha = 0.1
    Dri_theta = 1.0
    random_seed = 42
    
    print("="*60)
    print("PROFILING YOUR MCMC CODE")
    print("="*60)
    print(f"Running {num_iterations} iterations...")
    print("This will show you exactly where time is being spent.")
    print()
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Start timing and profiling
    start_time = time.time()
    profiler.enable()
    
    try:
        # Run your MCMC
        results = mcmc_simulation_hpo_cluster(
            num_iterations=num_iterations,
            M0=M0,
            O_a_i_dict=O_a_i_dict,
            observed_orders=observed_orders,
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
            Dri_alpha=Dri_alpha,
            Dri_theta=Dri_theta,
            random_seed=random_seed
        )
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure to update the parameters in this script with your actual values!")
        return None
        
    finally:
        profiler.disable()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✓ MCMC completed in {total_time:.2f} seconds")
    print(f"✓ Average time per iteration: {total_time/num_iterations:.4f} seconds")
    print()
    
    # Save detailed profile
    profiler.dump_stats("mcmc_detailed_profile.prof")
    
    # Analyze the results
    print("="*60)
    print("PERFORMANCE BOTTLENECKS - TOP 15 FUNCTIONS")
    print("="*60)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(15)
    
    print("\n" + "="*60)
    print("FOCUS ON YOUR CODE MODULES")
    print("="*60)
    
    # Filter to show only your code (not standard library)
    stats.print_stats("po_accelerator|mcmc_cl|po_fun|mallow")
    
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    # Get the hot spots
    hot_spots = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, function_name = func
        if ct > 0.02 * total_time:  # Functions taking >2% of total time
            hot_spots.append((function_name, filename, ct, nc, ct/nc if nc > 0 else 0))
    
    hot_spots.sort(key=lambda x: x[2], reverse=True)  # Sort by cumulative time
    
    print("Functions to optimize (taking >2% of total time):")
    for i, (func_name, filename, cum_time, calls, time_per_call) in enumerate(hot_spots[:10], 1):
        percentage = (cum_time / total_time) * 100
        print(f"{i:2d}. {func_name}")
        print(f"    File: {os.path.basename(filename)}")
        print(f"    Time: {cum_time:.3f}s ({percentage:.1f}% of total)")
        print(f"    Calls: {calls:,}, Avg: {time_per_call:.6f}s per call")
        
        # Specific recommendations
        if "nle" in func_name.lower():
            print("    💡 TIP: This is computing linear extensions - try caching or Numba optimization")
        elif "transitive_reduction" in func_name.lower():
            print("    💡 TIP: Graph operations - consider using faster graph libraries or caching")
        elif "calculate_log_likelihood" in func_name.lower():
            print("    💡 TIP: Main likelihood computation - look for vectorization opportunities")
        elif "build_hierarchical_partial_orders" in func_name.lower():
            print("    💡 TIP: Matrix construction - consider NumPy vectorization")
        elif any(x in func_name.lower() for x in ['loop', 'enumerate', 'list']):
            print("    💡 TIP: Python loops detected - candidate for Numba or vectorization")
        
        print()
    
    print("NEXT STEPS:")
    print("1. Focus on the top 2-3 functions above")
    print("2. For graph/matrix operations: Try Numba @jit decoration")
    print("3. For repeated computations: Add caching")
    print("4. For Python loops: Look for NumPy vectorization opportunities")
    print("5. Run this profiler again after each optimization to measure improvement")
    
    return results

if __name__ == "__main__":
    results = profile_your_mcmc() 