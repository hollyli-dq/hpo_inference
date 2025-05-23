"""
Common optimization techniques for MCMC bottlenecks
Use these examples after profiling identifies your hot spots
"""

import numpy as np
from numba import jit
import functools

# ================================================================
# 1. CACHING EXPENSIVE COMPUTATIONS
# ================================================================

class LRUCache:
    """Simple LRU cache implementation"""
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.access_order = []
        self.maxsize = maxsize
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)

# Example: Cache expensive matrix computations
matrix_cache = LRUCache(maxsize=500)

def cached_matrix_operation(matrix):
    """Cache expensive matrix operations"""
    # Create a hashable key from matrix
    key = matrix.tobytes()
    
    # Check cache first
    result = matrix_cache.get(key)
    if result is not None:
        return result
    
    # Compute expensive operation
    result = your_expensive_matrix_function(matrix)
    
    # Store in cache
    matrix_cache.put(key, result)
    return result

# ================================================================
# 2. NUMBA JIT COMPILATION FOR LOOPS
# ================================================================

# BEFORE: Slow Python loops
def slow_likelihood_computation(data, params):
    total = 0.0
    for i in range(len(data)):
        for j in range(len(params)):
            total += data[i] * params[j] * np.exp(data[i] - params[j])
    return total

# AFTER: Fast Numba compilation
@jit(nopython=True)
def fast_likelihood_computation(data, params):
    total = 0.0
    for i in range(len(data)):
        for j in range(len(params)):
            total += data[i] * params[j] * np.exp(data[i] - params[j])
    return total

# Example for your specific code:
@jit(nopython=True)
def fast_adjacency_matrix(U_matrix):
    """Fast computation of adjacency matrix"""
    n = U_matrix.shape[0]
    adj_matrix = np.zeros((n, n), dtype=np.int32)
    
    for i in range(n):
        for j in range(n):
            # Check if all dimensions of U_i > U_j
            all_greater = True
            for k in range(U_matrix.shape[1]):
                if U_matrix[i, k] <= U_matrix[j, k]:
                    all_greater = False
                    break
            adj_matrix[i, j] = 1 if all_greater else 0
    
    return adj_matrix

# ================================================================
# 3. VECTORIZATION WITH NUMPY
# ================================================================

# BEFORE: Python loops for matrix operations
def slow_matrix_comparison(U_matrix):
    n = U_matrix.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = np.all(U_matrix[i] > U_matrix[j])
    return result

# AFTER: Vectorized NumPy
def fast_matrix_comparison(U_matrix):
    # Broadcasting magic - much faster
    return (U_matrix[:, None, :] > U_matrix[None, :, :]).all(axis=2).astype(int)

# ================================================================
# 4. OPTIMIZING YOUR SPECIFIC BOTTLENECKS
# ================================================================

# If BasicUtils.nle is slow, try to cache it
nle_cache = {}

def cached_nle(adj_matrix):
    key = adj_matrix.tobytes()
    if key not in nle_cache:
        nle_cache[key] = BasicUtils.nle(adj_matrix)  # Your original function
    return nle_cache[key]

# If transitive_reduction is slow, cache it too
tr_cache = {}

def cached_transitive_reduction(adj_matrix):
    key = adj_matrix.tobytes()
    if key not in tr_cache:
        tr_cache[key] = BasicUtils.transitive_reduction(adj_matrix)
    return tr_cache[key]

# ================================================================
# 5. MEMORY-EFFICIENT OPERATIONS
# ================================================================

def memory_efficient_likelihood(data_chunks, chunk_size=1000):
    """Process data in chunks to avoid memory issues"""
    total_likelihood = 0.0
    
    # Process in chunks instead of all at once
    for i in range(0, len(data_chunks), chunk_size):
        chunk = data_chunks[i:i+chunk_size]
        chunk_likelihood = compute_likelihood_chunk(chunk)
        total_likelihood += chunk_likelihood
    
    return total_likelihood

# ================================================================
# 6. PARALLEL COMPUTATION (if applicable)
# ================================================================

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_likelihood_computation(data_list, num_processes=None):
    """Compute likelihoods in parallel"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        futures = [executor.submit(compute_single_likelihood, data) 
                  for data in data_list]
        
        # Collect results
        results = [future.result() for future in futures]
    
    return sum(results)

# ================================================================
# 7. SPECIFIC OPTIMIZATIONS FOR YOUR CODE
# ================================================================

# Optimize the weight computation in weighted_queue_jump
@jit(nopython=True)
def fast_weight_computation(remaining_indices, U0, item_indices):
    """Fast computation of Plackett-Luce weights"""
    weights = np.zeros(len(remaining_indices))
    
    for i, local_id in enumerate(remaining_indices):
        g_idx = item_indices[local_id]
        bar_eta = np.mean(U0[g_idx, :])  # Numba can handle this
        weights[i] = np.exp(bar_eta)
    
    return weights

# Cache hierarchical partial orders
partial_order_cache = {}

def cached_hierarchical_partial_orders(M0, assessors, M_a_dict, U0, U_a_dict, alpha):
    """Cache hierarchical partial order computations"""
    # Create cache key from relevant parameters
    key = (
        tuple(M0),
        tuple(sorted(assessors)),
        hash(str(M_a_dict)),  # Simple hash - you might want something more robust
        U0.tobytes(),
        hash(str(U_a_dict)),
        alpha.tobytes() if alpha is not None else None
    )
    
    if key not in partial_order_cache:
        partial_order_cache[key] = StatisticalUtils.build_hierarchical_partial_orders(
            M0, assessors, M_a_dict, U0, U_a_dict, alpha
        )
    
    return partial_order_cache[key]

# ================================================================
# 8. HOW TO APPLY THESE OPTIMIZATIONS
# ================================================================

def apply_optimizations_to_your_code():
    """
    Step-by-step guide to apply optimizations:
    
    1. RUN PROFILING FIRST:
       python profile_my_mcmc.py
    
    2. IDENTIFY TOP 3 BOTTLENECKS from the profiler output
    
    3. FOR EACH BOTTLENECK:
       - If it's a matrix operation → Try vectorization
       - If it's a Python loop → Try Numba @jit
       - If it's repeated computation → Add caching
       - If it's I/O bound → Try parallel processing
    
    4. APPLY ONE OPTIMIZATION AT A TIME
    
    5. RE-RUN PROFILING to measure improvement
    
    6. REPEAT until satisfied
    """
    pass

# ================================================================
# 9. MONITORING CACHE PERFORMANCE
# ================================================================

class MonitoredCache:
    """Cache with performance monitoring"""
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.maxsize = maxsize
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, key, value):
        if len(self.cache) >= self.maxsize:
            # Simple eviction - remove first item
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value
    
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def stats(self):
        print(f"Cache stats: {self.hits} hits, {self.misses} misses")
        print(f"Hit rate: {self.hit_rate():.2%}")

# Usage example:
# monitored_cache = MonitoredCache(maxsize=500)
# ... use cache in your code ...
# monitored_cache.stats()  # Check if caching is helping

if __name__ == "__main__":
    print("This file contains optimization examples.")
    print("Run your profiling first, then apply these techniques to your bottlenecks.")
    print("\nSteps:")
    print("1. python profile_my_mcmc.py")
    print("2. Identify your top bottlenecks")
    print("3. Apply relevant optimizations from this file")
    print("4. Re-run profiling to measure improvement") 