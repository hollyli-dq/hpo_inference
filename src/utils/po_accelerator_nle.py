import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from scipy.stats import norm
import networkx as nx
from tqdm import tqdm
import time
import random
import math
import json
import yaml
import logging
from datetime import datetime

# Import from the package
from src.utils.po_fun import BasicUtils, StatisticalUtils
from src.models.mallow_function import Mallows

import threading
from concurrent.futures import ThreadPoolExecutor
from numba import jit

##########################################################
#                   LogLikelihoodCache
##########################################################
class LogLikelihoodCache:
    """
    Caches and parallelizes the computation of number of linear extensions (nle) and
    number of extensions with a specific first item (nle_first).
    """

    # Class-level dictionaries for caching
    nle_cache = {}
    nle_first_cache = {}

    # Thread pool for parallel computations
    _pool = ThreadPoolExecutor(max_workers=16)

    # A lock to ensure thread-safe reads/writes to the caches
    _cache_lock = threading.Lock()

    @staticmethod
    def _matrix_key(adj_matrix: np.ndarray) -> bytes:
        """Convert adjacency matrix to a bytes object as a cache key."""
        return adj_matrix.tobytes()

    @classmethod
    def _get_nle(cls, adj_matrix: np.ndarray) -> int:
        """
        Retrieve or compute the number of linear extensions with caching,
        offloading the BasicUtils.nle(...) computation to the thread pool.
        """
        key = cls._matrix_key(adj_matrix)

        # Check the cache under the lock
        with cls._cache_lock:
            if key in cls.nle_cache:
                return cls.nle_cache[key]

        # If not cached, compute in a worker thread
        future = cls._pool.submit(BasicUtils.nle, adj_matrix)
        val = future.result()  # Wait for worker to finish

        # Store result back to cache
        with cls._cache_lock:
            cls.nle_cache[key] = val
        return val

    @classmethod
    def _get_nle_first(cls, adj_matrix: np.ndarray, local_idx: int) -> int:
        """
        Retrieve or compute the number of linear extensions with a specific first item,
        using caching and parallel computation.
        """
        matrix_key = cls._matrix_key(adj_matrix)
        cache_key = (matrix_key, local_idx)

        # Check the cache
        with cls._cache_lock:
            if cache_key in cls.nle_first_cache:
                return cls.nle_first_cache[cache_key]

        # If not found, compute in parallel
        future = cls._pool.submit(BasicUtils.num_extensions_with_first, adj_matrix, local_idx)
        val = future.result()

        # Update the cache
        with cls._cache_lock:
            cls.nle_first_cache[cache_key] = val

        return val
    @classmethod
    def _compute_bar_eta(cls, j, U0):
        """
        Suppose 'U0' is shape (n_global, K).
        For item 'j', we have a latent vector: U0[j,:].
        We define bar(eta_j) = average(U0[j,:]).
        """
        # If item j is an integer index, then:
        latent_vec = U0[j,:]  # shape (K,)
        bar_eta_j = np.mean(latent_vec)
        return bar_eta_j
    
    @classmethod
    def calculate_log_likelihood(
        cls,
        Z,
        h_Z,
        observed_orders_idx,
        choice_sets,
        item_to_index,
        prob_noise,
        mallow_theta,
        noise_option
    ):
        if noise_option not in ["queue_jump", "mallows_noise"]:
            raise ValueError(f"Invalid noise_option: {noise_option}. Valid options are ['queue_jump', 'mallows_noise'].")

        log_likelihood = 0.0

        for idx, y_i in enumerate(observed_orders_idx):
            O_i = choice_sets[idx]
            O_i_indices = sorted([item_to_index[item] for item in O_i])
            m = len(y_i)

            if noise_option == "queue_jump":
                for j, y_j in enumerate(y_i):
                    remaining_indices = y_i[j:]
                    h_Z_remaining = h_Z[np.ix_(remaining_indices, remaining_indices)]
                    tr_remaining = BasicUtils.transitive_reduction(h_Z_remaining)

                    num_le = cls._get_nle(tr_remaining)  # parallel call
                    local_idx = remaining_indices.index(y_j)
                    num_first_item = cls._get_nle_first(tr_remaining, local_idx)

                    prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)
                    prob_jump = prob_noise * (1 / (m - j))
                    prob_observed = prob_no_jump + prob_jump
                    log_likelihood += math.log(prob_observed)

            elif noise_option == "mallows_noise":
                h_Z_Oi = h_Z[np.ix_(O_i_indices, O_i_indices)]
                mallows_prob = Mallows.compute_mallows_likelihood(
                    y=y_i,
                    h=h_Z_Oi,
                    theta=mallow_theta,
                    O_i_indice=O_i_indices
                )
                log_likelihood += math.log(mallows_prob if mallows_prob > 0 else 1e-20)

        return log_likelihood


##########################################################
#                   HPO_LogLikelihoodCache
##########################################################
class HPO_LogLikelihoodCache:
    """
    Similar to LogLikelihoodCache but used for hierarchical partial orders.
    Uses the same approach of parallelizing nle computations and caching results.
    """

    # Class-level dictionaries for caching
    nle_cache = {}
    nle_first_cache = {}

    # Thread pool and lock for concurrency
    _pool = ThreadPoolExecutor(max_workers=4)
    _cache_lock = threading.Lock()

    @staticmethod
    def _matrix_key(adj_matrix: np.ndarray) -> bytes:
        """Convert adjacency matrix to a bytes object as a cache key."""
        return adj_matrix.tobytes()

    @classmethod
    def _get_nle(cls, adj_matrix: np.ndarray) -> int:
        """Retrieve or compute the number of linear extensions with caching & parallel."""
        key = cls._matrix_key(adj_matrix)
        with cls._cache_lock:
            if key in cls.nle_cache:
                return cls.nle_cache[key]

        future = cls._pool.submit(BasicUtils.nle, adj_matrix)
        val = future.result()

        with cls._cache_lock:
            cls.nle_cache[key] = val
        return val

    @classmethod
    def _get_nle_first(cls, adj_matrix: np.ndarray, local_idx: int) -> int:
        """Retrieve or compute the number of extensions with a specific first item, in parallel."""
        matrix_key = cls._matrix_key(adj_matrix)
        cache_key = (matrix_key, local_idx)

        with cls._cache_lock:
            if cache_key in cls.nle_first_cache:
                return cls.nle_first_cache[cache_key]

        future = cls._pool.submit(BasicUtils.num_extensions_with_first, adj_matrix, local_idx)
        val = future.result()

        with cls._cache_lock:
            cls.nle_first_cache[cache_key] = val
        return val
    @classmethod
    def _compute_bar_eta(cls, j, U0):
        """
        Suppose 'U0' is shape (n_global, K).
        For item 'j', we have a latent vector: U0[j,:].
        We define bar(eta_j) = average(U0[j,:]).
        """
        # If item j is an integer index, then:
        latent_vec = U0[j,:]  # shape (K,)
        bar_eta_j = np.mean(latent_vec)
        return bar_eta_j
    @classmethod
    def calculate_log_likelihood_hpo(
        cls,
        U,                # global + local latents
        h_U,              # { assessor : adjacency_matrix_local_items }
        observed_orders,  # { assessor : [observed_orders], ... }
        M_a_dict,
        O_a_i_dict,       # { assessor : [choice_set for each task], ... }
        item_to_index,    # item→int map
        prob_noise,
        mallow_theta,
        noise_option,
        alpha
    ):
        if noise_option not in ["queue_jump", "mallows_noise"]:
            raise ValueError(f"Invalid noise_option: {noise_option}.")
        if observed_orders is None:
            return 0.0

        log_likelihood = 0.0
        U0 = U.get("U0", np.array([]))
        U_a_dict = U.get("U_a_dict", {})

        # For each assessor, we go through the tasks
        for a in O_a_i_dict.keys():
            tasks_choice_sets = O_a_i_dict[a]
            tasks_observed = observed_orders.get(a, [])
            Ma = M_a_dict.get(a, [])
            
            # Get or compute the hierarchical partial orders for this assessor
            if a in h_U:
                tasks_h = h_U[a]
            else:
                # Build partial orders if not provided
                Ua = U_a_dict.get(a, np.array([]))
                
                # Create adjacency matrix for the items in this cluster
                tasks_h = np.zeros((len(Ma), len(Ma)), dtype=int)
                for r, item_r in enumerate(Ma):
                    local_r = Ma.index(item_r)
                    i_global = item_to_index.get(item_r, -1)
                    
                    for c, item_c in enumerate(Ma):
                        local_c = Ma.index(item_c)
                        j_global = item_to_index.get(item_c, -1)
                        
                        if i_global >= 0 and j_global >= 0:
                            # If using U_a directly
                            if len(Ua) > 0 and local_r < Ua.shape[0] and local_c < Ua.shape[0]:
                                tasks_h[r, c] = 1 if np.all(Ua[local_r] > Ua[local_c]) else 0
                            # Else use global U0 with alpha term
                            else:
                                alpha_i = alpha[i_global] if alpha is not None else 0
                                alpha_j = alpha[j_global] if alpha is not None else 0
                                tasks_h[r, c] = 1 if np.all(U0[i_global] + alpha_i > U0[j_global] + alpha_j) else 0

            for i_task, choice_set in enumerate(tasks_choice_sets):
                if i_task >= len(tasks_observed):
                    continue
                    
                sub_size = len(choice_set)
                h_sub = np.zeros((sub_size, sub_size), dtype=int)
                local_map = {item: idx for idx, item in enumerate(choice_set)}

                # Build the adjacency among only the chosen items
                for r, item_r in enumerate(choice_set):
                    if item_r in Ma:
                        local_r = Ma.index(item_r)
                        for c, item_c in enumerate(choice_set):
                            if item_c in Ma:
                                local_c = Ma.index(item_c)
                                if local_r < len(tasks_h) and local_c < len(tasks_h):
                                    h_sub[r, c] = tasks_h[local_r, local_c]

                y_i = tasks_observed[i_task]
                y_i_local = [local_map[item] for item in y_i if item in local_map]
                m = len(y_i_local)

                if noise_option == "queue_jump":
                    for j, y_j in enumerate(y_i_local):
                        remaining_indices = y_i_local[j:]
                        h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)]
                        tr_remaining = BasicUtils.transitive_reduction(h_remaining)

                        num_le = cls._get_nle(tr_remaining)  # parallel call
                        local_idx = remaining_indices.index(y_j)
                        num_first_item = cls._get_nle_first(tr_remaining, local_idx)

                        prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)
                        prob_jump = prob_noise * (1.0 / (m - j))
                        log_likelihood += math.log(max(prob_no_jump + prob_jump, 1e-20))

                elif noise_option=="weighted_queue_jump":
                    # New approach
                    for j, y_j in enumerate(y_i_local):
                        remaining_indices = y_i_local[j:]
                        h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)]
                        tr_remaining = BasicUtils.transitive_reduction(h_remaining)

                        num_le = cls._get_nle(tr_remaining)
                        local_idx = remaining_indices.index(y_j)
                        num_first_item = cls._get_nle_first(tr_remaining, local_idx)
                        prob_no_jump = (1 - prob_noise)*(num_first_item/num_le)

                        # Weighted jump with Plackett-Luce weights = exp(bar_eta_j).
                        # We find the global item index from 'choice_set'
                        # Then bar_eta_j = mean(U0[ that_global_idx, : ])

                        # sum_of_weights
                        sum_w = 0.0
                        weight_map = {}
                        for local_id in remaining_indices:
                            # item in the original 'choice_set'
                            item_name = choice_set[local_id]
                            # global idx
                            g_idx = item_to_index[item_name]
                            # bar_eta = average of U0[g_idx,:]
                            bar_eta = cls._compute_bar_eta(g_idx, U0)
                            w_val = math.exp(bar_eta)
                            weight_map[local_id] = w_val
                            sum_w += w_val

                        w_j = weight_map[y_j]
                        prob_jump_for_yj = prob_noise*(w_j/sum_w)

                        total_p = prob_no_jump + prob_jump_for_yj
                        log_likelihood += math.log(max(total_p, 1e-20))


                elif noise_option == "mallows_noise":
                    mallows_prob = Mallows.compute_mallows_likelihood(
                        y=y_i,
                        h=h_sub,
                        theta=mallow_theta
                    )
                    log_likelihood += math.log(max(mallows_prob, 1e-20))

        return log_likelihood
        
    @classmethod
    def calculate_single_list_likelihood(
        cls,
        list_idx,         # ID of the list/order to evaluate
        e_list,           # The observed order for this list/task
        cluster_id,       # ID of the cluster to consider for this list
        U0,               # Global latent positions
        U_a_dict,         # Assessor-specific latent positions
        observed_orders,  # All observed orders
        M_a_dict,         # Mapping from assessor to item indices
        O_a_i_dict,       # Choice sets for each assessor
        item_to_index,    # Map from item to index
        prob_noise,       # Noise probability
        mallow_theta,     # Mallows dispersion parameter
        noise_option,     # Noise model type
        alpha,            # Parameter for the hierarchical model
        h_U         # Pre-computed hierarchical partial orders      
    ):

        if noise_option not in ["queue_jump", "weighted_queue_jump", "mallows_noise"]:
            raise ValueError(f"Invalid noise_option: {noise_option}.")
            
        # Get the observed order for this list
        y_i = e_list
        # If no order observed, return 0
        if not y_i:
            return 0.0
            
        # Get the choice set for this list (all elements in the observed order)
        choice_set = sorted(e_list)

        # If not provided, we need to compute it based on U0 and U_a
        Ma = M_a_dict.get(cluster_id, [])
        U_a = U_a_dict.get(cluster_id, np.array([]))
        h_a = h_U.get(cluster_id, np.array([])) 
    
        # Subset to only the items in the choice set
        sub_size = len(choice_set)
        h_sub = np.zeros((sub_size, sub_size), dtype=int)
        local_map = {item: idx for idx, item in enumerate(choice_set)}
        
        # Build the adjacency matrix for only the chosen items
        for r, item_r in enumerate(choice_set):
            if item_r in Ma:
                local_r = Ma.index(item_r)
                for c, item_c in enumerate(choice_set):
                    if item_c in Ma:
                        local_c = Ma.index(item_c)
                        if local_r < len(h_a) and local_c < len(h_a):
                            h_sub[r, c] = h_a[local_r, local_c]
        
        # Convert observed order to local indices
        y_i_local = [local_map[item] for item in y_i if item in local_map]
        m = len(y_i_local)
        
        # Calculate log-likelihood based on noise model
        log_likelihood = 0.0
        
        if noise_option == "queue_jump":
            for j, y_j in enumerate(y_i_local):
                remaining_indices = y_i_local[j:]
                h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)]
                tr_remaining = BasicUtils.transitive_reduction(h_remaining)
                
                num_le = cls._get_nle(tr_remaining)
                local_idx = remaining_indices.index(y_j)
                num_first_item = cls._get_nle_first(tr_remaining, local_idx)
                
                prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)
                prob_jump = prob_noise * (1.0 / (m - j))
                log_likelihood += math.log(max(prob_no_jump + prob_jump, 1e-20))
        elif noise_option=="weighted_queue_jump":
            # New approach
            for j, y_j in enumerate(y_i_local):
                remaining_indices = y_i_local[j:]
                h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)]
                tr_remaining = BasicUtils.transitive_reduction(h_remaining)

                num_le = cls._get_nle(tr_remaining)
                local_idx = remaining_indices.index(y_j)
                num_first_item = cls._get_nle_first(tr_remaining, local_idx)
                prob_no_jump = (1 - prob_noise)*(num_first_item/num_le)

                # Weighted jump with Plackett-Luce weights = exp(bar_eta_j).
                # We find the global item index from 'choice_set'
                # Then bar_eta_j = mean(U0[ that_global_idx, : ])

                # sum_of_weights
                sum_w = 0.0
                weight_map = {}
                for local_id in remaining_indices:
                    # item in the original 'choice_set'
                    item_name = choice_set[local_id]
                    # global idx
                    g_idx = item_to_index[item_name]
                    # bar_eta = average of U0[g_idx,:]
                    bar_eta = cls._compute_bar_eta(g_idx, U0)
                    w_val = math.exp(bar_eta)
                    weight_map[local_id] = w_val
                    sum_w += w_val

                w_j = weight_map[y_j]
                prob_jump_for_yj = prob_noise*(w_j/sum_w)

                total_p = prob_no_jump + prob_jump_for_yj
                log_likelihood += math.log(max(total_p, 1e-20))
        elif noise_option == "mallows_noise":
            mallows_prob = Mallows.compute_mallows_likelihood(
                y=y_i,
                h=h_sub,
                theta=mallow_theta
            )
            log_likelihood += math.log(max(mallows_prob, 1e-20))
            
        return log_likelihood
        
