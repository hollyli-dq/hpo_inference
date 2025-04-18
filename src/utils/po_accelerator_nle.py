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

        # For each assessor, we go through the tasks
        for a in O_a_i_dict.keys():
            tasks_choice_sets = O_a_i_dict[a]
            tasks_observed = observed_orders.get(a, [])
            tasks_h = h_U.get(a, {})
            Ma = M_a_dict.get(a, [])

            for i_task, choice_set in enumerate(tasks_choice_sets):
                sub_size = len(choice_set)
                h_sub = np.zeros((sub_size, sub_size), dtype=int)
                local_map = {item: idx for idx, item in enumerate(choice_set)}

                # Build the adjacency among only the chosen items
                for r, item_r in enumerate(choice_set):
                    local_r = Ma.index(item_r)
                    for c, item_c in enumerate(choice_set):
                        local_c = Ma.index(item_c)
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
                        log_likelihood += math.log(prob_no_jump + prob_jump)

                elif noise_option == "mallows_noise":
                    mallows_prob = Mallows.compute_mallows_likelihood(
                        y=y_i,
                        h=h_sub,
                        theta=mallow_theta
                    )
                    log_likelihood += math.log(mallows_prob if mallows_prob > 0 else 1e-20)

        return log_likelihood
