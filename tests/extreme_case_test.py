import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import yaml
import numpy as np
import seaborn as sns
import math
from scipy.integrate import quad
import random
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, expon, norm, uniform, kstest
from typing import List, Dict, Any
import copy  # Added for deep copying

import matplotlib.pyplot as plt

#!/usr/bin/env python3

"""
test_extreme_tau.py

Demonstrates handling two extreme scenarios:
  1) tau = 0 => local embeddings are independent N(0, Sigma_rho)
  2) tau = 1 => local embeddings match the global embedding exactly

Then it performs a fixed-tau MCMC to confirm partial orders behave as expected.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import yaml
import numpy as np
import seaborn as sns
import math
from scipy.integrate import quad
import random
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, expon, norm, uniform, kstest
from typing import List, Dict, Any
import copy  # Added for deep copying

import matplotlib.pyplot as plt


# Import from the correct paths
from src.mcmc.hpo_po_hm_mcmc import mcmc_simulation_hpo
from src.mcmc.hpo_po_hm_mcmc_k import mcmc_simulation_hpo_k
from src.utils.po_fun import BasicUtils, StatisticalUtils, GenerationUtils
from src.utils.po_accelerator_nle import HPO_LogLikelihoodCache

# Path to configuration file and output folder
current_dir  = Path.cwd()           # /…/hpo_inference/hpo_inference/notebooks
project_root = current_dir.parents[0]   # /…/hpo_inference
config_path = project_root / "config" / "hpo_mcmc_configuration.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
OUTPUT_DIR= project_root / "tests" / "hpo_test_output"


def mcmc_simulation_hpo_fixed_tau_queue_jump(
    num_iterations: int,
    # Hierarchy definition
    M0: List[int],
    assessors: List[int],
    M_a_dict: Dict[int, List[int]],
    # Observed data
    O_a_i_dict: Dict[int, List[List[int]]],
    observed_orders: Dict[int, List[List[int]]],
    alpha: np.ndarray,
    K: int,
    dr: float,
    mcmc_pt: List[float],    # [rho, noise, U0, Ua]
    rho_prior: float,
    noise_beta_prior: float, # Beta(1, noise_beta_prior) => prob_noise
    tau_value: float,        # fixed
    noise_option: str = "queue_jump",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    MCMC that does NOT sample tau, but does sample:
      - rho
      - prob_noise (queue-jump)
      - U0
      - Ua
    tau is fixed = tau_value.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    items = sorted(set(M0))
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # 1) Sample initial rho
    rho = StatisticalUtils.rRprior(rho_prior)
    # 2) Sample initial prob_noise
    prob_noise = StatisticalUtils.rPprior(noise_beta_prior)

    # 3) Build covariance
    Sigma_rho = BasicUtils.build_Sigma_rho(K, rho)

    # 4) Sample global U0
    rng = np.random.default_rng(random_seed)
    n_global = len(M0)
    U0 = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma_rho, size=n_global)

    # 5) Initialize U_a
    U_a_dict = {}
    for a in assessors:
        M_a = M_a_dict[a]
        n_a = len(M_a)
        Ua = np.zeros((n_a, K), dtype=float)
        for i_loc, j_global in enumerate(M_a):
            mean_vec = tau_value * U0[j_global, :]
            cov_mat = (1.0 - tau_value**2) * Sigma_rho
            Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
        U_a_dict[a] = Ua

    llk_cache = HPO_LogLikelihoodCache()
    U0_trace = []
    Ua_trace = []
    rho_trace = []
    prob_noise_trace = []
    acceptance_rates = []
    log_llk_current = None
    num_accepts = 0

    rho_pct, noise_pct, U0_pct, Ua_pct = mcmc_pt

    for it in range(num_iterations):
        # Build partial orders
        h_U = StatisticalUtils.build_hierarchical_partial_orders(
            M0=M0, assessors=assessors, M_a_dict=M_a_dict,
            U0=U0, U_a_dict=U_a_dict, alpha=alpha
        )

        if log_llk_current is None:
            log_llk_current = llk_cache.calculate_log_likelihood_hpo(
                U={"U0": U0, "U_a_dict": U_a_dict},
                h_U=h_U,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=0.0,    # ignoring Mallows
                noise_option=noise_option,
                alpha=alpha
            )

        r = random.random()
        # 1) Update rho
        if r < rho_pct:
            delta = random.uniform(dr, 1.0/dr)
            rho_prime = 1.0 - (1.0 - rho)*delta
            if not (0< rho_prime <1):
                rho_prime = rho
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K, rho_prime)

            lp_curr = StatisticalUtils.dRprior(rho, rho_prior)
            lp_prop = StatisticalUtils.dRprior(rho_prime, rho_prior)
            llk_prop = log_llk_current  # not re-sampled
            log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr) - math.log(delta)

            if math.log(random.random()) < log_acc_ratio:
                rho = rho_prime
                Sigma_rho = Sigma_rho_prime
                log_llk_current = llk_prop
                num_accepts += 1

        # 2) Update prob_noise
        elif r < rho_pct + noise_pct:
            prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)
            lp_curr = StatisticalUtils.dPprior(prob_noise, noise_beta_prior)
            lp_prop = StatisticalUtils.dPprior(prob_noise_prime, noise_beta_prior)

            llk_prop = llk_cache.calculate_log_likelihood_hpo(
                U={"U0": U0,"U_a_dict": U_a_dict}, 
                h_U=h_U,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise_prime,
                mallow_theta=0.0,
                noise_option=noise_option,
                alpha=alpha
            )
            log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr)
            if math.log(random.random()) < log_acc_ratio:
                prob_noise = prob_noise_prime
                log_llk_current = llk_prop
                num_accepts += 1

        # 3) Update U0
        elif r < rho_pct + noise_pct + U0_pct:
            j_global = random.randint(0, n_global-1)
            old_val = U0[j_global,:].copy()
            proposed_val = np.random.multivariate_normal(mean=old_val, cov=Sigma_rho)
            U0_prime = U0.copy()
            U0_prime[j_global,:] = proposed_val

            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0, assessors=assessors, M_a_dict=M_a_dict,
                U0=U0_prime, U_a_dict=U_a_dict, alpha=alpha
            )
            llk_prop = llk_cache.calculate_log_likelihood_hpo(
                U={"U0": U0_prime,"U_a_dict": U_a_dict},
                h_U=h_U_prime,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=0.0,
                noise_option=noise_option,
                alpha=alpha
            )

            lp_curr = (StatisticalUtils.log_U_prior(U0, rho, K)
                      +StatisticalUtils.log_U_a_prior(U_a_dict, tau_value, rho, K, M_a_dict, U0))
            lp_prop = (StatisticalUtils.log_U_prior(U0_prime, rho, K)
                      +StatisticalUtils.log_U_a_prior(U_a_dict, tau_value, rho, K, M_a_dict, U0_prime))

            log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr)
            if math.log(random.random()) < log_acc_ratio:
                U0 = U0_prime
                log_llk_current = llk_prop
                num_accepts += 1

        # 4) Update Ua
        else:
            a_key = random.choice(assessors)
            M_a = M_a_dict[a_key]
            if M_a:
                row_loc = random.randint(0, len(M_a)-1)
                old_val = U_a_dict[a_key][row_loc,:].copy()
                proposed_val = np.random.multivariate_normal(mean=old_val, cov=Sigma_rho)
                U_a_dict_prime = {}
                for a_ in assessors:
                    U_a_dict_prime[a_] = U_a_dict[a_].copy()
                U_a_dict_prime[a_key][row_loc,:] = proposed_val

                h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                    M0=M0, assessors=assessors, M_a_dict=M_a_dict,
                    U0=U0, U_a_dict=U_a_dict_prime, alpha=alpha
                )
                llk_prop = llk_cache.calculate_log_likelihood_hpo(
                    U={"U0":U0, "U_a_dict":U_a_dict_prime},
                    h_U=h_U_prime,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=0.0,
                    noise_option=noise_option,
                    alpha=alpha
                )
                lp_curr = (StatisticalUtils.log_U_prior(U0, rho, K)
                          +StatisticalUtils.log_U_a_prior(U_a_dict, tau_value, rho, K, M_a_dict, U0))
                lp_prop = (StatisticalUtils.log_U_prior(U0, rho, K)
                          +StatisticalUtils.log_U_a_prior(U_a_dict_prime, tau_value, rho, K, M_a_dict, U0))
                log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr)
                if math.log(random.random()) < log_acc_ratio:
                    U_a_dict = U_a_dict_prime
                    log_llk_current = llk_prop
                    num_accepts += 1

        rho_trace.append(rho)
        prob_noise_trace.append(prob_noise)
        U0_trace.append(U0.copy())
        Ua_trace.append(copy.deepcopy(U_a_dict))
        acceptance_rates.append(num_accepts/(it+1))

    return {
        "rho_trace": rho_trace,
        "prob_noise_trace": prob_noise_trace,
        "U0_trace": U0_trace,
        "Ua_trace": Ua_trace,
        "acceptance_rates": acceptance_rates,
        "rho_final": rho,
        "prob_noise_final": prob_noise,
        "U0_final": U0,
        "U_a_final": U_a_dict
    }

# Import from the correct paths
from src.mcmc.hpo_po_hm_mcmc import mcmc_simulation_hpo
from src.mcmc.hpo_po_hm_mcmc_k import mcmc_simulation_hpo_k
from src.utils.po_fun import BasicUtils, StatisticalUtils, GenerationUtils
from src.utils.po_accelerator_nle import HPO_LogLikelihoodCache

# Path to configuration file and output folder
current_dir  = Path.cwd()           # /…/hpo_inference/hpo_inference/notebooks
project_root = current_dir.parents[0]   # /…/hpo_inference
config_path = project_root / "config" / "hpo_mcmc_configuration.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
OUTPUT_DIR= project_root / "tests" / "hpo_test_output"


def mcmc_simulation_hpo_fixed_tau_queue_jump(
    num_iterations: int,
    # Hierarchy definition
    M0: List[int],
    assessors: List[int],
    M_a_dict: Dict[int, List[int]],
    # Observed data
    O_a_i_dict: Dict[int, List[List[int]]],
    observed_orders: Dict[int, List[List[int]]],
    alpha: np.ndarray,
    K: int,
    dr: float,
    mcmc_pt: List[float],    # [rho, noise, U0, Ua]
    rho_prior: float,
    noise_beta_prior: float, # Beta(1, noise_beta_prior) => prob_noise
    tau_value: float,        # fixed
    noise_option: str = "queue_jump",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    MCMC that does NOT sample tau, but does sample:
      - rho
      - prob_noise (queue-jump)
      - U0
      - Ua
    tau is fixed = tau_value.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    items = sorted(set(M0))
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # 1) Sample initial rho
    rho = StatisticalUtils.rRprior(rho_prior)
    # 2) Sample initial prob_noise
    prob_noise = StatisticalUtils.rPprior(noise_beta_prior)

    # 3) Build covariance
    Sigma_rho = BasicUtils.build_Sigma_rho(K, rho)

    # 4) Sample global U0
    rng = np.random.default_rng(random_seed)
    n_global = len(M0)
    U0 = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma_rho, size=n_global)

    # 5) Initialize U_a
    U_a_dict = {}
    for a in assessors:
        M_a = M_a_dict[a]
        n_a = len(M_a)
        Ua = np.zeros((n_a, K), dtype=float)
        for i_loc, j_global in enumerate(M_a):
            mean_vec = tau_value * U0[j_global, :]
            cov_mat = (1.0 - tau_value**2) * Sigma_rho
            Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
        U_a_dict[a] = Ua

    llk_cache = HPO_LogLikelihoodCache()
    U0_trace = []
    Ua_trace = []
    rho_trace = []
    prob_noise_trace = []
    acceptance_rates = []
    log_llk_current = None
    num_accepts = 0

    rho_pct, noise_pct, U0_pct, Ua_pct = mcmc_pt

    for it in range(num_iterations):
        # Build partial orders
        h_U = StatisticalUtils.build_hierarchical_partial_orders(
            M0=M0, assessors=assessors, M_a_dict=M_a_dict,
            U0=U0, U_a_dict=U_a_dict, alpha=alpha
        )

        if log_llk_current is None:
            log_llk_current = llk_cache.calculate_log_likelihood_hpo(
                U={"U0": U0, "U_a_dict": U_a_dict},
                h_U=h_U,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=0.0,    # ignoring Mallows
                noise_option=noise_option,
                alpha=alpha
            )

        r = random.random()
        # 1) Update rho
        if r < rho_pct:
            delta = random.uniform(dr, 1.0/dr)
            rho_prime = 1.0 - (1.0 - rho)*delta
            if not (0< rho_prime <1):
                rho_prime = rho
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K, rho_prime)

            lp_curr = StatisticalUtils.dRprior(rho, rho_prior)
            lp_prop = StatisticalUtils.dRprior(rho_prime, rho_prior)
            llk_prop = log_llk_current  # not re-sampled
            log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr) - math.log(delta)

            if math.log(random.random()) < log_acc_ratio:
                rho = rho_prime
                Sigma_rho = Sigma_rho_prime
                log_llk_current = llk_prop
                num_accepts += 1

        # 2) Update prob_noise
        elif r < rho_pct + noise_pct:
            prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)
            lp_curr = StatisticalUtils.dPprior(prob_noise, noise_beta_prior)
            lp_prop = StatisticalUtils.dPprior(prob_noise_prime, noise_beta_prior)

            llk_prop = llk_cache.calculate_log_likelihood_hpo(
                U={"U0": U0,"U_a_dict": U_a_dict}, 
                h_U=h_U,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise_prime,
                mallow_theta=0.0,
                noise_option=noise_option,
                alpha=alpha
            )
            log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr)
            if math.log(random.random()) < log_acc_ratio:
                prob_noise = prob_noise_prime
                log_llk_current = llk_prop
                num_accepts += 1

        # 3) Update U0
        elif r < rho_pct + noise_pct + U0_pct:
            j_global = random.randint(0, n_global-1)
            old_val = U0[j_global,:].copy()
            proposed_val = np.random.multivariate_normal(mean=old_val, cov=Sigma_rho)
            U0_prime = U0.copy()
            U0_prime[j_global,:] = proposed_val

            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0, assessors=assessors, M_a_dict=M_a_dict,
                U0=U0_prime, U_a_dict=U_a_dict, alpha=alpha
            )
            llk_prop = llk_cache.calculate_log_likelihood_hpo(
                U={"U0": U0_prime,"U_a_dict": U_a_dict},
                h_U=h_U_prime,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=0.0,
                noise_option=noise_option,
                alpha=alpha
            )

            lp_curr = (StatisticalUtils.log_U_prior(U0, rho, K)
                      +StatisticalUtils.log_U_a_prior(U_a_dict, tau_value, rho, K, M_a_dict, U0))
            lp_prop = (StatisticalUtils.log_U_prior(U0_prime, rho, K)
                      +StatisticalUtils.log_U_a_prior(U_a_dict, tau_value, rho, K, M_a_dict, U0_prime))

            log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr)
            if math.log(random.random()) < log_acc_ratio:
                U0 = U0_prime
                log_llk_current = llk_prop
                num_accepts += 1

        # 4) Update Ua
        else:
            a_key = random.choice(assessors)
            M_a = M_a_dict[a_key]
            if M_a:
                row_loc = random.randint(0, len(M_a)-1)
                old_val = U_a_dict[a_key][row_loc,:].copy()
                proposed_val = np.random.multivariate_normal(mean=old_val, cov=Sigma_rho)
                U_a_dict_prime = {}
                for a_ in assessors:
                    U_a_dict_prime[a_] = U_a_dict[a_].copy()
                U_a_dict_prime[a_key][row_loc,:] = proposed_val

                h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                    M0=M0, assessors=assessors, M_a_dict=M_a_dict,
                    U0=U0, U_a_dict=U_a_dict_prime, alpha=alpha
                )
                llk_prop = llk_cache.calculate_log_likelihood_hpo(
                    U={"U0":U0, "U_a_dict":U_a_dict_prime},
                    h_U=h_U_prime,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=0.0,
                    noise_option=noise_option,
                    alpha=alpha
                )
                lp_curr = (StatisticalUtils.log_U_prior(U0, rho, K)
                          +StatisticalUtils.log_U_a_prior(U_a_dict, tau_value, rho, K, M_a_dict, U0))
                lp_prop = (StatisticalUtils.log_U_prior(U0, rho, K)
                          +StatisticalUtils.log_U_a_prior(U_a_dict_prime, tau_value, rho, K, M_a_dict, U0))
                log_acc_ratio = (llk_prop + lp_prop) - (log_llk_current + lp_curr)
                if math.log(random.random()) < log_acc_ratio:
                    U_a_dict = U_a_dict_prime
                    log_llk_current = llk_prop
                    num_accepts += 1

        rho_trace.append(rho)
        prob_noise_trace.append(prob_noise)
        U0_trace.append(U0.copy())
        Ua_trace.append(copy.deepcopy(U_a_dict))
        acceptance_rates.append(num_accepts/(it+1))

    return {
        "rho_trace": rho_trace,
        "prob_noise_trace": prob_noise_trace,
        "U0_trace": U0_trace,
        "Ua_trace": Ua_trace,
        "acceptance_rates": acceptance_rates,
        "rho_final": rho,
        "prob_noise_final": prob_noise,
        "U0_final": U0,
        "U_a_final": U_a_dict
    }


if __name__ == "__main__":
    # Toy example with N=5 items, K=2 dimension
    N = 5
    K = 2
    assessors = [101, 202]
    alpha = np.array([0.5]*(N+1))  # or adapt
    M_a_dict = {
        101: [0,1,2],
        202: [2,3,4]
    }
    M0 = list(range(N))

    # Minimal tasks, orders for demonstration
    O_a_i_dict = {101:[[0,1],[1,2]], 202:[[2,3],[3,4]]}
    observed_orders = O_a_i_dict

    # Hyperparams for MCMC
    rho_prior = 0.1667
    noise_beta_prior = 1.0
    mcmc_pt = [0.2, 0.2, 0.3, 0.3]  # [rho, prob_noise, U0, Ua]
    num_iterations = 500000
    dr = 1.1

    # ================== TAU=0 scenario ==================
    tau0 = 0.0
    results_tau0 = mcmc_simulation_hpo_fixed_tau_queue_jump(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,
        dr=dr,
        mcmc_pt=mcmc_pt,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        tau_value=tau0,
        noise_option="queue_jump",
        random_seed=111
    )


    # ... inside your if __name__ == "__main__": block, after you get results_tau0 ...


    print("\n************* MCMC RESULTS for TAU=0 (free-floating local) *************\n")
    print(f"Final rho: {results_tau0['rho_final']:.4f}")
    print(f"Final prob_noise: {results_tau0['prob_noise_final']:.4f}")

    U0_final_0 = results_tau0["U0_final"]
    Ua_final_0 = results_tau0["U_a_final"]
    Ua_trace_tau0 = results_tau0["Ua_trace"]  # shape: (num_iterations,) each is a dict
    print("----- Global U0 matrix (final) -----")
    print(U0_final_0)


    for dim_k in range(K):
        for a in assessors:
            local_mat = Ua_final_0[a]
            print(f"----- Local Ua for assessor {a} (final) -----")
            print(local_mat)

            # gather all local coords for assessor a in dimension k

            # Gather all local coords for assessor 'assessor_to_check' in dimension 'dim_k'
            all_vals = []
            for it in range(num_iterations):
                # local_mat has shape (n_a, K)
                local_mat_iter = Ua_trace_tau0[it][a]
                # Now gather dimension k for each item
                val_dim_k = local_mat_iter[0, dim_k]
                all_vals.append(val_dim_k)

            print(f"Collected {len(all_vals)} samples across {num_iterations} iterations.")

            # If you want to skip early burn-in iterations, you could do:
            # for it in range(num_burnin, num_iterations):
            #   ...

            # Now we do a histogram vs. Normal(0, Sigma_rho[k,k]) if tau=0
            # We'll build Sigma_rho from the final rho, or from the posterior mean of rho, etc.
            final_rho = results_tau0["rho_final"]
            Sigma_rho = BasicUtils.build_Sigma_rho(K, final_rho)

            mean_0 = 0.0
            std_0 = math.sqrt(Sigma_rho[dim_k, dim_k])

            plt.figure(figsize=(6,4))
            plt.hist(all_vals, bins=30, density=True, alpha=0.6, label=f"Ua dimension={dim_k}")

            x_vals = np.linspace(min(all_vals), max(all_vals), 200)
            pdf_vals = norm.pdf(x_vals, mean_0, std_0)
            plt.plot(x_vals, pdf_vals, 'r-', label=f"N(0, var={Sigma_rho[dim_k, dim_k]:.2f})")

            plt.title(f"Tau=0: dimension={dim_k}, assessor={a}")
            plt.xlabel("Ua samples (all iterations)")
            plt.ylabel("Density")
            plt.legend()

            out_filename = f"tau0_assessor{a}_dim{dim_k}_trace_hist.pdf"
            plt.savefig(out_filename, dpi=150)
            print(f"[INFO] Saved histogram to {out_filename}")



    # ================== TAU=1 scenario ==================
    tau1 = 1.0
    results_tau1 = mcmc_simulation_hpo_fixed_tau_queue_jump(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,
        dr=dr,
        mcmc_pt=mcmc_pt,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        tau_value=tau1,
        noise_option="queue_jump",
        random_seed=222
    )

    print("\n************* MCMC RESULTS for TAU=1 (local == global) *************\n")
    print(f"Final rho: {results_tau1['rho_final']:.4f}")
    print(f"Final prob_noise: {results_tau1['prob_noise_final']:.4f}")

    U0_final_1 = results_tau1["U0_final"]
    Ua_final_1 = results_tau1["U_a_final"]

    print("----- Global U0 matrix (final) -----")
    print(U0_final_1)

    for a in assessors:
        local_mat = Ua_final_1[a]
        print(f"----- Local Ua for assessor {a} (final) -----")
        print(local_mat)

    print("=> Because tau=1, each local Ua[j] is (in theory) identical to U0[j]. No random variation.")
    print("All local partial orders match the global partial order exactly.\n")
