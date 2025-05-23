import os
import sys
import copy
import time
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from collections import Counter, defaultdict

from src.utils.po_fun import BasicUtils, StatisticalUtils
from src.utils.po_accelerator_nle import HPO_LogLikelihoodCache


def mcmc_simulation_hpo_cluster(
    num_iterations: int,
    # Hierarchy definition
    M0: List[int],
    # Observed data
    O_a_i_dict: List[List[int]], 
    observed_orders: List[List[int]],
    # Additional model parameters
    sigma_beta: float,  # scalar only 
    X,  # X: np.ndarray of covariates 
    dr: float,          # multiplicative step size for rho
    drrt: float,        # multiplicative step size for tau and rho 
    drbeta: float,      # the beta updated step size
    sigma_mallow: float,
    noise_option: str,
    # pcts for update frequencies:
    # we have 8 updates: [rho_pct, tau_pct, rho_tau_pct, noise_pct, U0_pct, Ua_pct, K_pct, beta_pct]
    mcmc_pt: List[float],
    # priors
    rho_prior, 
    noise_beta_prior: float,
    mallow_ua: float,
    K_prior: int, 
    Dri_alpha: float,
    Dri_theta: float,
      # e.g. the mean of truncated Poisson or some hyperparam
    # Optional
    random_seed: int = 42 
) -> Dict[str, Any]:
    """
    MCMC for a hierarchical partial-order model with reversible-jump update for K (the latent dimension).
    """
    rng = np.random.default_rng(random_seed)


    # 1) Prepare global item set and initialization
    items = sorted(set(M0))
    item_to_index = {item: idx for idx, item in enumerate(items)}
    n_global = len(M0)
    n_lists = len(observed_orders)

    # Sample initial parameters
    rho = StatisticalUtils.rRprior(rho_prior)
    prob_noise = StatisticalUtils.rPprior(noise_beta_prior)
    mallow_theta = StatisticalUtils.rTprior(mallow_ua)
    tau = StatisticalUtils.rTauprior()
    K = K_prior

    Sigma_rho = BasicUtils.build_Sigma_rho(K, rho)

    # the covariates
    p = X.shape[0]

    # Generate initial beta using the array version
    beta= rng.normal(loc=0.0, scale=sigma_beta, size=(p,))
    alpha = X.T @ beta


    # 3) Initialize cluster structures (We start the list cluster to be 1)
   
    c_vec = [1]*n_lists
    cluster_ids  = [1] 
    na_dict = {1: n_lists}   # number of lists in cluster 1


    ### This is usded to make]\ a dictionary to calcualte the likelihood 
    observed_orders_cl = {1: [ord_ for ord_ in observed_orders]}
    O_a_i_dict_cl = {1: [cs for cs in O_a_i_dict]}

    all_items = set()
    for ord_ in observed_orders:
        all_items.update(ord_)
    M_a_dict = {1: sorted(all_items)}

    # 2) Initialize U0 (shape = (n_global, K))
    U0 = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma_rho, size=n_global)
    
    # 3) Initialize U_a_dict 
    U_a_dict = {}
    for k in cluster_ids:
        n_a = na_dict.get(k, [])
        M_a = M_a_dict.get(k, [])
        Ua = np.zeros((len(M_a), K), dtype=float)
        for i_loc, j_global in enumerate(M_a):
            mean_vec = tau * U0[j_global, :]
            cov_mat  = (1.0 - tau**2) * Sigma_rho
            Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
        U_a_dict[k] = Ua
    
    h_U = StatisticalUtils.build_hierarchical_partial_orders(
        M0=M0,
        assessors=cluster_ids,
        M_a_dict=M_a_dict,
        U0=U0,
        U_a_dict=U_a_dict,
        alpha=alpha
    )
    # 4) Setup traces
    U0_trace = []
    Ua_trace = []
    H_trace = []  
    rho_trace = []
    tau_trace = []
    prob_noise_trace = []
    mallow_theta_trace = []
    K_trace = []
    beta_trace=[]
    acceptance_decisions = []
    acceptance_rates = []
    log_likelihood_currents = []
    log_likelihood_primes = []
    update_records = []
    c_vec_trace = []
    num_acceptances = 0

    # Store proposed values if needed
    proposed_rho_vals = []
    proposed_tau_vals = []
    proposed_prob_noise_vals = []
    proposed_mallow_theta_vals = []
    proposed_U0 = []


    iteration_list = []
    update_category_list = []
    prior_timing_list = []
    likelihood_timing_list = []
    update_timing_list = []

    mcmc_pt = np.array(mcmc_pt, dtype=float)
    mcmc_pt = mcmc_pt / mcmc_pt.sum()   # now sums to 1.0

    rho_pct, tau_pct, rho_tau_pct, noise_pct, U0_pct, Ua_pct, K_pct, beta_pct,cl_pct = mcmc_pt

    # Determine thresholds
    thresh_rho     = rho_pct
    thresh_tau     = thresh_rho + tau_pct
    thresh_rho_tau = thresh_tau + rho_tau_pct
    thresh_noise   = thresh_rho_tau + noise_pct
    thresh_U0      = thresh_noise + U0_pct
    thresh_Ua      = thresh_U0 + Ua_pct
    thresh_Kdim    = thresh_Ua + K_pct
    thresh_beta    = thresh_Kdim + beta_pct
    thresh_cl      = thresh_beta + cl_pct


    # 6) Start MCMC
    log_llk_proposed = -float("inf")
    log_llk_current = -float("inf")

    progress_intervals = set(
        int(num_iterations * frac) for frac in np.arange(0.1, 1.01, 0.1)
    )

    def print_progress_bar(iteration, total):
        fraction = iteration / total
        bar_len = 40
        filled = int(bar_len * fraction)
        bar = "="*filled + "-"*(bar_len - filled)
        print(f"\rProgress: [{bar}] {100*fraction:.1f}%  ({iteration}/{total})", end="")
        if iteration == total:
            print()


    for iteration in range(1, num_iterations + 1):
        r = rng.random()

        iteration_list.append(iteration)
        accepted_this_iter = False
        update_category = None
        total_prior_time = 0.0
        total_likelihood_time = 0.0
        update_type_timing = 0.0 
            # Build partial orders from current (U0, Ua)

        
        log_llk_current= HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U={"U0": U0, "U_a_dict": U_a_dict},
                    h_U=h_U,
                    observed_orders=observed_orders_cl,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict_cl,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha
                )
        # ------------------------------------------------
        # Update 1: Rho
        # ------------------------------------------------
        if r < thresh_rho:
            update_category = "rho"
            upd_start = time.time()
            delta = rng.uniform(dr, 1.0 / dr)
            rho_prime = 1.0 - (1.0 - rho) * delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho

            prior_start = time.time()
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K,rho_prime) 
            log_prior_current = StatisticalUtils.dRprior(rho, rho_prior)
            log_prior_proposed = StatisticalUtils.dRprior(rho_prime, rho_prior)
            total_prior_time = time.time() - prior_start

            # Assume likelihood remains unchanged with rho update
                   
            llk_start = time.time()     
            log_llk_proposed = log_llk_current
            dt = time.time() - llk_start
            total_likelihood_time = dt

            log_accept = (log_prior_proposed ) - (log_prior_current ) - math.log(delta)
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                rho = rho_prime
                Sigma_rho = Sigma_rho_prime  # for future steps
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_rho_vals.append(rho_prime)
            update_type_timing = time.time() - upd_start

        # -------------------------------------------
        # -----
        # Update 2: Tau
        # ------------------------------------------------
        elif r < thresh_tau:
            update_category = "tau"
            upd_start = time.time()
            tau_prime = StatisticalUtils.rTauprior()

            prior_start = time.time()
            lp_current = StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
            lp_proposed = StatisticalUtils.log_U_a_prior(U_a_dict, tau_prime, rho, K, M_a_dict, U0)
            total_prior_time = time.time() - prior_start

            llk_start = time.time()     
            log_llk_proposed = log_llk_current
            dt = time.time() - llk_start
            total_likelihood_time = dt

            log_accept = lp_proposed - lp_current
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                tau = tau_prime
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_tau_vals.append(tau_prime)
            update_type_timing= time.time() - upd_start

        # ------------------------------------------------
        # Update 3: (Rho, Tau) jointly
        # ------------------------------------------------
        elif r < thresh_rho_tau:
            update_category = "rho_tau"
            upd_start = time.time()
            delta = rng.uniform(drrt, 1.0 / drrt)
            rho_prime = 1.0 - (1.0 - rho) * delta
            tau_prime = 1.0 - (1.0 - tau) / delta
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K,rho_prime) 
            if not (0 < rho_prime < 1):
                rho_prime = rho
            if not (0 < tau_prime < 1):
                tau_prime = tau

            prior_start = time.time()
            lp_current = ( StatisticalUtils.log_U_prior(U0, rho, K)
                           + StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
                           + StatisticalUtils.dTauprior(tau)
                           + StatisticalUtils.dRprior(rho, rho_prior) )
            lp_proposed = ( StatisticalUtils.log_U_prior(U0, rho_prime, K)
                           + StatisticalUtils.log_U_a_prior(U_a_dict, tau_prime, rho_prime, K, M_a_dict, U0)
                           + StatisticalUtils.dTauprior(tau_prime)
                           + StatisticalUtils.dRprior(rho_prime, rho_prior) )
            total_prior_time = time.time() - prior_start


            llk_start = time.time()     
            log_llk_proposed = log_llk_current
            dt = time.time() - llk_start
            total_likelihood_time = dt

            log_accept = (lp_proposed) - (lp_current) - 2 * math.log(delta)
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                rho = rho_prime
                tau = tau_prime
                Sigma_rho = Sigma_rho_prime
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_rho_vals.append(rho_prime)
            proposed_tau_vals.append(tau_prime)
            update_type_timing= time.time() - upd_start

        # ------------------------------------------------
        # Update 4: Noise parameter(s)
        # ------------------------------------------------
        elif r < thresh_noise:
            update_category = "noise"
            upd_start = time.time()

            if noise_option == "mallows_noise":
                prior_start = time.time()
                epsilon = rng.normal(0, 1)
                mallow_theta_prime = mallow_theta * math.exp(sigma_mallow * epsilon)

                lp_current = StatisticalUtils.dTprior(mallow_theta, ua=mallow_ua)
                lp_proposed = StatisticalUtils.dTprior(mallow_theta_prime, ua=mallow_ua)
                total_prior_time = time.time() - prior_start

                llk_start = time.time()
        
                log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U={"U0": U0, "U_a_dict": U_a_dict},
                    h_U=h_U,
                    observed_orders=observed_orders_cl,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict_cl,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta_prime,
                    noise_option=noise_option,
                    alpha=alpha
                )
                dt = time.time() - llk_start
                total_likelihood_time = dt

                log_accept = ((lp_proposed + log_llk_proposed) 
                              - (lp_current + log_llk_current)
                              + math.log(mallow_theta / mallow_theta_prime))

                accept_prob = min(1.0, math.exp(min(log_accept, 700)))
                if rng.random() < accept_prob:
                    mallow_theta = mallow_theta_prime
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                proposed_mallow_theta_vals.append(mallow_theta_prime)
                update_type_timing= time.time() - upd_start

            elif noise_option == "queue_jump":
                prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)
                prior_start = time.time()
                lp_current = StatisticalUtils.dPprior(prob_noise, noise_beta_prior)
                lp_proposed= StatisticalUtils.dPprior(prob_noise_prime, noise_beta_prior)
                dt = time.time() - prior_start
                total_prior_time = dt

                llk_start = time.time()
                
                log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U={"U0": U0, "U_a_dict": U_a_dict},
                    h_U=h_U,
                    observed_orders=observed_orders_cl,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict_cl,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise_prime,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha
                )
                dt = time.time() - llk_start
                total_likelihood_time = dt

                log_accept = (log_llk_proposed - log_llk_current)
                accept_prob = min(1.0, math.exp(min(log_accept, 700)))
                if rng.random() < accept_prob:
                    prob_noise = prob_noise_prime
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                proposed_prob_noise_vals.append(prob_noise_prime)

    
            update_type_timing=  time.time() - upd_start

        # ------------------------------------------------
        # Update 5: U0 
        # ------------------------------------------------
        elif r < thresh_U0:
            update_category = "U0"
            upd_start = time.time()
            j_global = rng.integers(0, n_global)
            # Propose a new row for U0

            proposed_row = StatisticalUtils.U0_conditional_update(
                j_global,
                U0,
                U_a_dict,
                M_a_dict,
                tau,
                Sigma_rho,
                rng
            )
            # Make a copy
            U0_prime = U0.copy()
            U0_prime[j_global,:] = proposed_row

            # Build partial orders
            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=cluster_ids,
                M_a_dict=M_a_dict,
                U0=U0_prime,
                U_a_dict=U_a_dict,
                alpha=alpha
            )

            llk_start = time.time()     
            log_llk_proposed = log_llk_current
            dt = time.time() - llk_start
            total_likelihood_time = dt

            prior_start = time.time()

            lp_current = ( StatisticalUtils.log_U_prior(U0, rho, K)
                         + StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0) )
            lp_proposed= ( StatisticalUtils.log_U_prior(U0_prime, rho, K)
                         + StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0_prime) )
  
            total_prior_time = time.time() - prior_start



            log_accept = (lp_proposed - lp_current )
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                U0 = U0_prime
                U_a_dict=U_a_dict
                h_U = h_U_prime
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
                num_acceptances +=1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_U0.append(U0_prime)
            update_type_timing = time.time() - upd_start

        # ------------------------------------------------
        # Update 6: Ua
        # ------------------------------------------------
        elif r < thresh_Ua:
            update_category = "Ua"
            upd_start = time.time()
            a_key = random.choice(cluster_ids)
            M_a = M_a_dict.get(a_key, [])

            n_a = len(M_a)
            row_loc = rng.integers(0, n_a)
            old_val = U_a_dict[a_key][row_loc, :].copy()
            Sigma = BasicUtils.build_Sigma_rho(K, rho) * (1 - tau**2)
            proposed_row = rng.multivariate_normal(mean=old_val, cov=Sigma)

            U_a_dict_prime = copy.deepcopy(U_a_dict)
            U_a_dict_prime[a_key][row_loc, :] = proposed_row

            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=cluster_ids,
                M_a_dict=M_a_dict,
                U0=U0,
                U_a_dict=U_a_dict_prime,
                alpha=alpha
            )

            prior_start = time.time()
            lp_current = StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
            lp_proposed= StatisticalUtils.log_U_a_prior(U_a_dict_prime, tau, rho, K, M_a_dict, U0)
      
            total_prior_time = time.time() - prior_start

            llk_start = time.time()

            log_llk_proposed= HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U={"U0": U0, "U_a_dict": U_a_dict_prime},
                h_U=h_U_prime,
                observed_orders=observed_orders_cl,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict_cl,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha
            )

            total_likelihood_time = time.time() - llk_start

            log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                U_a_dict = U_a_dict_prime
                log_llk_current = log_llk_proposed
                h_U = h_U_prime
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)


            update_type_timing= time.time() - upd_start
        # ------------------------------------------------
        # Update 7: Reversible jump for K dimension
        # ------------------------------------------------
        elif r < thresh_Kdim:
            update_category = "K_dim"
            upd_start = time.time()

            # Decide whether to do an "up" or "down" move
            if K == 1:
                move = "up"
            else:
                move = "up" if rng.random() < 0.5 else "down"

            if move == "up":
                K_prime = K + 1
                # We'll pick a random column index to insert, or just append at the end
                col_ins = rng.integers(0, K_prime)  # Insert at [0..K_prime-1]
                
                # Insert a new dimension in U0
                new_col_U0 = StatisticalUtils.sample_conditional_column(U0, rho)  # shape (n_global,)
                U0_prime = np.insert(U0, col_ins, new_col_U0, axis=1)
                Sigma_rho_prime = BasicUtils.build_Sigma_rho(K_prime, rho)

                # For each assessor's Ua, insert a new dimension, we need to find the global k_prime index 
                U_a_dict_prime = copy.deepcopy(U_a_dict)
                for a in cluster_ids:
                    Ua_mat = U_a_dict_prime[a]
                    new_col_Ua = StatisticalUtils.sample_conditional_column(Ua_mat, rho)  ## when we generate 
                    U_a_dict_prime[a] = np.insert(Ua_mat, col_ins, new_col_Ua, axis=1)

                # Build partial orders
                h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                    M0=M0,
                    assessors=cluster_ids,
                    M_a_dict=M_a_dict,
                    U0=U0_prime,
                    U_a_dict=U_a_dict_prime,
                    alpha=alpha
                )
                prior_start = time.time()
                # Evaluate log-prob K, K_prime
                logK_current  = StatisticalUtils.dKprior(K, K_prior)
                logK_proposed = StatisticalUtils.dKprior(K_prime, K_prior)

                # Evaluate prior and likelihood
                
                lp_current = logK_current 
                lp_proposed= logK_proposed 
            
                total_prior_time = time.time() - prior_start

                llk_start = time.time()
                log_llk_proposed= HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U={"U0": U0_prime, "U_a_dict": U_a_dict_prime},
                    h_U=h_U_prime,
                    observed_orders=observed_orders_cl,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict_cl,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha
                )
            
                total_likelihood_time = time.time() - llk_start
                ## this is for the adjustment for k--1
                rho_fk = 1.0 if K==1 else 0.5
                rho_bk = 0.5      # always 1/2 on the way back from Kp→K
                log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)+ math.log(rho_bk) - math.log(rho_fk)
                accept_prob = min(1.0, math.exp(min(log_accept, 700)))
                if rng.random() < accept_prob:
                    K = K_prime
                    U0 = U0_prime
                    U_a_dict = U_a_dict_prime
                    Sigma_rho = Sigma_rho_prime
                    h_U = h_U_prime
                    log_llk_current =  log_llk_proposed
                    accepted_this_iter = True
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                update_type_timing= time.time() - upd_start
            else:
                # Move down
                K_prime = K - 1
                if K_prime < 1:
                    acceptance_decisions.append(0)
                else:
                    col_del = rng.integers(0, K)
                    # Remove that dimension from U0, shape => (n_global, K_prime)
                    U0_prime = np.delete(U0, col_del, axis=1)
                    Sigma_rho_prime = BasicUtils.build_Sigma_rho(K_prime, rho)

                    # Remove that dimension from each Ua
                    U_a_dict_prime = copy.deepcopy(U_a_dict)
                    for a in cluster_ids:
                        Ua_mat = U_a_dict_prime[a]
                        U_a_dict_prime[a] = np.delete(Ua_mat, col_del, axis=1)

                    h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                        M0=M0,
                        assessors=cluster_ids,
                        M_a_dict=M_a_dict,
                        U0=U0_prime,
                        U_a_dict=U_a_dict_prime,
                        alpha=alpha
                    )
                    prior_start = time.time()
                    # Evaluate prior and likelihood
                    logK_current  = StatisticalUtils.dKprior(K, K_prior)
                    logK_proposed = StatisticalUtils.dKprior(K_prime, K_prior)
                    lp_current = logK_current 
                    lp_proposed= logK_proposed
              
                    total_prior_time = time.time() - prior_start

                    llk_start = time.time()

                    log_llk_proposed= HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                        U={"U0": U0_prime, "U_a_dict": U_a_dict_prime},
                        h_U=h_U_prime,
                        observed_orders=observed_orders_cl,
                        M_a_dict=M_a_dict,
                        O_a_i_dict=O_a_i_dict_cl,
                        item_to_index=item_to_index,
                        prob_noise=prob_noise,
                        mallow_theta=mallow_theta,
                        noise_option=noise_option,
                        alpha=alpha
                    )
                    total_likelihood_time = time.time() - llk_start

                    log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)
                    accept_prob = min(1.0, math.exp(min(log_accept, 700)))
           
                    if rng.random() < accept_prob:
                        K = K_prime
                        U0 = U0_prime
                        Sigma_rho = Sigma_rho_prime
                        U_a_dict = U_a_dict_prime
                        h_U = h_U_prime
                        log_llk_current = log_llk_proposed
                        accepted_this_iter = True
                        num_acceptances += 1
                        acceptance_decisions.append(1)
                    else:
                        acceptance_decisions.append(0)

                update_type_timing= time.time() - upd_start 

        # ----------------------------------------------------------------
        # End of updates, record iteration info
        # ----------------------------------------------------------------
        elif r < thresh_beta:
            update_category = "beta"
            upd_start = time.time()
            prior_start = time.time()
            # we update one beta each time
            j = (iteration-1) % p
            epsilon =   rng.normal(loc=0.0, scale=drbeta * sigma_beta)
            beta_prime = beta.copy()
            beta_prime[j] += epsilon

            alpha_prime = X.T @ beta_prime
            
            # Use sigma_beta_array for the prior calculation
            lp_current = StatisticalUtils.dBetaprior(beta, sigma_beta)
            lp_proposed = StatisticalUtils.dBetaprior(beta_prime, sigma_beta)
            total_prior_time = time.time() - prior_start

            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=cluster_ids,
                M_a_dict=M_a_dict,
                U0=U0,
                U_a_dict=U_a_dict,
                alpha=alpha_prime
            )

            llk_start = time.time()
        
            log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U={"U0": U0, "U_a_dict": U_a_dict},
                h_U=h_U_prime,
                observed_orders=observed_orders_cl,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict_cl,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha_prime
            )
            total_likelihood_time = time.time() - llk_start

            log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                beta = beta_prime
                alpha = alpha_prime  # update alpha as well
                log_llk_current = log_llk_proposed
                h_U = h_U_prime
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            update_type_timing= time.time() - upd_start

        # ------------------------------------------------
        # Update 8: Cluster
        # ------------------------------------------------
        elif r < thresh_cl:
            update_category = "cluster"
            upd_start = time.time()
            # Sample a list to reassign
            e = rng.integers(0, n_lists)
            old_k = c_vec[e]
            e_list = observed_orders[e]
            emptied_now    = (na_dict[old_k] == 1) 
            # 2) Remove from old cluster
            pos_in_cluster = None
            for iPos, local_ord in enumerate(observed_orders_cl[old_k]):
                if local_ord == e_list:
                    pos_in_cluster = iPos
                    break
            if pos_in_cluster is not None:
                observed_orders_cl[old_k].pop(pos_in_cluster)
                O_a_i_dict_cl[old_k].pop(pos_in_cluster)
                na_dict[old_k] -= 1
        
            
            # Calculate weights for each existing cluster and a potential new cluster
            logw = []
            # For each existing cluster, calculate prior and likelihood
            for k in cluster_ids:
                nk_minus = na_dict[k]
                # Prior based on Chinese Restaurant Process
                prior_weight = math.log(max(nk_minus - Dri_alpha, 0.0))
                # Create a temporary merged M_a that would include the new elements
                M_a_temp = set(M_a_dict[k])
                # Add any items from e_list that aren't already in the cluster
                for it in e_list:
                    M_a_temp.add(it)
                M_a_temp = list(sorted(M_a_temp))

                old_Ua = U_a_dict[k]
                # Create a temporary U_a for the augmented cluster
                if len(M_a_temp) == len(M_a_dict[k]):
                    Ua_temp = old_Ua
                    
                else:
                    Kdim = old_Ua.shape[1]
                    Ua_temp = np.zeros((len(M_a_temp), Kdim))
                    for iPos, item_i in enumerate(M_a_temp):
                        if item_i in M_a_dict[k]:
                            idx_old = M_a_dict[k].index(item_i)
                            Ua_temp[iPos] = old_Ua[idx_old]
                        else:
                            j_global = item_to_index[item_i]
                            mean_vec = tau*U0[j_global,:]
                            cov_mat = (1 - tau**2)*Sigma_rho
                            Ua_temp[iPos] = rng.multivariate_normal(mean_vec, cov_mat)
                Ua_dict_temp=  copy.deepcopy(U_a_dict)
                Ua_dict_temp[k] = Ua_temp

                M_a_dict_temp =copy.deepcopy(M_a_dict)
                M_a_dict_temp[k] = M_a_temp

                h_U_temp = StatisticalUtils.build_hierarchical_partial_orders(
                    M0=M0,
                    assessors=cluster_ids,
                    M_a_dict= M_a_dict_temp,
                    U0=U0,
                    U_a_dict=Ua_dict_temp,
                    alpha=alpha
                )
                
                # Likelihood of assigning list e to cluster k
                llk_e = HPO_LogLikelihoodCache.calculate_single_list_likelihood(
                    list_idx=e,
                    e_list=e_list,
                    cluster_id=k,
                    U0=U0,
                    U_a_dict=Ua_dict_temp,
                    observed_orders=observed_orders_cl,
                    M_a_dict=M_a_dict_temp,
                    O_a_i_dict=O_a_i_dict_cl,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha,
                    h_U=h_U_temp
                )
                
                logw.append(prior_weight + llk_e)
            
            # Add weight for creating a new cluster
            prior_new = math.log(Dri_theta + Dri_alpha * len(cluster_ids))
            
            Kdim = K
            Ua_temp= np.zeros((len(e_list), Kdim))
            for iPos, it in enumerate(e_list):
                j_global = item_to_index[it]
                mean_vec = tau*U0[j_global,:]
                cov_mat = (1 - tau**2)*Sigma_rho
                Ua_temp[iPos,:] = rng.multivariate_normal(mean_vec, cov_mat)

            # Build hierarchical partial orders for the new cluster
            new_k = max(cluster_ids) + 1 
            Ua_dict_temp= copy.deepcopy(U_a_dict)
            Ua_dict_temp[new_k] = Ua_temp

            M_a_dict_temp =copy.deepcopy(M_a_dict)
            M_a_dict_temp[new_k] = M_a_temp
            
            h_U_temp = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=cluster_ids + [new_k],
                M_a_dict= M_a_dict_temp,
                U0=U0,
                U_a_dict=Ua_dict_temp,
                alpha=alpha
            )
            
            # Calculate likelihood for new cluster
            likelihood_new = HPO_LogLikelihoodCache.calculate_single_list_likelihood(
                list_idx=e,
                e_list=e_list,
                cluster_id=new_k,
                U0=U0,
                U_a_dict=Ua_dict_temp,
                observed_orders=observed_orders_cl,
                M_a_dict=M_a_dict_temp,
                O_a_i_dict=O_a_i_dict_cl,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha,
                h_U=h_U_temp
            )
            logw.append(prior_new+likelihood_new)
            
            # Normalize weights and sample
            max_logw = np.max(logw)
            w = np.exp(np.array(logw) - max_logw)
            w /= w.sum()
            choice = rng.choice(len(w), p=w)
            
            # Make a deep copy of current structure for proposed state
            c_vec_prime = c_vec.copy()
            M_a_dict_prime = copy.deepcopy(M_a_dict)
            U_a_dict_prime = copy.deepcopy(U_a_dict)
            na_dict_prime = copy.deepcopy(na_dict)
            cluster_ids_prime = cluster_ids.copy() 
            observed_orders_cl_prime = copy.deepcopy(observed_orders_cl)
            O_a_i_dict_cl_prime = copy.deepcopy(O_a_i_dict_cl)
            # Assign to chosen cluster
            if choice < len(cluster_ids):
                # Existing cluster
                new_k = cluster_ids[choice]
      
                M_a_dict_prime[new_k]= M_a_dict[new_k]
                U_a_dict_prime[new_k] = Ua_dict_temp[new_k]
                c_vec_prime[e] = new_k
                na_dict_prime[new_k] += 1
                observed_orders_cl_prime[new_k].append(e_list)
                O_a_i_dict_cl_prime[new_k].append(sorted(e_list))  # the choice set
                                         
            else:
                # Create new cluster
                chosen_k = new_k
                cluster_ids_prime.append(chosen_k)
                c_vec_prime[e] = chosen_k
                na_dict_prime[chosen_k] = 1
                M_a_dict_prime =M_a_dict_temp
                U_a_dict_prime =Ua_dict_temp
                observed_orders_cl_prime[chosen_k] = [ e_list ]
                O_a_i_dict_cl_prime[chosen_k] = [ sorted(e_list)]
                            
            if emptied_now and new_k != old_k:
                # old_k is truly empty => purge it
                del na_dict_prime[old_k]
                del M_a_dict_prime[old_k]
                del U_a_dict_prime[old_k]
                del observed_orders_cl_prime[old_k]
                del O_a_i_dict_cl_prime[old_k]
                cluster_ids_prime.remove(old_k)           
            # Build hierarchical partial orders with proposed assignments
            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=cluster_ids_prime,
                M_a_dict=M_a_dict_prime,
                U0=U0,
                U_a_dict=U_a_dict_prime,
                alpha=alpha
            )
            
            # Calculate likelihood of proposed state
            log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U={"U0": U0, "U_a_dict": U_a_dict_prime},
                h_U=h_U_prime,
                observed_orders=observed_orders_cl_prime,
                M_a_dict=M_a_dict_prime,
                O_a_i_dict=O_a_i_dict_cl_prime,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha
            )
            
            # Calculate acceptance probability (Metropolis-Hastings)
            
            accept_prob = 1
            
            # Timing
            update_type_timing = time.time() - upd_start
            
            # Accept or reject
            c_vec = c_vec_prime
            M_a_dict = M_a_dict_prime
            U_a_dict = U_a_dict_prime
            observed_orders_cl = observed_orders_cl_prime
            na_dict = na_dict_prime
            cluster_ids = cluster_ids_prime
            h_U = h_U_prime
            log_llk_current = log_llk_proposed
            accepted_this_iter = True
            num_acceptances += 1
            acceptance_decisions.append(1)


        # ----------------------------------------------------------------
        # End of updates, record iteration info
        # ----------------------------------------------------------------
        current_accept_rate = num_acceptances / iteration
        acceptance_rates.append(current_accept_rate)
        update_category_list.append(update_category)
        prior_timing_list.append(total_prior_time)
        likelihood_timing_list.append(total_likelihood_time)
        update_timing_list.append(update_type_timing)

        # If iteration in some storing scheme
        if iteration % 100 == 0:
            rho_trace.append(rho)
            tau_trace.append(tau)
            prob_noise_trace.append(prob_noise)
            mallow_theta_trace.append(mallow_theta)
            K_trace.append(K)
            beta_trace.append(beta)
            U0_trace.append(U0.copy())
            Ua_trace.append(copy.deepcopy(U_a_dict))
            H_trace.append(copy.deepcopy(h_U))
            update_records.append((iteration, update_category, accepted_this_iter))

        log_likelihood_currents.append(log_llk_current)
        # We used log_llk_proposed for new states, but we can store it if needed
        log_likelihood_primes.append(log_llk_proposed)  # or keep a separate variable

        if iteration in progress_intervals:
            print(f"Iteration {iteration}/{num_iterations} - Accept Rate: {current_accept_rate:.2%}")

    # Done
    overall_acceptance_rate = num_acceptances / num_iterations
    print_progress_bar(num_iterations, num_iterations)
    update_df = pd.DataFrame(update_records, columns=["iteration", "category", "accepted"])

    result_dict = {
        "rho_trace": rho_trace,
        "tau_trace": tau_trace,
        "prob_noise_trace": prob_noise_trace,
        "mallow_theta_trace": mallow_theta_trace,
        "K_trace": K_trace,
        "beta_trace": beta_trace,

        "U0_trace": U0_trace,
        "Ua_trace": Ua_trace,
        "H_trace": H_trace,

        "proposed_rho_vals": proposed_rho_vals,
        "proposed_tau_vals": proposed_tau_vals,
        "proposed_prob_noise_vals": proposed_prob_noise_vals,
        "proposed_mallow_theta_vals": proposed_mallow_theta_vals,
        "proposed_U0": proposed_U0,

        "acceptance_decisions": acceptance_decisions,
        "acceptance_rates": acceptance_rates,
        "overall_acceptance_rate": overall_acceptance_rate, 
        "log_likelihood_currents": log_likelihood_currents,
        "log_likelihood_primes": log_likelihood_primes,
        "num_acceptances": num_acceptances,

        # Final state
        "rho_final": rho,
        "tau_final": tau,
        "beta_final": beta,
        "prob_noise_final": prob_noise,
        "mallow_theta_final": mallow_theta,
        "K_final": K,
        "U0_final": U0,
        "U_a_final": U_a_dict,
        "H_final": h_U,
        "update_df": update_df,

        # Timing
        "total_prior_timing": total_prior_time,
        "total_likelihood_timing": total_likelihood_time,
        "update_type_timing": update_type_timing,

        # Per-iteration lists
        "iteration_list": iteration_list,
        "update_category_list": update_category_list,
        "prior_timing_list": prior_timing_list,
        "likelihood_timing_list": likelihood_timing_list,
        "update_timing_list": update_timing_list
    }
    return result_dict