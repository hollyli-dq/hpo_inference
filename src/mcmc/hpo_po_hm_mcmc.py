import numpy as np
import math
from scipy.stats import multivariate_normal, norm

import random
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import sys as sys
import copy
import time  # <-- For timing

import itertools
import random
import math
import numpy as np
from scipy.stats import beta, gamma

from typing import Dict, List

from src.utils.po_fun import BasicUtils, StatisticalUtils
from src.utils.po_accelerator_nle import HPO_LogLikelihoodCache




def mcmc_simulation_hpo(
    num_iterations: int,
    # Hierarchy definition
    M0: List[int],
    assessors: List[int],
    M_a_dict:Dict[int, List[int]],
    # Observed data
    O_a_i_dict: Dict[int, List[List[int]]], 
    observed_orders: Dict[int, List[List[int]]],
    # Additional model parameters
    sigma_beta: np.ndarray,  
    X,  # X: np.ndarray of covariates 
    K: int,                  # dimension of latent space
    dr: float,  # multiplicative step size for rho
    drrt: float,  # multiplicative step size for tau and rho 
    drbeta: float,      # the beta updated step size
    # noise / priors
    sigma_mallow: float,
    noise_option: str,
    # pcts 
    mcmc_pt: List[float],
    # priors
    rho_prior, 
    noise_beta_prior: float,
    mallow_ua: float,
    # Optional
    rho_tau_update: bool = False,
    random_seed: int = 42 
                       ) -> Dict[str, Any]:


    # -----------------[ Per-Iteration Timing Lists ]-------------


    iteration_list = []
    update_category_list = []
    prior_timing_list = []      # time for prior computations in this iteration
    likelihood_timing_list = [] # time for likelihood calculation in this iteration
    update_timing_list = [] 


    prior_start = time.time()

    # ----------------- Initialization -----------------
    # 0) Seeds
    rng = np.random.default_rng(random_seed)

    # M0 is given; create an item mapping (assuming items are unique in M0)
    items = sorted(set(M0))
    item_to_index = {item: idx for idx, item in enumerate(items)}
    n_global = len(M0)

    # 1) Sample initial rho, tau from prior
    rho = StatisticalUtils.rRprior(rho_prior)  # initial rho from its prior
    prob_noise = StatisticalUtils.rPprior(noise_beta_prior) # Beta(1, noise_beta_prior)
    mallow_theta = StatisticalUtils.rTprior(mallow_ua)   # e.g. uniform(0.1,0.9)
    tau = StatisticalUtils.rTauprior()  # initial tau from its prior
    Sigma_rho = BasicUtils.build_Sigma_rho(K,rho)



    p = X.shape[0]
    beta_true = rng.normal(loc=0.0, scale=sigma_beta, size=(p,))
    beta = beta_true.copy()  # add beta so that update beta later works
    alpha = X.T @ beta
    Sigma_prop = (drbeta ** 2) * np.eye(p)
    # 3) Sample global U^(0) from N(0, Sigma_rho)

    U0 = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma_rho, size=n_global)

    # Initilize of the Ua 

    U_a_dict = {}
    for a in assessors:
        M_a = M_a_dict.get(a, [])
        n_a = len(M_a)
        Ua = np.zeros((n_a, K), dtype=float)
        for i_loc, j_global in enumerate(M_a):
            mean_vec = tau * U0[j_global, :]
            cov_mat = (1.0 - tau**2) * Sigma_rho
            Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
        U_a_dict[a] = Ua


    # Storage for traces.
    U0_trace = []
    Ua_trace = []
    H_trace = []  
    rho_trace = []
    tau_trace = []
    prob_noise_trace = []
    beta_trace=[]
    mallow_theta_trace = []
    proposed_rho_vals = []
    proposed_tau_vals = []
    proposed_prob_noise_vals = []
    proposed_mallow_theta_vals = []
    proposed_U0 = []
    proposed_U_a = {}
    acceptance_decisions = []
    acceptance_rates = []
    log_likelihood_currents = []
    log_likelihood_primes = []
    update_records=[]
    
    num_acceptances = 0
    # Unpack update probabilities.

     # Convert to a NumPy array and normalize:
    mcmc_pt = np.array(mcmc_pt)
    mcmc_pt = mcmc_pt / mcmc_pt.sum()

# Now unpack the normalized percentages:
    rho_pct, tau_pct, rho_tau_pct, noise_pct, U0_pct, Ua_pct, beta_pct = mcmc_pt

    thresh_rho = rho_pct
    thresh_tau =thresh_rho + tau_pct
    threshold_rho_tau_pct = thresh_tau + rho_tau_pct
    thresh_noise =  threshold_rho_tau_pct+ noise_pct
    threshold_U0_pct =  thresh_noise+ U0_pct
    threshold_Ua_pct = threshold_U0_pct+ Ua_pct
    thresh_beta       =threshold_Ua_pct  + beta_pct  
 
    progress_intervals = set([0]) | set(int(num_iterations * frac) for frac in np.arange(0.05, 1.05, 0.05))


    def print_progress_bar(iteration, total):
        fraction = iteration / total
        bar_length = 40  # adjust for width
        filled = int(bar_length * fraction)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\rProgress: [{bar}] {fraction*100:.1f}%  (Iteration {iteration}/{total})", end="")
        if iteration == total:
            print()  # newline at the end
    log_llk_current = -np.inf

    for iteration in range(1,num_iterations+1):
        upd_start = time.time()  # Start timing the update block

        # Reset per-iteration timers
        iter_prior_time = 0.0
        iter_likelihood_time = 0.0
        iter_update_time = 0.0
        accepted_this_iter = False
        update_category = None
        # Record iteration number
        iteration_list.append(iteration)
        r = random.random()
        # We'll define "U" as a dict holding both global a local latents

        U= {"U0": U0, "U_a_dict": U_a_dict}


        #  Build partial orders h_U from the current latents
        h_U = StatisticalUtils.build_hierarchical_partial_orders(
            M0=M0,
            assessors=assessors,
            M_a_dict=M_a_dict,
            U0=U0,
            U_a_dict=U_a_dict,
            alpha=alpha
            # link_inv=... if not the default

        )

        if r < thresh_rho:
            upd_start = time.time()
            update_category = "rho"  # For timing
            delta = random.uniform(dr, 1.0 / dr)
            rho_prime = 1.0 - (1.0 - rho) * delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho

            prior_start = time.time()

            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K,rho_prime) 
            log_prior_current = (
                StatisticalUtils.dRprior(rho,rho_prior) 
            )
            log_prior_proposed = StatisticalUtils.dRprior(rho_prime,rho_prior)

            total_prior_time = time.time() - prior_start

            llk_start = time.time()     
            log_llk_proposed = log_llk_current
            dt = time.time() - llk_start
            total_likelihood_time = dt

            log_acceptance_ratio = (log_prior_proposed + log_llk_proposed) - np.log(delta)


            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
            if random.random() < acceptance_probability:
                rho = rho_prime
                Sigma_rho = Sigma_rho_prime  # for future steps
                log_llk_current = log_llk_proposed
                num_acceptances += 1
                acceptance_decisions.append(1)
                accepted_this_iter = True
            else:
                acceptance_decisions.append(0)
            proposed_rho_vals.append(rho_prime)                        
            update_type_timing = time.time() - upd_start



        # ---- B) Update tau ----

        elif r < thresh_tau:
            update_category = "tau"
            upd_start = time.time()
            tau_prime = StatisticalUtils.rTauprior()

            prior_start = time.time()
            log_prior_current = (

                 StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
      
            )
            # new prior
            log_prior_proposed = (
      
                StatisticalUtils.log_U_a_prior(U_a_dict, tau_prime, rho, K, M_a_dict, U0)
            )
            total_prior_time = time.time() - prior_start





            llk_start = time.time()     
            log_llk_proposed = log_llk_current
            dt = time.time() - llk_start
            total_likelihood_time = dt

            # Data-likelihood may or may not change if code uses tau explicitly in the likelihood
            # We'll assume it does not, or does so in partial
            # We'll skip re-sampling U => same partial approach
            log_accept_ratio = log_prior_proposed - log_prior_current 
            
            log_accept_ratio=min(log_accept_ratio,700)
            accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))


            if random.random() < accept_prob:
                tau = tau_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                accepted_this_iter = True

            else:
                acceptance_decisions.append(0)
            proposed_tau_vals.append(tau_prime)
            update_type_timing= time.time() - upd_start



        ## Update rho and tau together
        elif r < threshold_rho_tau_pct:
            update_category = "rho_tau"
            delta = random.uniform(drrt, 1.0 / drrt)
            upd_start = time.time()
            rho_prime = 1.0 - (1.0 - rho) * delta
            tau_prime =1.0 - (1.0 - tau) * (1/delta)

            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho
            if not (0.0 < tau_prime < 1.0):
                tau_prime = tau
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K,rho_prime) 

            prior_start = time.time()
            log_prior_current= (
                StatisticalUtils.log_U_prior(U0, rho, K)
                +StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
                + StatisticalUtils.dTauprior(tau)
                + StatisticalUtils.dRprior(rho, rho_prior)
            )
            log_prior_proposed= (
                StatisticalUtils.log_U_prior(U0, rho_prime, K)
                + StatisticalUtils.log_U_a_prior(U_a_dict, tau_prime, rho_prime, K, M_a_dict, U0)
                + StatisticalUtils.dTauprior(tau_prime)
                + StatisticalUtils.dRprior(rho_prime, rho_prior)
            )
            total_prior_time = time.time() - prior_start



            llk_start = time.time()     
            log_llk_proposed = log_llk_current
            dt = time.time() - llk_start
            total_likelihood_time = dt

            ## The likelihood terms cancels out since there is no change in the input for likelihood calcualtion 
            log_acceptance_ratio = log_prior_proposed- log_prior_current - 2 * math.log(delta)
            

            if random.random() < min(1.0, np.exp(log_acceptance_ratio)):
                rho = rho_prime
                tau = tau_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                accepted_this_iter = True
            else:
                acceptance_decisions.append(0)
            proposed_tau_vals.append(tau_prime)
            proposed_rho_vals.append(rho_prime)   
            update_type_timing= time.time() - upd_start


        # ---- B) Update noise parameter ----
        elif r < thresh_noise:
            update_category = "noise"
            upd_start = time.time()
            if noise_option == "mallows_noise":
                prior_start = time.time()
                epsilon = np.random.normal(0, 1)
                mallow_theta_prime = mallow_theta * np.exp(sigma_mallow * epsilon)
                log_prior_current = StatisticalUtils.dTprior(mallow_theta, ua=mallow_ua)
                log_prior_proposed = StatisticalUtils.dTprior(mallow_theta_prime, ua=mallow_ua)
                total_prior_time = time.time() - prior_start

                # Evaluate new likelihood
                llk_start = time.time()
                log_llk_current = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U=U,
                    h_U=h_U,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,  # or your real index map if needed
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha
                )
                log_llk_proposed =HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U=U,
                    h_U=h_U,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta_prime,
                    noise_option=noise_option,
                    alpha=alpha
                )
                dt = time.time() - llk_start
                total_likelihood_time = dt

                log_accept_ratio = (
                    (log_prior_proposed + log_llk_proposed)
                    - (log_prior_current + log_llk_current)
                    # Jacobian for multiplicative => log(mallow_theta/mallow_theta_prime)
                    + math.log(mallow_theta / mallow_theta_prime)
                )
                accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))


                if random.random() < accept_prob:
                    mallow_theta = mallow_theta_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                else:
                    acceptance_decisions.append(0)

                proposed_mallow_theta_vals.append(mallow_theta_prime)
                update_type_timing= time.time() - upd_start



            elif noise_option == "queue_jump":

                prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)
                prior_start = time.time()

                log_prior_current = StatisticalUtils.dPprior(prob_noise, beta_param=noise_beta_prior)
                log_prior_proposed = StatisticalUtils.dPprior(prob_noise_prime, beta_param=noise_beta_prior)
                dt = time.time() - prior_start
                total_prior_time = dt


                # Evaluate new likelihood
                llk_start = time.time()
                log_llk_current = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U=U,
                    h_U=h_U,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,  # or your real index map if needed
                    prob_noise=prob_noise,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha
                )
                log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                    U= U,
                    h_U=h_U,
                    observed_orders=observed_orders,
                    M_a_dict=M_a_dict,
                    O_a_i_dict=O_a_i_dict,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise_prime,
                    mallow_theta=mallow_theta,
                    noise_option=noise_option,
                    alpha=alpha
                )

                dt = time.time() - llk_start
                total_likelihood_time = dt

                log_accept_ratio =log_llk_proposed-log_llk_current
                accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))



                if random.random() < accept_prob:
                    prob_noise = prob_noise_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                else:
                    acceptance_decisions.append(0)

                proposed_prob_noise_vals.append(prob_noise_prime)

            update_type_timing=  time.time() - upd_start



        elif r <threshold_U0_pct: # Update global latent U0
            update_category = "U0"

            # Update global latent U0
            n_global = len(M0)
            j_global = np.random.randint(0, n_global)

  
            Sigma = BasicUtils.build_Sigma_rho(K,rho)

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
                assessors=assessors,
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
            lp_proposed = (
                StatisticalUtils.log_U_prior(U0_prime, rho, K)
                + StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0_prime)
            )
     
            lp_current = (
                StatisticalUtils.log_U_prior(U0, rho, K)
                + StatisticalUtils.log_U_a_prior(U_a_dict, tau, rho, K, M_a_dict, U0)
            )

            total_prior_time += time.time() - prior_start


            # Acceptance ratio
            log_accept_ratio = (lp_proposed - lp_current )
            accept_prob = min(1.0, math.exp(min(log_accept_ratio, 700)))


            if random.random() < accept_prob:
                U0 = U0_prime
                U_a_dict=U_a_dict
                num_acceptances += 1
                acceptance_decisions.append(1)
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
            else:
                acceptance_decisions.append(0)
            proposed_U0.append(U0_prime)
            update_type_timing= time.time() - upd_start

        elif r<threshold_Ua_pct:
            update_category = "Ua"
            upd_start = time.time()
            a_key = random.choice(assessors)
            M_a = M_a_dict.get(a_key, [])
 
            n_a = len(M_a)
            row_loc = np.random.randint(0, n_a)
            old_value = U_a_dict[a_key][row_loc, :].copy()
            Sigma = BasicUtils.build_Sigma_rho(K, rho)*(1 - tau**2)
            proposed_row = rng.multivariate_normal(mean=old_value, cov=Sigma)
            U_a_dict_prime = copy.deepcopy(U_a_dict)
            U_a_dict_prime[a_key][row_loc, :] = proposed_row

            U_prime = {"U0": U0, "U_a_dict": U_a_dict_prime}

            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=assessors,
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

            log_llk_current = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
            U=U,
            h_U=h_U,
            observed_orders=observed_orders,
            M_a_dict=M_a_dict,
            O_a_i_dict=O_a_i_dict,
            item_to_index=item_to_index,  # or your real index map if needed
            prob_noise=prob_noise,
            mallow_theta=mallow_theta,
            noise_option=noise_option,
            alpha=alpha
        )

            log_llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U=U_prime,
                h_U=h_U_prime,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha
            )

            total_likelihood_time = time.time() - llk_start

            log_acceptance_ratio = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)
            if random.random() < min(1.0, np.exp(min(log_acceptance_ratio,700))):
                U_a_dict = U_a_dict_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
            else:
                acceptance_decisions.append(0)
            update_type_timing= time.time() - upd_start

        else:
            update_category = "beta"
            upd_start = time.time()


            prior_start = time.time()
            epsilon =  rng.multivariate_normal(np.zeros(p), Sigma_prop)
            beta_prime = beta + epsilon
            alpha_prime = X.T @ beta_prime
            lp_current = StatisticalUtils.dBetaprior(beta,sigma_beta)
            lp_proposed = StatisticalUtils.dBetaprior(beta_prime,sigma_beta)
            total_prior_time = time.time() - prior_start

            h_U_prime = StatisticalUtils.build_hierarchical_partial_orders(
                M0=M0,
                assessors=assessors,
                M_a_dict=M_a_dict,
                U0=U0,
                U_a_dict=U_a_dict,
                alpha=alpha_prime
            )

            llk_start = time.time()
            llk_current = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U=U,
                h_U=h_U,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha
            )
            llk_proposed = HPO_LogLikelihoodCache.calculate_log_likelihood_hpo(
                U=U,
                h_U=h_U_prime,
                observed_orders=observed_orders,
                M_a_dict=M_a_dict,
                O_a_i_dict=O_a_i_dict,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                mallow_theta=mallow_theta,
                noise_option=noise_option,
                alpha=alpha_prime
            )
            total_likelihood_time = time.time() - llk_start

            log_accept = (lp_proposed + llk_proposed) - (lp_current + llk_current)
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                beta = beta_prime
                alpha = alpha_prime  # update alpha as well
                log_llk_current = llk_proposed
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            update_type_timing= time.time() - upd_start
            
        current_accept_rate = num_acceptances / iteration
        acceptance_rates.append(current_accept_rate)
        update_category_list.append(update_category)
        prior_timing_list.append(total_prior_time)
        likelihood_timing_list.append(total_likelihood_time)
        update_timing_list.append(update_type_timing)
        # Append current parameter values to trace lists for debugging
        if iteration % 100 == 0:
            rho_trace.append(rho)
            tau_trace.append(tau)
            prob_noise_trace.append(prob_noise)
            mallow_theta_trace.append(mallow_theta)
            U0_trace.append(U0.copy())
            # For U_a_dict, store a deep copy.
            Ua_trace.append(copy.deepcopy(U_a_dict))
            H_trace.append(copy.deepcopy(h_U))
            acceptance_rates.append(num_acceptances / (iteration + 1))
            update_records.append((iteration, update_category, accepted_this_iter))

        current_acceptance_rate = num_acceptances / iteration

        log_likelihood_currents.append(log_llk_current)
        log_likelihood_primes.append(log_llk_proposed)

        if iteration in progress_intervals:
            print(f"Iteration {iteration}/{num_iterations} - Accept Rate: {current_acceptance_rate:.2%}")



    overall_acceptance_rate = num_acceptances / num_iterations
    update_df = pd.DataFrame(update_records, columns=["iteration", "category", "accepted"])

    print_progress_bar(num_iterations, num_iterations)
    
    result_dict = {
        "rho_trace": rho_trace,
        "tau_trace": tau_trace,
        "prob_noise_trace": prob_noise_trace,
        "mallow_theta_trace": mallow_theta_trace,
        "U0_trace": U0_trace,
        "Ua_trace": Ua_trace,
        "H_trace": H_trace,
        "beta_trace": beta_trace,
        "proposed_rho_vals": proposed_rho_vals,
        "proposed_tau_vals": proposed_tau_vals,
        "proposed_prob_noise_vals": proposed_prob_noise_vals,
        "proposed_mallow_theta_vals": proposed_mallow_theta_vals,
        "acceptance_decisions": acceptance_decisions,
        "acceptance_rates": acceptance_rates,
        "overall_acceptance_rate": overall_acceptance_rate, 
        "log_likelihood_currents": log_likelihood_currents,
        "log_likelihood_primes": log_likelihood_primes,
        "num_acceptances": num_acceptances,
        # Final state
        "rho_final": rho,
        "tau_final": tau,
        "prob_noise_final": prob_noise,
        "mallow_theta_final": mallow_theta,
        "beta_final": beta,
        "U0_final": U0,
        "U_a_final": U_a_dict,
        "H_final": h_U,
        "update_df": update_df,
        # Global Timing Outputs (cumulative)
        "total_prior_timing": total_prior_time,
        "total_likelihood_timing": total_likelihood_time,
        "update_type_timing": update_type_timing,
        # Per-Iteration Timing Lists
        "iteration_list": iteration_list,
        "update_category_list": update_category_list,
        "prior_timing_list": prior_timing_list,
        "likelihood_timing_list": likelihood_timing_list,
        "update_timing_list": update_timing_list
    }
    return result_dict
