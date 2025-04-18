import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
import seaborn as sns
import math
from scipy.integrate import quad
import random
import matplotlib.pyplot as plt
from scipy.stats import beta, expon, norm, uniform, kstest

# Import from the correct paths
from src.mcmc.hpo_po_hm_mcmc import mcmc_simulation_hpo
from src.mcmc.hpo_po_hm_mcmc_k import mcmc_simulation_hpo_k
from src.utils.po_fun import BasicUtils, StatisticalUtils, GenerationUtils

# Path to configuration file and output folder
YAML_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hpo_inference", "config", "hpo_mcmc_configuration.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "hpo_test_output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Example input data
M0 = [0, 1, 2, 3, 4, 5]
assessors = [1, 2, 3]
M_a_dict = {
    1: [0, 2, 4],
    2: [1, 2, 3, 4, 5],
    3: [1, 2, 3, 4, 5]
}
O_a_i_dict = {
    1: [[0, 2, 4], [0, 2]],
    2: [[1, 2, 5], [1, 3], [1, 4, 5]],
    3: [[1, 3], [1, 3, 4], [2, 4, 5]]
}
observed_orders = None 
alpha = np.array([0.5, -0.2, 0.3, 0.1, 0.0, 1.2]) 

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
config = load_config(YAML_FILE)

# Extract configuration parameters
num_iterations = config["mcmc"]["num_iterations_debug"]
K = config["mcmc"]["K"]
dr = config["rho"]["dr"]
drrt = config["rhotau"]["drrt"]
noise_option = config["noise"]["noise_option"]
sigma_mallow = config["noise"]["sigma_mallow"]

prior_config = config["prior"]
rho_prior = prior_config["rho_prior"]
noise_beta_prior = prior_config["noise_beta_prior"]
mallow_ua = prior_config["mallow_ua"] 
K_prior = prior_config["k_prior"]   

rho_tau_update = config["reversible_two_factors"]["rho_tau_update"] 

def check_log_likelihood(results: dict) -> float:
    """
    Check that the sum of the current and proposed log likelihoods is close to 0.
    """
    llk_sum = np.sum(results["log_likelihood_currents"] + results["log_likelihood_primes"])
    print("Sum of log likelihood currents and proposed values:", llk_sum)
    if not np.isclose(llk_sum, 0.0, atol=1e-6):
        print("WARNING: The sum of log likelihood values is not 0!")
    else:
        print("Log likelihood values sum to 0 as expected.")
    return llk_sum

def check_param(samples, label, dist, dist_params, output_filename, num_bins=300, tol=1e-4):
    """
    General function to check MCMC chain diagnostics against a theoretical distribution.

    - For 'rho', we assume a truncated Beta on [0, 1-tol].
    - For other *continuous* parameters, we plot a histogram and compare to dist.pdf.
    - For 'K' or a truncated Poisson distribution, we plot a discrete bar chart.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples (1D).
    label : str
        Name of the parameter (e.g., 'rho', 'P', 'theta', 'U_entry', 'K').
    dist : object
        A distribution object. Could be a scipy.stats distribution (beta, expon, etc.)
        or our custom TruncatedPoissonDist.
    dist_params : tuple
        Parameters for the distribution (e.g., (1, beta_param) for Beta(1, beta_param)).
    output_filename : str
        File name to save the plot.
    num_bins : int
        Number of bins for histogram (if continuous).
    tol : float
        Tolerance for 'rho' truncation.
    """

    samples = np.array(samples)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    full_path = os.path.join(OUTPUT_DIR, output_filename)

    # Case 1: K or a discrete distribution (TruncatedPoisson).
    if label.lower() == "k" or dist.__class__.__name__ == "TruncatedPoissonDist":
        print(f"--- {label} Diagnostics (Discrete) ---")

        unique_ks, counts = np.unique(samples, return_counts=True)
        total_count = len(samples)
        empirical_probs = counts / total_count

        # Build a dictionary from integer k -> probability
        empirical_dict = {kval: prob for kval, prob in zip(unique_ks, empirical_probs)}

        k_min = int(np.min(unique_ks))
        k_max = int(np.max(unique_ks))
        k_min = max(k_min, 1)  # ensure at least 1

        k_values = np.arange(k_min, k_max + 1)

        # Theoretical PMF
        pmf_vals = [dist.pdf(kval) for kval in k_values]

        # Optional: If the distribution object has mean()/var(), print them
        try:
            theoretical_mean = dist.mean()
            theoretical_var = dist.var()
        except AttributeError:
            theoretical_mean, theoretical_var = None, None

        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        print(f"Sample Mean: {sample_mean:.4f}, Sample Var: {sample_var:.4f}")
        if theoretical_mean is not None:
            print(f"Theoretical Mean: {theoretical_mean:.4f}")
        if theoretical_var is not None:
            print(f"Theoretical Var: {theoretical_var:.4f}")

        # Plot a bar chart of empirical probabilities
        heights = [empirical_dict.get(x, 0.0) for x in k_values]

        plt.figure(figsize=(8, 5))
        plt.bar(k_values, heights, alpha=0.5, label="Samples", edgecolor='black', color='skyblue')
        # Plot theoretical PMF
        plt.plot(k_values, pmf_vals, 'ro-', lw=2, label="Theoretical PMF")

        plt.xlabel(label)
        plt.ylabel("Probability")
        plt.title(f"Discrete Distribution Check: {label}")
        plt.legend()
        plt.savefig(full_path)
        plt.show()


    else:
        # Case 2: Continuous distribution

        # For 'rho', we handle a truncated Beta in [0, 1 - tol].
        if label.lower() == "rho":
            a, b = dist_params
            truncation_point = 1 - tol
            norm_const = dist.cdf(truncation_point, a, b)

            # Compute truncated PDF
            def truncated_pdf(x):
                return dist.pdf(x, a, b) / norm_const

            # Theoretical mean and variance for truncated Beta
            theoretical_mean, _ = quad(lambda x: x * truncated_pdf(x), 0, truncation_point)
            theoretical_var, _ = quad(lambda x: (x - theoretical_mean) ** 2 * truncated_pdf(x),
                                      0, truncation_point)

            # Prepare range for plotting PDF
            x_vals = np.linspace(0, truncation_point, 1000)
            pdf_vals = [truncated_pdf(x) for x in x_vals]

            # Print numeric summary
            print(f"--- {label} Diagnostics (Truncated Beta) ---")
            print(f"Theoretical vs Sample Mean: {theoretical_mean:.4f} | {sample_mean:.4f}")
            print(f"Theoretical vs Sample Var:  {theoretical_var:.4f} | {sample_var:.4f}")

            # KS test with a truncated CDF
            def ks_cdf(x):
                return dist.cdf(x, a, b) / norm_const

            ks_stat, p_value = kstest(samples, ks_cdf)
            print(f"KS Stat: {ks_stat:.3f}, p-value: {p_value:.3f}")

            # Plot
            bin_edges = np.linspace(0, truncation_point, num_bins + 1)

            # Basic histogram + PDF
            plt.figure(figsize=(8, 5))
            plt.hist(samples, bins=bin_edges, density=True, alpha=0.5, 
                     label="Samples", edgecolor='black')
            plt.plot(x_vals, pdf_vals, 'r-', lw=2, label="Truncated Beta PDF")
            plt.xlabel(label)
            plt.ylabel("Density")
            plt.title(f"{label} Distribution Check (Truncated Beta)")
            plt.legend()
            plt.savefig(output_filename)
            plt.show()

            # Enhanced subplots for trace, etc. (similar to your original)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(samples, color='#1f77b4', lw=1, alpha=0.8)
            ax1.set_title(f"Trace Plot: {label}", fontsize=12)
            ax1.set_xlabel("Iteration", fontsize=10)
            ax1.set_ylabel(label, fontsize=10)
            ax1.set_ylim(0, 1 - tol)
            ax1.grid(True, alpha=0.3)

            hist = sns.histplot(
                samples,
                bins=bin_edges,
                kde=True,
                ax=ax2,
                color='#2ca02c',
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7,
                stat='density'
            )
            ax2.plot(x_vals, pdf_vals, 'r-', lw=2, label='Truncated PDF')
            ax2.set_xlim(0, truncation_point)

            ax2.axvline(theoretical_mean, color='purple', linestyle='--', 
                        label=f'Theoretical Mean: {theoretical_mean:.3f}')
            ax2.axvline(sample_mean, color='orange', linestyle='--', 
                        label=f'Sample Mean: {sample_mean:.3f}')

            ax2.legend(loc='upper left')
            ax2.set_title(f"Distribution: {label}", fontsize=12)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            subplot_filename = full_path.replace(".png", "_subplots.png")
            plt.savefig(subplot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved enhanced subplots to {subplot_filename}")

        else:
            # For other continuous parameters (P, theta, U_entry, etc.)
            print(f"--- {label} Diagnostics (Continuous) ---")

            # Theoretical mean/var
            theoretical_mean = dist.mean(*dist_params)
            theoretical_var = dist.var(*dist_params)
            print(f"Theoretical vs Sample Mean: {theoretical_mean:.4f} | {sample_mean:.4f}")
            print(f"Theoretical vs Sample Var:  {theoretical_var:.4f} | {sample_var:.4f}")

            # KS test with full cdf
            def cdf_func(x):
                return dist.cdf(x, *dist_params)
            ks_stat, p_value = kstest(samples, cdf_func)
            print(f"KS Stat: {ks_stat:.3f}, p-value: {p_value:.3f}")

            # Plot histogram vs. PDF
            x_vals = np.linspace(np.min(samples), np.max(samples), 1000)
            pdf_vals = dist.pdf(x_vals, *dist_params)

            plt.figure(figsize=(8, 5))
            plt.hist(samples, bins=num_bins, density=True, alpha=0.5,
                     label="Samples", edgecolor='black')
            plt.plot(x_vals, pdf_vals, 'r-', lw=2, label=f"{dist.name} PDF")
            plt.xlabel(label)
            plt.ylabel("Density")
            plt.title(f"{label} Distribution Check")
            plt.legend()
            plt.savefig(full_path)
            plt.show()

def check_rho():
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,          
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[1, 0, 0, 0, 0, 0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=42
    )
    
    rho_samples = np.array(results["rho_trace"])
    check_log_likelihood(results)
    check_param(rho_samples, "rho", beta, (1, rho_prior), "hpo_rho_hist.png")

def check_tau():
    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 1, 0, 0, 0, 0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )
    
    tau_samples = np.array(results["tau_trace"])
    check_log_likelihood(results)
    check_param(tau_samples, "tau", uniform, (0, 1), "hpo_tau_hist.png")


def check_rho_tau():    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0, 1, 0, 0, 0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=True,
        random_seed=123
    )
    
    tau_samples = np.array(results["tau_trace"])
    rho_samples = np.array(results["rho_trace"]) 
    check_log_likelihood(results)
    check_param(tau_samples, "tau", uniform, (0, 1), "hpo_tau_hist_drrt.png")
    check_param(rho_samples, "rho", beta, (1, rho_prior), "hpo_rho_hist_drrt.png")



def check_P():
    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0, 0, 1, 0, 0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )
    
    p_samples = np.array(results["prob_noise_trace"])
    check_log_likelihood(results)
    print(np.mean(results['acceptance_rates']))
    check_param(p_samples, "P", beta, (1, noise_beta_prior), "hpo_p_hist.png")

def check_theta():    
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,            
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0, 0, 1, 0, 0],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )
    
    theta_samples = np.array(results["mallow_theta_trace"])
    check_log_likelihood(results)
    check_param(theta_samples, "theta", expon, (0, 1/mallow_ua), "hpo_theta_hist.png")
def check_U0():
    """
    Test U0 when tau=0.
    Run MCMC with only U0 updates, print diagnostics, and save marginal histograms
    overlaid with a standard normal density curve to 'u0_marginals_tau0.png'.
    """
    # 1) Run MCMC with only U0 updates
    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0, 0, 0, 1, 0],  # Only update U0
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=123
    )

    # 2) Convert to NumPy array
    U0_trace_arr = np.array(results["U0_trace"])  # shape: (n_iter, n_global, K_val)
    n_iter, n_global, K_val = U0_trace_arr.shape

    # 3) Debug output
    tau_value = results["tau_trace"][-1]  # Last tau value
    print(f"tau_value: {results['tau_trace'][-1]}")
    print(f"Ua (last iteration): {results['Ua_trace'][-1]}")
    print(f"U0 (last iteration): {results['U0_trace'][-1]}")
    print(f"U0_init (first iteration): {results['U0_trace'][0]}")

    # 4) Per-item coordinate means
    print("\n--- Per-item Overall Means ---")
    for i in range(n_global):
        mean_i = U0_trace_arr[:, i, :].mean()
        print(f"Item {i} mean: {mean_i:.4f}")
    print(f"Overall mean: {U0_trace_arr.mean():.4f}")

    # 5) Cross-item correlation by dimension
    print("\n--- Cross-item Correlation by Dimension (Expected ~ 0) ---")
    for k in range(K_val):
        all_pairs = []
        for sample in U0_trace_arr:
            col_k = sample[:, k]
            for n1 in range(len(col_k)):
                for n2 in range(n1 + 1, len(col_k)):
                    all_pairs.append((col_k[n1], col_k[n2]))
        if all_pairs:
            u1, u2 = zip(*all_pairs)
            correlation_col = np.corrcoef(u1, u2)[0, 1]
            print(f"Dimension {k}: Correlation = {correlation_col:.4f}")

    # 6) Intra-item coordinate correlation
    print("\n--- Intra-item Pairwise Coordinate Correlation ---")
    all_dim_pairs = []
    for sample in U0_trace_arr:
        for row in sample:
            for k1 in range(K_val):
                for k2 in range(k1 + 1, K_val):
                    all_dim_pairs.append((row[k1], row[k2]))
    if all_dim_pairs:
        d1, d2 = zip(*all_dim_pairs)
        correlation = np.corrcoef(d1, d2)[0, 1]
        rho_true = results["rho_trace"][-1]
        print(f"Sampled intra-item correlation: {correlation:.4f}")
        print(f"rho_true (last): {rho_true:.4f}")
        print(f"Difference: {correlation - rho_true:.4f}")

    # 7) Plot marginal distributions and overlay N(0,1)
    plt.figure(figsize=(7, 5))
    sns.set_style("whitegrid")
    x_vals = np.linspace(-4, 4, 300)
    pdf_vals = norm.pdf(x_vals, loc=0, scale=1)

    for k in range(K_val):
        data_k = U0_trace_arr[:, :, k].ravel()
        plt.hist(data_k, bins=50, alpha=0.5, density=True, label=f"dim {k}")

    plt.title(f"Marginal Distributions of U0 Components (τ = {tau_value:.2f})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()



    # Save the plot
    filename = os.path.join(OUTPUT_DIR,
                                f"u0_marginals_tau0.png")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    print(f"\nHistogram plot saved to: {filename}")

def check_Ua():
    """
    Runs MCMC updating only Ua (assessor-level latent coordinates).
    Then checks that U_a[j,k] - tau * U0[j,k] matches the hierarchical
    model's assumption that it is ~ N(0, (1 - tau^2) * Sigma_rho[k,k]).

    Specifically, we:
      1) Extract chain of Ua, U0, tau.
      2) For each iteration it, each assessor a, each item j, each dimension k,
         compute d = Ua[a][j,k] - tau[it]*U0[it][j,k].
      3) Check the distribution of d vs. N(0, (1 - tau^2)*Var(U0[j,k])) or
         something similar, if you assume Sigma_rho = rho * I_K.
    """
    # Only update Ua
    mcmc_pt = [0, 0, 0, 0, 0, 1]

    results = mcmc_simulation_hpo(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,
        K=K,
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=mcmc_pt,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        rho_tau_update=rho_tau_update,
        random_seed=42
    )

    # 2) Extract necessary traces
    Ua_trace  = results["Ua_trace"]  # length = num_iterations
    U0_trace  = results["U0_trace"]  # length = num_iterations
    tau_trace = np.array(results["tau_trace"])
    rho_trace = np.array(results["rho_trace"])

    n_iter = len(Ua_trace)
    tau_mean_sq = tau_trace[-1]**2
    tau_mean    = tau_trace[-1]
    rho_mean    = rho_trace[-1]

    # 4) Prepare a data structure to hold differences for each assessor/dimension
    #    diffs[a][k] will be a list of differences over all items j and iterations
    diffs = {
        a: {kdim: [] for kdim in range(K)} 
        for a in assessors
    }

    # 5) Collect differences across all iterations, items, dimensions
    for it in range(n_iter):
        Ua_dict_it = Ua_trace[it]  # assessor -> (|M_a|, K)
        U0_it      = U0_trace[it]  # shape: (|M0|, K)
     
        for a in assessors:
            if a not in M_a_dict or a not in Ua_dict_it:
                continue
            items_a = M_a_dict[a]
            Ua_mat = Ua_dict_it[a]
            for idx, item_global_id in enumerate(items_a):
                ua_val = Ua_mat[idx, :]            # shape (K,)
                u0_val = U0_it[item_global_id, :]  # shape (K,)
                d_vec = ua_val - tau_mean  * u0_val
                for kdim in range(K):
                    diffs[a][kdim].append(d_vec[kdim])

    # 6) For each assessor, produce a figure with K subplots
    for a in assessors:
        if a not in diffs:
            continue
        # Create figure
        fig, axes = plt.subplots(nrows=1, ncols=K, figsize=(5*K, 4), sharey=False)
        if K == 1:
            # If there's only one dimension, axes is not a list
            axes = [axes]

        fig.suptitle(f"Assessor {a}: Differences U_a - tau U_0", fontsize=14)

        for kdim in range(K):
            ax = axes[kdim]
            data_k = np.array(diffs[a][kdim])
            if data_k.size == 0:
                ax.set_title(f"Dim {kdim} (No data)")
                continue

            # Theoretical variance from the aggregated approach:
            var_theory = (1 - tau_mean_sq) * 1
            std_theory = math.sqrt(var_theory)

            # Sample stats
            sample_mean = np.mean(data_k)
            sample_var  = np.var(data_k)

            # Plot histogram
            sns.histplot(data_k, bins=50, stat="density", kde=True, ax=ax, color="skyblue", alpha=0.5)

            # Overlay theoretical normal (mean=0, std=std_theory)
            x_vals = np.linspace(data_k.min(), data_k.max(), 200)
            pdf_vals = norm.pdf(x_vals, loc=0, scale=std_theory)
            ax.plot(x_vals, pdf_vals, "r--", lw=2, label=f"N(0, {std_theory:.2f}^2)")


            # Title with stats
            ax.set_title(
                f"Dim {kdim}: mean={sample_mean:.3f}\n"
                f"sample_var={sample_var:.3f}, theor_var={var_theory:.3f}\n"
            )
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        # Save or show the figure
        outname = f"check_Ua_assessor_{a}.png"
        outpath = os.path.join(OUTPUT_DIR, outname)
        plt.savefig(outpath, dpi=150)
        print(f"Saved figure for assessor {a}: {outpath}")
        plt.close(fig)

    print("Done checking (Ua - tau U0) against Normal(0, (1 - tau^2)*rho).")


def check_K():
    results = mcmc_simulation_hpo_k(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        alpha=alpha,      
        dr=dr,
        drrt=drrt,
        sigma_mallow=sigma_mallow,
        noise_option=noise_option,
        mcmc_pt=[0, 0, 0, 0, 0, 0,1],
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        mallow_ua=mallow_ua,
        K_prior=K_prior,
        rho_tau_update=rho_tau_update,
        random_seed=42
    )
    
    K_samples = np.array(results["K_trace"])
    # Create truncated Poisson object
    truncated_poisson = StatisticalUtils.TruncatedPoisson(K_prior)
    # Now do a discrete bar chart comparison
    check_param(K_samples , "K", truncated_poisson, (), "hpo_k_hist.png")



# if __name__ == "__main__":
#     param = sys.argv[1].lower()
#     if param == "rho":
#         check_rho()
#     elif param == "p":
#         check_P()
#     elif param == "theta":
#         check_theta()
#     elif param == "tau":
#         check_tau()
#     elif param == "u0":
#         check_U0()
#     elif param == "ua":
#         check_Ua()
#     elif param == "rhotau":
#         check_rho_tau()
#     else:
#         print("Unknown parameter. Please choose one of: rho, p, theta, tau, u0, ua, rhotau.")

check_K()