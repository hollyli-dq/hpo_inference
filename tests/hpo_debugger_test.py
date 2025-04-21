#!/usr/bin/env python3
"""
extreme_tau_demo.py

Showcases two extreme hierarchical‑VSP scenarios:
  • **tau = 0**  ⟶  assessor‑specific embeddings are i.i.d  N(0, Σ_ρ)
  • **tau = 1**  ⟶  assessor embeddings equal the global embedding

For each case we run a **fixed‑tau Metropolis sampler** that updates
ρ, queue‑jump noise *p*, and latent matrices U₀ / Uₐ.
The script saves diagnostic histograms and prints final states.
"""

# ‑‑‑- Standard library ------------------------------------------------------
from __future__ import annotations
import os, sys, math, random, copy, pathlib
from pathlib import Path
from typing import Dict, List, Any

# ‑‑‑- Third‑party -----------------------------------------------------------
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ‑‑‑- Local project utilities ----------------------------------------------
#  project_root/  (add to path once)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.hpo_po_hm_mcmc_fixed_tau import mcmc_simulation_hpo_fixed_tau
from src.utils.po_fun import BasicUtils, StatisticalUtils

# ---------------------------------------------------------------------------
# >>> Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "config" / "hpo_mcmc_configuration.yaml"
with CONFIG_PATH.open() as fh:
    CFG = yaml.safe_load(fh)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "hpo_test_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# >>> Toy data set
# ---------------------------------------------------------------------------
N_ITEMS     = 5
ASSESSORS   = [101, 202]
ALPHA       = np.full(N_ITEMS + 1, 0.5)
M_A_DICT    = {101: [0, 1, 2], 202: [2, 3, 4]}
M0_GLOBAL   = list(range(N_ITEMS))
ORDERS      = {101: [[0, 1], [1, 2]], 202: [[2, 3], [3, 4]]}
X                = np.random.randn(N_ITEMS, K)
# ---------------------------------------------------------------------------
# >>> MCMC hyper‑parameters (mini demo)
# ---------------------------------------------------------------------------
N_ITR            = 50000
RHO_HYPER        = 1/6        # mode at ~0 (weak corr)
NOISE_BETA_PRIOR = 1.0        # Beta(1,β) prior on queue‑jump p
DR_STEP          = 1.1        # multiplicative proposal for ρ
UPDATE_PCTS      = [0.2, 0.2, 0.3, 0.2,0.1]  # [ρ, p, U₀, Uₐ]
DRBETA           = 0.01
K                = 3
SIGMA_BETA       = 0.1

# ---------------------------------------------------------------------------
# >>> Helper: run one scenario and plot a histogram
# ---------------------------------------------------------------------------

def run_and_plot(tau_value: float, seed: int) -> None:
    """Run the fixed‑tau sampler and save histogram diagnostics."""

    res = mcmc_simulation_hpo_fixed_tau(
        num_iterations   = N_ITR,
        M0               = M0_GLOBAL,
        assessors        = ASSESSORS,
        M_a_dict         = M_A_DICT,
        O_a_i_dict       = ORDERS,
        observed_orders  = ORDERS,
        sigma_beta       = SIGMA_BETA,
        X                = X,
        K                = K,
        dr               = DR_STEP,
        drbeta           = DRBETA,
        mcmc_pt          = UPDATE_PCTS,
        rho_prior        = RHO_HYPER,
        noise_beta_prior = NOISE_BETA_PRIOR,
        tau_value        = tau_value,
        random_seed      = seed,
    )

    print(f"\n=== τ = {tau_value} run complete ===")
    print(f"ρ_final         : {res['rho_final']:.3f}")
    print(f"prob_noise_final: {res['prob_noise_final']:.3f}\n")

    # plot first assessor, first dim as illustration
    assessor_id = ASSESSORS[0]
    dim_idx     = 0
    samples = [ua_dict[assessor_id][0, dim_idx]
               for ua_dict in res['Ua_trace']]

    rho_final = res['rho_final']
    sigma     = math.sqrt(BasicUtils.build_Sigma_rho(K, rho_final)[dim_idx, dim_idx])

    plt.figure(figsize=(6, 4))
    plt.hist(samples, bins=40, density=True, alpha=0.6,
             label=r"$U_{a}^{(k)}$ trace")
    x = np.linspace(min(samples), max(samples), 200)
    plt.plot(x, norm.pdf(x, 0, sigma), 'r-',
             label=fr"$\mathcal{{N}}(0, {sigma**2:.2f})$")
    plt.title(fr"τ = {tau_value}  (assessor {assessor_id}, dim {dim_idx})")
    plt.legend()
    out = OUTPUT_DIR / f"hist_tau{tau_value}_{assessor_id}_d{dim_idx}.pdf"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# >>> Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_and_plot(tau_value=0.0, seed=111)  # independent local embeddings
    run_and_plot(tau_value=1.0, seed=222)  # identical local == global
