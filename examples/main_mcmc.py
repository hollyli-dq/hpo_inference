from __future__ import annotations

import argparse
import sys
import time
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from io import StringIO
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ────────────────────────────────────────────────────────────────────────────
# Project paths
# ────────────────────────────────────────────────────────────────────────────
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive / ipython
    # Fallback for interactive use (adjust if your CWD isn't the script dir)
    PROJECT_ROOT = Path.cwd().parent  # Assumes script is in a subdir like 'scripts' or 'notebooks'

SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure source directory is in Python path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:  # Also add src dir if imports are relative to it
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Project‑specific imports (samplers and utilities)
# ---------------------------------------------------------------------------
try:
    # Import MCMC modules
    from mcmc.hpo_po_hm_mcmc import mcmc_simulation_hpo
    from mcmc.hpo_po_hm_mcmc_k import mcmc_simulation_hpo_k
    
    # Import analysis and plotting utilities
    from src.utils.po_fun_plot import PO_plot
    from src.utils.po_fun import BasicUtils, StatisticalUtils, GenerationUtils
except ImportError as err:
    sys.exit(
        f"ERROR importing required modules – check PYTHONPATH or project structure.\n"
        f"PROJECT_ROOT='{PROJECT_ROOT}', sys.path includes: '{SRC_DIR}'\n"
        f"Original error: {err}"
    )

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_FILE = "hpo_mcmc_configuration.yaml"

def load_config(path: Path) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not path.is_file():
        sys.exit(f"Config file not found: {path}")
    try:
        with path.open('r') as fh:  # Specify read mode explicitly
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as err:
        sys.exit(f"YAML error parsing config file '{path}': {err}")
    except Exception as err:  # Catch other potential file reading errors
        sys.exit(f"Error reading config file '{path}': {err}")

    if cfg is None:  # Handle empty YAML file case
        print(f"WARNING: Config file '{path}' is empty or invalid, using defaults.")
        return {}
    print(f"Loaded config → {path}")
    return cfg

# ---------------------------------------------------------------------------
# CSV data loaders and helpers
# ---------------------------------------------------------------------------

def load_data_from_files(
    base_dir: Path,
    item_chars_file: str = "item_characteristics.csv",
    rankings_file: str = "observed_rankings.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads item characteristics and observed rankings CSVs,
    derives assessor and selective set information.
    
    Uses a structured CSV that may contain:
    - Items (up to the 'assessor_id' marker)
    - Assessors (between 'assessor_id' and 'selective_set_id' markers)
    - Selective sets (after 'selective_set_id' marker)
    """
    item_path = base_dir / item_chars_file
    rank_path = base_dir / rankings_file

    print(f"Loading data from: {item_path}")
    
    # Read the CSV file containing the structured data
    items_df1 = pd.read_csv(item_path, comment='#', skip_blank_lines=False)
    assessors_df = pd.DataFrame()
    selective_sets_df = pd.DataFrame()
    items_df = pd.DataFrame()
    
    # Find the marker rows
    cutoff_idx = items_df1.loc[items_df1['item_id'] == 'assessor_id'].index
    if len(cutoff_idx) > 0:
        # Keep only rows before assessor_id appears
        items_df = items_df1.iloc[:cutoff_idx[0]]
        
        # Find the selective_set_id marker
        cutoff_idx2 = items_df1.loc[items_df1['item_id'] == 'selective_set_id'].index
        if len(cutoff_idx2) > 0:
            # → slice BETWEEN the two sentinel rows
            assessors_df = items_df1.iloc[cutoff_idx[0]:cutoff_idx2[0]].copy()
            selective_sets_df = items_df1.iloc[cutoff_idx2[0]:].copy()
    else:
        # If no markers found, treat the entire file as items
        items_df = items_df1.copy()

    # Process assessors: treat first row as header
    if not assessors_df.empty:
        assessors_df.columns = assessors_df.iloc[0]  # promote row-0
        assessors_df = assessors_df.iloc[1:].reset_index(drop=True)
        # Drop columns that are entirely NaN
        assessors_df = assessors_df.dropna(axis=1, how='all')

    # Process selective sets: treat first row as header
    if not selective_sets_df.empty:
        selective_sets_df.columns = selective_sets_df.iloc[0]  # promote row-0
        selective_sets_df = selective_sets_df.iloc[1:].reset_index(drop=True)
        # Drop columns that are entirely NaN
        selective_sets_df = selective_sets_df.dropna(axis=1, how='all')

    # Clean up any remaining NaN rows
    items_df = items_df.dropna(how='all')
    assessors_df = assessors_df.dropna(how='all')
    selective_sets_df = selective_sets_df.dropna(how='all')

    # Convert item_id to numeric if possible
    items_df['item_id'] = pd.to_numeric(items_df['item_id'], errors='coerce')
    items_df = items_df.dropna(subset=['item_id'])

    # Load rankings data
    try:
        rankings_df = pd.read_csv(rank_path, comment='#', skip_blank_lines=True)
        rankings_df.dropna(how='all', inplace=True)
        print(f"Rankings loaded: {len(rankings_df)} observations")
    except FileNotFoundError:
        print(f"WARNING: Rankings file not found at {rank_path}. Creating empty DataFrame.")
        rankings_df = pd.DataFrame(columns=['assessor_id', 'observed_ranking', 'choice_set'])
    except Exception as e:
        print(f"ERROR loading rankings: {e}")
        rankings_df = pd.DataFrame(columns=['assessor_id', 'observed_ranking', 'choice_set'])

    print(f"Loaded {len(items_df)} items, {len(assessors_df)} assessors, {len(rankings_df)} ranking observations.")
    return items_df, assessors_df, selective_sets_df, rankings_df

# Helper function for parsing item lists
_SPLIT_RE = re.compile(r"[;,]")  # comma *or* semicolon

def _parse_ids(s: str) -> List[int]:
    """Split '4,7,8' or '4;7;8' → [4, 7, 8] (keeps nothing empty)."""
    return [int(tok) for tok in _SPLIT_RE.split(str(s)) if tok.strip().isdigit()]

def _coerce_items_col(raw_items: Any) -> List[int]:
    """Coerces various input types representing item lists into a list of ints."""
    if isinstance(raw_items, list):
        try:
            return [int(x) for x in raw_items]
        except (ValueError, TypeError) as e:
            print(f"WARNING: Could not convert all elements in list {raw_items} to int: {e}. Returning empty list.", file=sys.stderr)
            return []
    if isinstance(raw_items, str):
        # Try splitting by common delimiters (comma or semicolon)
        # Prioritize comma, then semicolon if comma yields only one element
        delimiters = [',', ';']
        parsed = []
        for delim in delimiters:
            tokens = [tok.strip() for tok in raw_items.split(delim) if tok.strip()]
            if len(tokens) > 1 or delim == delimiters[-1]:  # Use this delimiter if it splits or it's the last option
                try:
                    parsed = [int(tok) for tok in tokens]
                    return parsed
                except ValueError as e:
                    print(f"WARNING: Could not parse token in string '{raw_items}' (delim='{delim}') as integer: {e}. Trying next delimiter or returning empty.", file=sys.stderr)
                    # Continue to next delimiter if parsing failed

        # If loop finishes without successful parsing
        print(f"WARNING: Could not parse string '{raw_items}' into list of ints. Returning empty list.", file=sys.stderr)
        return []

    if pd.isna(raw_items):
        return []

    # Handle single numeric value
    if isinstance(raw_items, (int, float)):
        try:
            return [int(raw_items)]
        except (ValueError, TypeError):
            print(f"WARNING: Could not convert numeric value {raw_items} to int. Returning empty list.", file=sys.stderr)
            return []

    print(f"WARNING: Unhandled type '{type(raw_items)}' for item list coercion: {raw_items}. Returning empty list.", file=sys.stderr)
    return []

# ---------------------------------------------------------------------------
# Data preparation for MCMC
# ---------------------------------------------------------------------------

def prepare_mcmc_input_data(
    items_df: pd.DataFrame,
    assessors_df: pd.DataFrame,
    selective_sets_df: pd.DataFrame, 
    rankings_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Turns the four CSV-derived frames into the dicts the MCMC code needs."""

    # ------------------------------------------------------------------ M0
    items = items_df["item_id"].astype(int).unique().tolist()

    # ------------------------------------------------------------------ X
    cov_cols = sorted(c for c in items_df.columns if c.startswith("covariate_"))
    X = items_df[cov_cols].apply(pd.to_numeric, errors="coerce").T.values  # (features × items)

    # ------------------------------------------------------------------ assessor list
    assessors = assessors_df["assessor_id"].astype(int).unique().tolist()

    # ------------------------------------------------------------------ dictionaries
    M_a_set = defaultdict(set)     # will be turned into list later
    O_a_i_dict = defaultdict(list)    # assessor → [choice-list, …]
    observed_orders = defaultdict(list)    # assessor → {ranking_id: ranking-list}

    for _, row in rankings_df.iterrows():
        aid = int(row["assessor_id"])

        # selective set for *this* task
        choice_items = _parse_ids(row["choice_set"])
        M_a_set[aid].update(choice_items)
        O_a_i_dict[aid].append(choice_items)

        # observed ranking for this task
        ranking_items = _parse_ids(row["observed_ranking"])
        observed_orders[aid].append(ranking_items)

    # finalise M_a_dict: assessor → sorted unique list
    M_a_dict = {aid: sorted(items) for aid, items in M_a_set.items()}

    print(
        f"Prepared data: |M0|={len(items)} items, |A|={len(assessors)} assessors, "
        f"Total choice-sets={sum(len(v) for v in O_a_i_dict.values())}"
    )

    return dict(
        M0=items,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        X=X,
    )

# ---------------------------------------------------------------------------
# MCMC simulation runner
# ---------------------------------------------------------------------------

def run_mcmc(data: Dict[str, Any], cfg: Dict[str, Any], rjmcmc: bool = False) -> Dict[str, Any] | None:
    """Runs the appropriate MCMC simulation with parameters from config."""

    mcmc_cfg = cfg.get("mcmc", {})
    noise_cfg = cfg.get("noise", {})
    prior_cfg = cfg.get("prior", {})
    rho_cfg = cfg.get("rho", {})  # Added for rho params
    rhotau_cfg = cfg.get("rhotau", {})  # Added for rhotau params
    beta_cfg = cfg.get("beta", {})  # Added for beta params

    # --- Parameter Extraction with Defaults --------------------------------
    num_iter = mcmc_cfg.get("num_iterations", 10000)
    if not isinstance(num_iter, int) or num_iter <= 0:
        print(
            f"WARNING: Invalid num_iterations ({num_iter}). Using default 10000.",
            file=sys.stderr,
        )
        num_iter = 10000

    # Use provided data structures directly
    common_args = dict(
        num_iterations=num_iter,
        M0=data.get("M0"),
        assessors=data.get("assessors"),
        M_a_dict=data.get("M_a_dict"),
        O_a_i_dict=data.get("O_a_i_dict"),
        observed_orders=data.get("observed_orders"),
        X=data.get("X"),
        # Priors
        sigma_beta=prior_cfg.get("sigma_beta", 0.5),
        rho_prior=prior_cfg.get("rho_prior", [1.0, 1.0]),  # Ensure float
        noise_beta_prior=prior_cfg.get("noise_beta_prior", 10.0),  # Ensure float
        mallow_ua=prior_cfg.get("mallow_ua", [0.1, 0.9]),
        # Noise model
        sigma_mallow=noise_cfg.get("sigma_mallow", 0.1),
        noise_option=noise_cfg.get("noise_option", "queue_jump"),
        # Step sizes / tuning params
        dr=rho_cfg.get("dr", 1.1),
        drrt=rhotau_cfg.get("drrt", 1.1),
        drbeta=beta_cfg.get("drbeta", 0.1),
        # Seed
        random_seed=cfg.get("random_seed", int(time.time() * 1000) % (2**32 - 1)),  # Ensure seed is valid
    )

    start_time = time.time()

    if rjmcmc:
        print(f"Running RJMCMC (Reversible Jump) for {num_iter} iterations…")
        # --- RJMCMC Specific Args -------------------------------------
        rjmcmc_pt_cfg = mcmc_cfg.get("update_probabilities", {})
        UPD_KEYS = [
            "rho",
            "tau",
            "rho_tau",
            "noise",
            "U_0",
            "U_a",
            "K",
            "beta",
        ]
        
        try:
            rjmcmc_pt = [float(rjmcmc_pt_cfg[k]) for k in UPD_KEYS]
        except (KeyError, TypeError) as e:
            print(f"WARNING: Error extracting update probabilities, using defaults: {e}")
            rjmcmc_pt = [0.1] * 8  # Default update probabilities
            
        k_prior = prior_cfg.get("k_prior", 3)  # Prior on number of clusters
        
        # Update relevant arguments
        common_args.update({
            "mcmc_pt": rjmcmc_pt,
            "K_prior": k_prior,
        })
        
        # Run the RJMCMC simulation
        results = mcmc_simulation_hpo_k(**common_args)
        
    else:
        print(f"Running Fixed-K MCMC for {num_iter} iterations…")
        # --- Fixed-K Specific Args -----------------------------------
        fixed_k_pt_cfg = mcmc_cfg.get("update_probabilities", {})
        UPD_KEYS = [
            "rho",
            "tau",
            "rho_tau",
            "noise",
            "U_0",
            "U_a",
            "beta",
        ]
        
        try:
            fixed_k_pt = [float(fixed_k_pt_cfg[k]) for k in UPD_KEYS]
        except (KeyError, TypeError) as e:
            print(f"WARNING: Error extracting update probabilities, using defaults: {e}")
            fixed_k_pt = [0.1] * 7  # Default update probabilities
            
        # Update relevant arguments
        common_args.update({
            "mcmc_pt": fixed_k_pt,
            "K": mcmc_cfg.get("K", 3),
        })
        
        # Run the fixed-K MCMC simulation
        results = mcmc_simulation_hpo(**common_args)

    end_time = time.time()
    print(f"MCMC execution time: {end_time - start_time:.2f}s")

    return results

# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------

def analyze_results(
    results: Dict[str, Any],
    data: Dict[str, Any],
    config: Dict[str, Any],
    burn_in: Optional[int] = None,
    use_reversible_jump: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """High‑level post‑processing / diagnostics for an MCMC run.

    Parameters
    ----------
    results
        Dictionary returned by the sampler.
    data
        Prepared data dict (same one passed *into* the sampler).
    config
        Global YAML / dict configuration.
    burn_in
        Number of initial iterations to discard (if None, use 20% of iterations)
    use_reversible_jump
        Whether this was an RJMCMC run (affects diagnostics for K).
    output_dir
        Directory to save results and plots
    """

    # ─────────────────────── sanity checks ────────────────────────────────
    if not results:
        print("ERROR: MCMC results dictionary is empty or None.")
        return

    # Find or set burn in period
    if burn_in is None:
        # Default burn-in: 20% of iterations
        log_like = results.get("log_likelihood_currents", [])
        burn_in = int(0.2 * len(log_like)) if log_like else 0
        print(f"Using default burn-in period: {burn_in} iterations")

    # Set up analysis parameters
    analysis_cfg = config.get("analysis", {})
    edge_threshold = float(analysis_cfg.get("edge_threshold", 0.5))
    top_n = int(analysis_cfg.get("top_partial_orders", 4))

    # Set up output directory
    out_dir_raw = output_dir or analysis_cfg.get("output_dir", "mcmc_output")
    out_dir = str(out_dir_raw or "mcmc_output")  # ensure not None/empty
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving analysis results to: {os.path.abspath(out_dir)}")

    print("\n--- Analyzing MCMC Results ---")

    # ─────────────────────── log‑likelihood trace ─────────────────────────
    log_like = results.get("log_likelihood_currents", [])
    
    # Plot log likelihood
    try:
        likelihood_plot_path = os.path.join(out_dir, "log_likelihood.pdf")
        # Check if the function supports output_filename parameter
        import inspect
        if 'output_filename' in inspect.signature(PO_plot.plot_log_likelihood).parameters:
            PO_plot.plot_log_likelihood(
                log_like, burn_in=burn_in, output_filename=likelihood_plot_path
            )
        else:
            # Alternative: Plot and save manually
            plt.figure(figsize=(10, 6))
            PO_plot.plot_log_likelihood(log_like, burn_in=burn_in)
            plt.savefig(likelihood_plot_path)
            plt.close()
        print(f"Log-likelihood plot saved to: {likelihood_plot_path}")
    except Exception as e:
        print(f"ERROR plotting log-likelihood: {e}")

    # ─────────────────────── thin traces ----------------------------------
    filtered_results = {
        k: (v[burn_in:] if isinstance(v, (list, np.ndarray)) else v) 
        for k, v in results.items()
    }

    # ─────────────────────── final partial‑order matrices -----------------
    h_trace = filtered_results.get("H_trace", [])
    if not h_trace:
        print("WARNING: H_trace not found – skipping partial‑order aggregation.")
        return

    assessors = list(h_trace[0].keys())
    threshold = edge_threshold

    final_H = {}

    for aid in assessors:
        # gather this assessor's matrices over iterations
        matrices = [h_it[aid] for h_it in h_trace if aid in h_it]
        if not matrices:
            continue

        # mean presence of each edge
        mean_matrix = np.mean(matrices, axis=0)

        # count unique partial orders (for top‑n display)
        sorted_counts = StatisticalUtils.count_unique_partial_orders(matrices)
        total_samples = sum(cnt for _, cnt in sorted_counts)

        print(f"\nAssessor {aid}: total PO samples = {total_samples}")

        top_partial_orders = sorted_counts[:top_n]
        percentages = [
            (mat, cnt, (cnt / total_samples) * 100) for mat, cnt in top_partial_orders
        ]
        
        # Plot top partial orders
        try:
            po_plot_path = os.path.join(out_dir, f"top_PO_a{aid}.pdf")
            # Check function signature for compatibility
            if 'output_filename' in inspect.signature(PO_plot.plot_top_partial_orders).parameters:
                PO_plot.plot_top_partial_orders(
                    percentages,
                    top_n=top_n,
                    item_labels=data.get("M0"),
                    output_filename=po_plot_path,
                )
            else:
                # Alternative: Plot and save manually
                plt.figure(figsize=(12, 8))
                PO_plot.plot_top_partial_orders(
                    percentages,
                    top_n=top_n,
                    item_labels=data.get("M0")
                )
                plt.savefig(po_plot_path)
                plt.close()
            print(f"Top partial orders plot for assessor {aid} saved to: {po_plot_path}")
        except Exception as e:
            print(f"ERROR plotting top partial orders for assessor {aid}: {e}")

        # binarise & transitive reduction
        binary_matrix = (mean_matrix >= threshold).astype(int)
        final_H[aid] = BasicUtils.transitive_reduction(binary_matrix)

    # attach to results for downstream use
    results["H_final"] = final_H

    # ─────────────────────── parameter diagnostics ------------------------
    try:
        inferred_vars_path = os.path.join(out_dir, "mcmc_inferred_variables.pdf")
        # Check if function supports direct output path
        if 'output_filename' in inspect.signature(PO_plot.plot_inferred_variables).parameters:
            PO_plot.plot_inferred_variables(
                filtered_results,
                {
                    "rho_true": config.get("rho_true"),
                    "prob_noise_true": config.get("prob_noise_true"),
                    "tau_true": config.get("tau_true"),
                    "beta_true": config.get("beta_true"),
                },
                config,
                burn_in=0,
                output_filename=inferred_vars_path,
                assessors=data.get("assessors"),
                M_a_dict=data.get("M_a_dict"),
            )
        else:
            # The function may handle saving internally
            PO_plot.plot_inferred_variables(
                filtered_results,
                {
                    "rho_true": config.get("rho_true"),
                    "prob_noise_true": config.get("prob_noise_true"),
                    "tau_true": config.get("tau_true"),
                    "beta_true": config.get("beta_true"),
                },
                config,
                burn_in=0,
                assessors=data.get("assessors"),
                M_a_dict=data.get("M_a_dict"),
            )
            # May need to adjust path if function saves with a different name
        print(f"Inferred variables plot saved to: {inferred_vars_path}")
    except Exception as e:
        print(f"ERROR plotting inferred variables: {e}")

    # optional extra plots --------------------------------------------------
    try:
        beta_params_path = os.path.join(out_dir, "beta_parameters.pdf")
        # Check if function supports direct output path
        if 'output_filename' in inspect.signature(PO_plot.plot_beta_parameters).parameters:
            PO_plot.plot_beta_parameters(
                filtered_results,  # results after burn‑in
                {
                    "beta_true": config.get("beta_true"),
                },
                config,
                burn_in=0,
                output_filename=beta_params_path,
            )
        else:
            # Manual save approach
            plt.figure(figsize=(10, 6))
            PO_plot.plot_beta_parameters(
                filtered_results,  # results after burn‑in
                {
                    "beta_true": config.get("beta_true"),
                },
                config,
                burn_in=0,
            )
            plt.savefig(beta_params_path)
            plt.close()
        print(f"Beta parameters plot saved to: {beta_params_path}")
    except Exception as e:
        print(f"ERROR plotting beta parameters: {e}")

    # latent‑space diagnostics (U₀ / U_a) -----------------------------------
    try:
        # Check if function supports output_dir parameter
        if 'output_dir' in inspect.signature(PO_plot.plot_u0_ua_diagnostics).parameters:
            PO_plot.plot_u0_ua_diagnostics(
                results=filtered_results,
                assessors=data.get("assessors"),
                M0=data.get("M0"),
                K=config.get("mcmc", {}).get("K", 3),
                output_dir=out_dir,
            )
        else:
            # Try standard approach - function may save files itself
            PO_plot.plot_u0_ua_diagnostics(
                results=filtered_results,
                assessors=data.get("assessors"),
                M0=data.get("M0"),
                K=config.get("mcmc", {}).get("K", 3),
            )
            # May need to move files if the function saves elsewhere
        print(f"U0/Ua diagnostics saved to {out_dir}")
    except Exception as e:
        print(f"ERROR plotting U0/Ua diagnostics: {e}")

    # global vs. inferred PO comparison ------------------------------------
    try:
        h_true_global = data.get("h_true_global")  # expected key
        if h_true_global is not None and 0 in final_H:
            global_po_path = os.path.join(out_dir, "global_PO_comparison.pdf")
            # Check if function supports output_filename parameter
            if 'output_filename' in inspect.signature(PO_plot.compare_and_visualize_global).parameters:
                PO_plot.compare_and_visualize_global(
                    h_true_global,
                    final_H[0],
                    data.get("index_to_item"),
                    [idx for idx in data.get("M0", [])],
                    do_transitive_reduction=True,
                    output_filename=global_po_path,
                )
            else:
                # Manual approach
                plt.figure(figsize=(12, 8))
                PO_plot.compare_and_visualize_global(
                    h_true_global,
                    final_H[0],
                    data.get("index_to_item"),
                    [idx for idx in data.get("M0", [])],
                    do_transitive_reduction=True,
                )
                plt.savefig(global_po_path)
                plt.close()
            print(f"Global PO comparison saved to: {global_po_path}")
    except Exception as e:
        print(f"ERROR comparing global PO: {e}")

    # per‑assessor visualisation -------------------------------------------
    for aid in assessors:
        try:
            assessor_po_path = os.path.join(out_dir, f"PO_comparison_a{aid}.pdf")
            # Check if function supports output_filename parameter
            if 'output_filename' in inspect.signature(PO_plot.compare_and_visualize_assessor).parameters:
                PO_plot.compare_and_visualize_assessor(
                    aid,
                    data.get("M_a_dict", {}).get(aid),
                    data.get("h_true_a_dict", {}).get(aid),
                    final_H.get(aid),
                    data.get("index_to_item_local_dict", {}).get(aid),
                    do_transitive_reduction=True,
                    output_filename=assessor_po_path,
                )
            else:
                # Manual approach
                plt.figure(figsize=(12, 8))
                PO_plot.compare_and_visualize_assessor(
                    aid,
                    data.get("M_a_dict", {}).get(aid),
                    data.get("h_true_a_dict", {}).get(aid),
                    final_H.get(aid),
                    data.get("index_to_item_local_dict", {}).get(aid),
                    do_transitive_reduction=True,
                )
                plt.savefig(assessor_po_path)
                plt.close()
            print(f"Assessor {aid} PO comparison saved to: {assessor_po_path}")
        except Exception as e:
            print(f"ERROR comparing assessor {aid} PO: {e}")

    # RJMCMC‑specific diagnostics -----------------------------------------
    if use_reversible_jump and "K_trace" in results:
        k_post = results["K_trace"][burn_in:]
        if k_post:
            k_counts = np.bincount(k_post)
            print(f"\nK posterior – mode: {k_counts.argmax()}, mean: {np.mean(k_post):.2f}")
            
            # Plot K distribution
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                ks = range(len(k_counts))
                ax.bar(ks, k_counts / sum(k_counts))
                ax.set_xlabel('K')
                ax.set_ylabel('Posterior probability')
                ax.set_title('K posterior distribution')
                k_dist_path = os.path.join(out_dir, "k_posterior.pdf")
                plt.savefig(k_dist_path)
                plt.close()
                print(f"K posterior distribution plot saved to: {k_dist_path}")
            except Exception as e:
                print(f"ERROR plotting K posterior: {e}")

    # Save summary of main results to PDF
    try:
        # Create a PDF with a summary of the results
        from matplotlib.backends.backend_pdf import PdfPages
        
        summary_pdf_path = os.path.join(out_dir, "mcmc_summary.pdf")
        with PdfPages(summary_pdf_path) as pdf:
            # Summary page
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            summary_text = [
                "MCMC Simulation Summary",
                "=" * 50,
                f"Number of items: {len(data.get('M0', []))}",
                f"Number of assessors: {len(data.get('assessors', []))}",
                f"MCMC iterations: {len(log_like)}",
                f"Burn-in period: {burn_in}",
                f"Algorithm: {'RJMCMC' if use_reversible_jump else 'Fixed-K MCMC'}",
                "=" * 50,
            ]
            
            # Add parameter estimates
            if "rho_final" in results:
                summary_text.append(f"Final rho: {results['rho_final']}")
            if "noise_beta_final" in results:
                summary_text.append(f"Final noise beta: {results['noise_beta_final']}")
            if "tau_final" in results:
                # Handle both array and scalar tau values
                tau_final = results['tau_final']
                if hasattr(tau_final, '__len__') and not isinstance(tau_final, str):
                    summary_text.append(f"Cluster assignments: {tau_final}")
                else:
                    summary_text.append(f"Final tau: {tau_final}")
            
            # Add RJMCMC specific info
            if use_reversible_jump and "K_final" in results:
                summary_text.append(f"Final K: {results['K_final']}")
            
            ax.text(0.1, 0.9, "\n".join(summary_text), transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top', family='monospace')
            pdf.savefig()
            plt.close()
            
            # Try to include log-likelihood plot if it exists as an image file
            likelihood_img_path = likelihood_plot_path
            if os.path.exists(likelihood_img_path):
                try:
                    img = plt.imread(likelihood_img_path)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                except Exception as e:
                    print(f"Could not include likelihood plot in summary: {e}")
                
        print(f"Summary PDF saved to: {summary_pdf_path}")
    except Exception as e:
        print(f"ERROR creating summary PDF: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Analysis complete. All plots and results saved to:", os.path.abspath(out_dir))

# ---------------------------------------------------------------------------
# Main function & CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Main execution function: parse args, load data, run MCMC, show and save summary."""
    parser = argparse.ArgumentParser(
        description="Run MCMC simulation for Human Preference Optimization (HPO) based on rankings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--config",
        default=str(CONFIG_DIR / DEFAULT_CONFIG_FILE),
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--rjmcmc",
        action="store_true",
        help="Use the Reversible Jump MCMC sampler (variable K clusters)."
    )
    parser.add_argument(
        "--data_subdir",
        default=None,
        help="Sub-directory within './data' containing the input CSV files. If None, uses './data' directly."
    )
    parser.add_argument(
        "--item_chars",
        default="item_characteristics.csv",
        help="Filename for the item characteristics CSV."
    )
    parser.add_argument(
        "--rankings",
        default="observed_rankings.csv",
        help="Filename for the observed rankings CSV."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config file for reproducibility."
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        default=None,
        help="Number of initial MCMC iterations to discard. Default is 20% of total iterations."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output files and plots. Default is from config or './mcmc_output'."
    )

    args = parser.parse_args()

    # --- Load Config ---
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # Override seed if provided via CLI
    if args.seed is not None:
        print(f"INFO: Overriding random seed from config with CLI argument: {args.seed}")
        cfg['random_seed'] = args.seed

    # Set output directory
    output_dir = args.output_dir
    if output_dir:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Update config
        if "analysis" not in cfg:
            cfg["analysis"] = {}
        cfg["analysis"]["output_dir"] = output_dir

    # --- Load Data ---
    data_base_path = DATA_DIR / args.data_subdir if args.data_subdir else DATA_DIR
    print(f"Attempting to load data from directory: {data_base_path.resolve()}")
    try:
        items_df, assessors_df, selective_sets_df, rankings_df = load_data_from_files(
            data_base_path, args.item_chars, args.rankings
        )
    except Exception as e:
        print(f"\n--- Data Loading Failed ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Prepare Data ---
    try:
        prepared_data = prepare_mcmc_input_data(items_df, assessors_df, selective_sets_df, rankings_df)
    except Exception as e:
        print(f"\n--- Data Preparation Failed ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Run MCMC ---
    print(f"\n--- Running {'RJMCMC' if args.rjmcmc else 'Fixed-K MCMC'} ---")
    start_time = time.time()
    mcmc_results = run_mcmc(prepared_data, cfg, rjmcmc=args.rjmcmc)
    end_time = time.time()

    if mcmc_results is None:
        print("MCMC simulation encountered an error and did not complete.", file=sys.stderr)
        sys.exit(1)

    print(f"MCMC wall-time: {end_time - start_time:.2f}s")

    # --- Analyze Results ---
    print("\n--- Analyzing Results ---")
    try:
        analyze_results(
            results=mcmc_results,
            data=prepared_data,
            config=cfg,
            burn_in=args.burn_in,
            use_reversible_jump=args.rjmcmc,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\n--- Analysis Failed ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        
    print("\n--- MCMC Process Complete ---")

if __name__ == "__main__":
    main()