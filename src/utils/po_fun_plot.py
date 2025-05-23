from collections import Counter
import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import seaborn as sns
import pandas as pd  
import pygraphviz
from scipy.stats import beta as beta_dist
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math 
from scipy.stats import beta, kstest
from scipy.integrate import quad
import matplotlib.pyplot as plt
# import pygraphviz as pgv
from scipy.stats import expon, kstest, probplot
import os 
from typing import Dict, Any, List, Optional, Union, Tuple
from tabulate import tabulate
import sys
import csv 
# Make sure these paths and imports match your local project structure
sys.path.append('../src/utils')  # Example path
from po_fun import BasicUtils, StatisticalUtils,GenerationUtils,ConversionUtils

class PO_plot:
    @staticmethod
    def save_rankings_to_csv(y_a_i_dict, output_file='data/observed_rankings.csv'):
        """
        Save y_a_i_dict to a CSV file
        
        Args:
            y_a_i_dict: Dictionary with assessor rankings
            output_file: Path to output CSV file
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Open CSV file for writing
        with open(output_file, 'w', newline='') as csvfile:
            # Create CSV writer
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['assessor_id', 'task_id', 'ranking'])
            
            # Write data rows
            for assessor, tasks in y_a_i_dict.items():
                # Handle the case where tasks is a list (output from generate_total_orders_for_assessor)
                if isinstance(tasks, list):
                    for task_id, order in enumerate(tasks):
                        # Convert order to string if it's a list or tuple
                        if isinstance(order, (list, tuple)):
                            order_str = ','.join(map(str, order))
                        else:
                            order_str = str(order)
                        
                        writer.writerow([assessor, task_id, order_str])
                # Handle the case where tasks is a dictionary (alternative format)
                elif isinstance(tasks, dict):
                    for task_id, orders in tasks.items():
                        for order in orders:
                            # Convert order to string if it's a list or tuple
                            if isinstance(order, (list, tuple)):
                                order_str = ','.join(map(str, order))
                            else:
                                order_str = str(order)
                        
                            writer.writerow([assessor, task_id, order_str])
        
        print(f"Rankings saved to {output_file}")

    @staticmethod
    def plot_Z_trace(Z_trace, index_to_item):
        """
        Plots the trace of multidimensional latent variables Z over iterations.
        
        Parameters:
        - Z_trace: List of Z matrices over iterations. Each Z is an (n x K) array.
        - index_to_item: Dictionary mapping item indices to item labels.
        """
        Z_array = np.array(Z_trace)  # Shape should be (iterations, n, K)
        iterations = Z_array.shape[0]  # Number of iterations

        # Check dimensions
        if Z_array.ndim != 3:
            raise ValueError("Z_trace should be a list of Z matrices with shape (n, K).")

        _, n_items, K = Z_array.shape  # Extract number of items and latent dimensions

        # Create subplots for each dimension
        fig, axes = plt.subplots(K, 1, figsize=(12, 4 * K), sharex=True)
        if K == 1:
            axes = [axes]  # Ensure axes is iterable when K=1

        for k in range(K):
            ax = axes[k]
            for idx in range(n_items):
                ax.plot(range(iterations), Z_array[:, idx, k], label=f"{index_to_item[idx]}")
            ax.set_ylabel(f'Latent Variable Z (Dimension {k + 1})')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True)
        
        axes[-1].set_xlabel('Iteration')
        plt.suptitle('Trace Plot of Multidimensional Latent Variables Z', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_acceptance_rates(accepted_iterations, acceptance_rates):
        """
        Plots the acceptance rates over iterations.
        
        Parameters:
        - accepted_iterations: List of iteration numbers where acceptance rates were recorded.
        - acceptance_rates: List of acceptance rates corresponding to the iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(accepted_iterations, acceptance_rates, marker='o', linestyle='-', color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Acceptance Rate', fontsize=12)
        plt.title('Acceptance Rate Over Time', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_top_partial_orders(top_percentages, top_n=4, item_labels=None):
        """
        Plot the top N partial orders as heatmaps with their corresponding frequencies and percentages.
        
        Parameters:
        - top_percentages: List of tuples containing (partial_order_matrix, count, percentage).
        - top_n: Number of top partial orders to plot.
        - item_labels: List of labels for the items. If None, numerical indices are used.
        """
        # Determine the layout of subplots (e.g., 2x3 for 5 plots)
        n_cols = 2  # Number of columns in the subplot grid
        n_rows = (top_n + n_cols - 1) // n_cols  # Ceiling division for rows
        
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        for idx, (order, count, percentage) in enumerate(top_percentages[:top_n], 1):
            plt.subplot(n_rows, n_cols, idx)
            sns.heatmap(order, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=.5, linecolor='gray',
                        xticklabels=item_labels, yticklabels=item_labels)
            plt.title(f"Top {idx}: {percentage:.2f}%\nCount: {count}")
            plt.xlabel("Items")
            plt.ylabel("Items")
        
        # Remove any empty subplots
        total_plots = n_rows * n_cols
        if top_n < total_plots:
            for empty_idx in range(top_n + 1, total_plots + 1):
                plt.subplot(n_rows, n_cols, empty_idx)
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()



    @staticmethod
    def plot_log_likelihood(log_likelihood_data: Union[Dict[str, Any], List[float]], 
                          burn_in: int = 100,
                            title: str = 'Log Likelihood Over MCMC Iterations') -> None:
        """Plot the total log likelihood over MCMC iterations."""
        if isinstance(log_likelihood_data, dict):
            log_likelihood_currents = log_likelihood_data.get('log_likelihood_currents', [])
        else:
            log_likelihood_currents = log_likelihood_data
        
        if len(log_likelihood_currents) > burn_in:
            burned_ll = log_likelihood_currents[burn_in:]
            iterations = np.arange(burn_in + 1, len(log_likelihood_currents) + 1)
            print(f"Excluding {burn_in} burn-in iterations")
        else:
            burned_ll = log_likelihood_currents
            iterations = np.arange(1, len(log_likelihood_currents) + 1)
            burn_in = 0
            print("No burn-in period applied")
        
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        
        sns.lineplot(x=iterations, y=burned_ll, label='Current State', color='blue')
        
        plt.title(f'{title} (Post Burn-in)', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Total Log Likelihood', fontsize=14)
        
        plt.legend(title='State')
        plt.tight_layout()
        plt.show()
    @staticmethod
    def plot_acceptance_rate(acceptance_rates: List[float], num_iterations: int) -> None:
        """
        Plot the cumulative acceptance rate over MCMC iterations.
        
        Parameters:
        - acceptance_rates (List[float]): Cumulative acceptance rates up to each iteration.
        - num_iterations (int): Total number of iterations.
        
        Returns:
        - None. Displays a matplotlib plot.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        iterations = np.arange(1, num_iterations + 1)
        sns.lineplot(x=iterations, y=acceptance_rates, color='green')
        plt.title('Cumulative Acceptance Rate Over MCMC Iterations', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Cumulative Acceptance Rate', fontsize=14)
        plt.tight_layout()
        plt.show()
    @staticmethod
    def visualize_partial_order(
        final_h: np.ndarray,
        Ma_list: list,
        title: str = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """
        Visualizes the partial order adjacency matrix (final_h) as a graph.
        If 'ax' is given, it draws on that axes, otherwise it creates a new figure.

        final_h : np.ndarray
            N x N adjacency matrix for the partial order
        Ma_list : list
            Labels for the nodes
        title : str, optional
            Title for the plot
        ax : matplotlib.axes.Axes, optional
            If provided, we draw onto this axes. Otherwise, we create a new figure.
        """
        if title is None:
            title = "Partial Order Graph"

        # If the user didn't pass an axes, create a new figure and axes
        own_figure = False
        if ax is None:
            fig, ax = plt.subplots()
            own_figure = True

        # Build a directed graph from final_h
        G = nx.DiGraph(final_h)

        # Create a label mapping
        labels = {i: str(Ma_list[i]) for i in range(len(Ma_list))}

        # Attempt to use PyGraphviz for a 'dot' layout
        try:
            A = nx.nx_agraph.to_agraph(G)
            for node in A.nodes():
                # Convert node name to int if possible
                try:
                    node_int = int(node)
                except ValueError:
                    node_int = node
                node_label = labels.get(node_int, str(node))
                node.attr["label"] = node_label

            A.layout("dot")
            temp_png = "temp_graph.png"
            A.draw(temp_png)

            # Draw the resulting PNG onto the given Axes
            img = plt.imread(temp_png)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title)

        except (ImportError, nx.NetworkXException):
            # Fallback: standard NetworkX spring_layout
            pos = nx.spring_layout(G)
            nx.draw
    @staticmethod
    def visualize_total_orders(total_orders: List[List[int]], top_print: int = 15, top_plot: int = 10) -> None:        
        """
        Visualize the frequency of total orders.

        Parameters:
            total_orders (List[List[int]]): List of total orders, each order is a list of integers.
            top_print (int): Number of top total orders to print.
            top_plot (int): Number of top total orders to plot.

        Returns:
            None. Prints the top_print total orders and displays a bar plot of the top_plot orders.
        """
        
        # 1. Convert total orders to tuples for counting
        total_orders_tuples = [tuple(order) for order in total_orders]
        
        # 2. Count the frequency of each unique total order
        order_counts = Counter(total_orders_tuples)
        
        # 3. Convert tuples to readable strings for better visualization
        total_orders_strings = [' > '.join(map(str, order)) for order in order_counts.keys()]
        frequencies = list(order_counts.values())
        
        # 4. Create a DataFrame from the counter with readable total orders
        df_order_counts = pd.DataFrame({
            'Total Order': total_orders_strings,
            'Frequency': frequencies
        })
        
        # 5. Sort the DataFrame by frequency in descending order
        df_order_counts.sort_values(by='Frequency', ascending=False, inplace=True)
        
        # 6. Reset index for better readability
        df_order_counts.reset_index(drop=True, inplace=True)
        
        # 7. Print the top_print most frequent total orders
        print(f"\nTop {top_print} Most Frequent Total Orders:")
        print(df_order_counts.head(top_print))
        
        # 8. Visualize the frequency counts using Seaborn's barplot
        sns.set(style="whitegrid")  # Set the aesthetic style of the plots
        plt.figure(figsize=(14, 8))  # Set the figure size for better readability
        
        # Create the barplot for top_plot total orders
        sns.barplot(
            x='Total Order',
            y='Frequency',
            data=df_order_counts.head(top_plot),
            palette='viridis'  # Choose a color palette
        )
        
        # Add titles and labels with increased font sizes for clarity
        plt.title(f'Top {top_plot} Most Frequent Total Orders', fontsize=16)
        plt.xlabel('Total Order', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent clipping of tick-labels
        plt.tight_layout()
        
        # Display the plot
        plt.show()

        @staticmethod   
        def plot_heatmap_and_graph(h_matrix: np.ndarray, title: str, item_labels: Optional[List[str]] = None) -> plt.Figure:
            """
            Create a figure with two subplots:
            - Left: A heatmap of the partial order (h_matrix)
            - Right: A network graph visualization using a spring layout
            
            Parameters:
            h_matrix (np.ndarray): The partial order adjacency matrix.
            title (str): A title for the plots.
            item_labels (list, optional): Labels for items used as tick labels.
            
            Returns:
            fig (plt.Figure): The figure containing the two subplots.
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left subplot: Heatmap of h_matrix
            sns.heatmap(h_matrix, annot=True, cmap="viridis", 
                        xticklabels=item_labels, yticklabels=item_labels, ax=axes[0])
            axes[0].set_title("Heatmap: " + title)
            axes[0].set_xlabel("Items")
            axes[0].set_ylabel("Items")
            
            # Right subplot: Network graph using spring layout
            G = nx.DiGraph()
            n = h_matrix.shape[0]
            # Add nodes, using item_labels if available
            for idx in range(n):
                label = item_labels[idx] if item_labels and idx < len(item_labels) else str(idx)
                G.add_node(idx, label=label)
            # Add edges from the adjacency matrix
            for i in range(n):
                for j in range(n):
                    if h_matrix[i, j] == 1:
                        G.add_edge(i, j)
                        
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                    node_size=2000, node_color='lightblue', arrowsize=20, ax=axes[1])
            axes[1].set_title("Graph: " + title)
            axes[1].axis('off')
            
            plt.tight_layout()
            return fig
        @staticmethod   
        def plot_mcmc_results(result_dict: Dict[str, Any], pdf_filename: str, item_labels: Optional[List[str]] = None) -> None:
            pp = PdfPages(pdf_filename)
            
            # Use the length of rho_trace for the iteration axis.
            iterations = range(len(result_dict["rho_trace"]))
            
            # --- Page 1: rho trace ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["rho_trace"], label="rho", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("rho")
            ax.set_title("Trace of rho")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 2: tau trace ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["tau_trace"], label="tau", color="orange", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("tau")
            ax.set_title("Trace of tau")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 3: Noise parameters ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["prob_noise_trace"], label="prob_noise", color="green", marker='o')
            ax.plot(iterations, result_dict["mallow_theta_trace"], label="mallow_theta", color="red", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Noise Parameters")
            ax.set_title("Trace of Noise Parameters")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 4: Log Likelihoods ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["log_likelihood_currents"], label="Current Log Likelihood", color="blue", marker='o')
            ax.plot(iterations, result_dict["log_likelihood_primes"], label="Proposed Log Likelihood", 
                    color="purple", marker='o', linestyle="--")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log Likelihood")
            ax.set_title("Log Likelihood Trace")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 5: Acceptance Rates ---
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(iterations, result_dict["acceptance_rates"], label="Cumulative Acceptance Rate", color="magenta", marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Acceptance Rate")
            ax.set_title("Cumulative Acceptance Rate")
            ax.legend()
            ax.grid(True)
            pp.savefig(fig)
            plt.close(fig)
            
            # --- Page 6: Final Partial Orders with Graphs ---
            if "H_final" in result_dict:
                H_final = result_dict["H_final"]
                # Plot global partial order if available (assumed under key 0)
                if 0 in H_final:
                    fig = PO_plot.plot_heatmap_and_graph(H_final[0], "Final Global Partial Order (H0)", item_labels=item_labels)
                    pp.savefig(fig)
                    plt.close(fig)
                # Plot assessor-level partial orders
                for a in H_final:
                    if a == 0:
                        continue
                    value = H_final[a]
                    if isinstance(value, dict):
                        for task, hm in value.items():
                            fig = PO_plot.plot_heatmap_and_graph(hm, f"Assessor {a} - Task {task} Partial Order", item_labels=item_labels)
                            pp.savefig(fig)
                            plt.close(fig)
                    elif isinstance(value, np.ndarray):
                        fig = PO_plot.plot_heatmap_and_graph(value, f"Assessor {a} Partial Order", item_labels=item_labels)
                        pp.savefig(fig)
                        plt.close(fig)
            
            pp.close()
            print(f"Plots saved to {pdf_filename}")



    @staticmethod
    def plot_inferred_variables(mcmc_results: Dict[str, Any],
                                true_param: Dict[str, Any],
                                config: Dict[str, Any],
                                burn_in: int = 100,
                                output_filename: str = "inferred_parameters.pdf",
                                output_filepath: str = ".",
                                assessors: Optional[List[int]] = None,
                                M_a_dict: Optional[Dict[int, Any]] = None) -> None:
        """
        Plot MCMC traces and densities for inferred parameters (rho, tau, K, and noise parameters)
        excluding beta.
        """
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.1)

        # Define configurations for each variable we want to plot (excluding beta).
        var_configs = {
            'rho': {
                'color': '#1f77b4',
                'prior': 'beta',
                'prior_params': {
                    'a': 1.0,
                    'b': config["prior"].get("rho_prior", 1.0)
                },
                'truncated': True  # plot rho only up to 1 - tol
            },
            'tau': {
                'color': 'brown',
                'prior': 'uniform'
            },
            'K': {
                'color': 'darkcyan',
                # Use a truncated Poisson prior for K.
                'prior': 'truncated_poisson',
                'prior_params': {
                    'lambda': config["prior"].get("K_prior", 1.0)
                }
            }
        }
        
        # Add noise parameters if present.
        noise_model = config.get("noise", {}).get("noise_option", "").lower()
        if noise_model == "queue_jump":
            var_configs['prob_noise'] = {
                'color': 'orange',
                'prior': 'beta',
                'prior_params': {
                    'a': 1.0,
                    'b': config["prior"].get("noise_beta_prior", 1.0)
                }
            }
        elif noise_model == "mallows_noise":
            var_configs['mallow_theta'] = {
                'color': 'purple',
                'prior': None
            }
            
        # Extract traces and true values (post burn-in)
        traces = {}
        true_values = {}
        for var_name, var_config in var_configs.items():
            trace_key = f"{var_name}_trace"
            if trace_key in mcmc_results and mcmc_results[trace_key] is not None:
                traces[var_name] = np.array(mcmc_results[trace_key])[burn_in:]
                true_values[var_name] = true_param.get(f"{var_name}_true", None)
        
        # Create subplots: one row per variable, 2 columns (trace and density)
        n_vars = len(traces)
        # Increase width a bit if necessary (especially to widen the K axis)
        fig, axes = plt.subplots(n_vars, 2, figsize=(14, 4 * n_vars), squeeze=False)
        
        for idx, (var_name, trace) in enumerate(traces.items()):
            var_config = var_configs[var_name]
            true_val = true_values.get(var_name, None)
            
            # --- Trace Plot ---
            ax_trace = axes[idx, 0]
            iterations = np.arange(burn_in + 1, burn_in + 1 + len(trace))
            # Assume trace is 1D for these parameters.
            ax_trace.plot(iterations, trace, color=var_config['color'], lw=1.2, alpha=0.8)
            ax_trace.set_ylabel(var_name, fontsize=12)
            ax_trace.set_xlabel("Iteration", fontsize=12)
            ax_trace.set_title(f"Trace: {var_name}", fontsize=14)
            ax_trace.grid(True, alpha=0.3)
            
            # --- Density / Histogram Plot ---
            ax_hist = axes[idx, 1]
            if var_name == 'rho' and var_config.get('truncated', False):
                tol = 1e-4
                trunc_point = 1 - tol
                bin_edges = np.linspace(0, trunc_point, 101)
                bin_edges[-1] += 1e-6
                sns.histplot(trace, kde=False, ax=ax_hist, color=var_config['color'],
                            bins=bin_edges, edgecolor='black', linewidth=0)
                ax_hist.set_xlim(0.5, trunc_point)
                x_vals = np.linspace(0.5, trunc_point, 1000)
                norm_const = beta_dist.cdf(trunc_point, **var_config['prior_params'])
                norm_const = max(norm_const, 1e-15)
                prior_pdf = beta_dist.pdf(x_vals, **var_config['prior_params']) / norm_const
                ax_hist.plot(x_vals, prior_pdf, 'k-', lw=2, label='Theoretical PDF')
            elif var_name == 'K' and var_config['prior'] == 'truncated_poisson':
                lam = var_config['prior_params']['lambda']
                truncated_poisson = StatisticalUtils.TruncatedPoisson(lam)
                x_vals = np.arange(1, int(np.max(trace)) + 3)
                pdf_vals = np.array([truncated_poisson.pdf(x) for x in x_vals])
                counts, _ = np.histogram(trace, bins=np.append(x_vals, x_vals[-1]+1))
                ax_hist.bar(x_vals, counts, color=var_config['color'], alpha=0.5, width=0.8, label='Sampled')
                scale_factor = len(trace)
                ax_hist.plot(x_vals, pdf_vals * scale_factor, 'k--', lw=2, label='Trunc. Poisson Prior')
                ax_hist.set_xticks(x_vals)
                ax_hist.set_xlabel(var_name, fontsize=12)
            else:
                sns.histplot(trace, kde=True, ax=ax_hist, color=var_config['color'], alpha=0.5)
                if var_config.get('prior') == 'beta' and var_name != 'rho':
                    x_vals = np.linspace(0, 1, 300)
                    pdf_vals = beta_dist.pdf(x_vals, **var_config['prior_params'])
                    scale_factor = len(trace) * (ax_hist.get_xlim()[1] / 30.0)
                    ax_hist.plot(x_vals, pdf_vals * scale_factor, 'k--', label='Beta Prior')
                elif var_config.get('prior') == 'uniform':
                    x_vals = np.linspace(0, max(trace)*1.1, 300)
                    pdf_vals = np.ones_like(x_vals)
                    scale_factor = len(trace) * (ax_hist.get_xlim()[1] / 30.0)
                    ax_hist.plot(x_vals, pdf_vals * scale_factor, 'k--', label='Uniform Prior')
                    
            ax_hist.set_title(f"Density: {var_name}", fontsize=14)
            ax_hist.set_xlabel(var_name, fontsize=12)
            ax_hist.set_ylabel("Count", fontsize=12)
            if true_val is not None:
                # If true_val is an array, take its mean for display.
                if isinstance(true_val, np.ndarray):
                    true_val = float(np.mean(true_val))
                ax_hist.axvline(true_val, color='red', linestyle='--', linewidth=1.5, label='True')
            sample_mean = np.mean(trace)
            ax_hist.axvline(sample_mean, color='green', linestyle='--', linewidth=1.5, label='Sample Mean')
            ax_hist.legend(loc="best", fontsize=10)
        
        plt.tight_layout()
        full_output = os.path.join(output_filepath, output_filename)
        plt.savefig(full_output, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved inferred parameters plot to '{full_output}'")
        plt.show()
        
    
# -------------------------------
# Function 2: Plot Beta Parameters Separately
# -------------------------------
    @staticmethod
    def plot_beta_parameters(mcmc_results: Dict[str, Any],
                            true_param: Dict[str, Any],
                            config: Dict[str, Any],
                            burn_in: int = 100,
                            output_filepath: str = ".") -> None:
        """
        Plot each component of beta in a separate figure. Each figure has two subplots:
        one for the trace and one for the density. Font sizes for beta labels are set very small.
        
        Assumes that mcmc_results["beta_trace"] is a 2D array with shape (n_samples, p),
        and that true_param["beta_true"] is a NumPy array of length p.
        """
        sns.set_style("whitegrid")
        # Use a small font for beta plots.
        beta_font = {
            'title': 8,
            'label': 7,
            'legend': 6,
            'ticks': 6
        }
        
        # Extract beta trace and true beta
        beta_trace = np.array(mcmc_results.get("beta_trace", []))
        if beta_trace.size == 0:
            print("No beta trace available.")
            return
        beta_trace = beta_trace[burn_in:]
        true_beta = true_param.get("beta_true", None)
        
        # Determine dimensions
        n_samples, p_dim = beta_trace.shape
        
        # Create one separate figure per beta coefficient.
        for d in range(p_dim):
            fig, (ax_trace, ax_hist) = plt.subplots(1, 2, figsize=(8, 3))
            iterations = np.arange(burn_in + 1, burn_in + 1 + n_samples)
            
            # TRACE subplot for beta_d
            ax_trace.plot(iterations, beta_trace[:, d], color=plt.cm.tab10(d), lw=1.2, alpha=0.8)
            ax_trace.set_title(f"β{d} Trace", fontsize=beta_font['title'])
            ax_trace.set_xlabel("Iteration", fontsize=beta_font['label'])
            ax_trace.set_ylabel("β value", fontsize=beta_font['label'])
            ax_trace.tick_params(axis='both', labelsize=beta_font['ticks'])
            ax_trace.grid(True, alpha=0.3)
            
            # DENSITY subplot for beta_d
            sns.histplot(beta_trace[:, d], kde=True, ax=ax_hist, color=plt.cm.tab10(d), alpha=0.5)
            ax_hist.set_title(f"β{d} Density", fontsize=beta_font['title'])
            ax_hist.set_xlabel("β value", fontsize=beta_font['label'])
            ax_hist.set_ylabel("Count", fontsize=beta_font['label'])
            ax_hist.tick_params(axis='both', labelsize=beta_font['ticks'])
            if true_beta is not None and d < len(true_beta):
                ax_hist.axvline(true_beta[d], color=plt.cm.tab10(d), linestyle='--', lw=1,
                                label="True β")
            sample_mean = np.mean(beta_trace[:, d])
            ax_hist.axvline(sample_mean, color='green', linestyle='--', lw=1, label="Sample Mean")
            ax_hist.legend(fontsize=beta_font['legend'])
            
            plt.tight_layout()
            outname = os.path.join(output_filepath, f"beta_{d}_plot.pdf")
            plt.savefig(outname, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved beta coefficient {d} plot to '{outname}'")
            plt.show()



    @staticmethod
    #  This function is used to compare two MCMC states
    def compare_two_mcmc_states(
        results_dict: Dict[str, Any],
        h_U_dict: Dict[int, np.ndarray],   # True partial order dictionary: h_U_dict[0] = global, h_U_dict[a] = assessor a
        iteration_idx_1: int,
        iteration_idx_2: int,
        threshold: float = 0.5,
        assessors: Optional[List[int]] = None,
        items: Optional[List[str]] = None,
        plot_partial_orders: bool = True
    ) -> Dict[str, Any]:
        """
        Compare two sampled states from the MCMC trace and the true partial order (h_U_dict).
        We produce a 3-column plot showing:
        - True partial order
        - Iteration 1 partial order
        - Iteration 2 partial order

        The first (top) row is for the global partial order (key=0),
        and subsequent rows are for each assessor in 'assessors'.
        """

        # 1. Convert iteration to trace indices for partial orders
        idx1 = iteration_idx_1 // 100   # thi isbecause when we store the trace, we store it every 100 iterations 
        idx2 = iteration_idx_2 // 100

        # 2. Retrieve partial orders from the MCMC traces
        H_trace = results_dict.get("H_trace", [])
        if not H_trace:
            raise ValueError("No partial-order trace found in results_dict['H_trace'].")
        max_idx = len(H_trace) - 1
        if idx1 > max_idx or idx2 > max_idx:
            raise ValueError(f"H_trace has {len(H_trace)} samples, so max index is {max_idx}, "
                            f"but requested indices are {idx1} and {idx2}.")

        state1 = H_trace[idx1]  # dict: state1[0] = global adjacency, state1[a] = adjacency for assessor a
        state2 = H_trace[idx2]

        if assessors is None:
            assessors = []

        # We'll define a helper to threshold a matrix
        def threshold_matrix(mat: np.ndarray, thr: float) -> np.ndarray:
            return (mat >= thr).astype(int)

        # 3. Plot partial orders
        if plot_partial_orders:
            # Figure with (n_assessors + 1) rows, 3 columns
            n_rows = len(assessors) + 1
            n_cols = 3
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows))

            # If there's only 1 row (0 assessors), axes might be 1D. 
            # Convert to 2D array for consistent indexing
            if n_rows == 1:
                axes = np.array([axes])  # shape (1,3)

            # A small function to handle each "cell" in the grid
            def plot_PO_in_cell(ax, matrix: Optional[np.ndarray], item_labels: List[str], 
                                title: str):
                """
                Plots the partial order (matrix) onto a given Axes (ax).
                If matrix is None, we just leave it blank.
                """
                if matrix is None:
                    ax.set_title(f"{title}\n(no data)")
                    ax.axis("off")
                    return

                # If requested, do thresholding or transitive reduction.
                # But first do thresholding if needed
                # (in your code you did thresholding outside this function)
                matrix = BasicUtils.transitive_reduction(matrix)

                # Actually draw onto the Axes
                PO_plot.visualize_partial_order(
                    final_h=matrix,
                    Ma_list=item_labels,
                    title=title,
                    ax=ax
                )

            # (A) The top row is the global partial order (key=0)
            row_idx = 0
            # col 0: True partial order from h_U_dict[0]
            ax_true_global = axes[row_idx, 0]
            true_global = h_U_dict.get(0, None)
            plot_PO_in_cell(
                ax=ax_true_global,
                matrix=true_global,
                item_labels=items if items else [],
                title="True Global PO"
            )

            # col 1: iteration_idx_1 partial order
            ax_iter1_global = axes[row_idx, 1]
            mat1_global = state1.get(0, None)
            if mat1_global is not None:
                mat1_global = threshold_matrix(mat1_global, threshold)
            plot_PO_in_cell(
                ax=ax_iter1_global,
                matrix=mat1_global,
                item_labels=items if items else [],
                title=f"Global PO (Iter {iteration_idx_1})"
            )

            # col 2: iteration_idx_2 partial order
            ax_iter2_global = axes[row_idx, 2]
            mat2_global = state2.get(0, None)
            if mat2_global is not None:
                mat2_global = threshold_matrix(mat2_global, threshold)
            plot_PO_in_cell(
                ax=ax_iter2_global,
                matrix=mat2_global,
                item_labels=items if items else [],
                title=f"Global PO (Iter {iteration_idx_2})"
            )

            # (B) Rows for each assessor
            for i, assessor in enumerate(assessors, start=1):
                # col 0 => True partial order for this assessor
                ax_true_local = axes[i, 0]
                true_local = h_U_dict.get(assessor, None)
                plot_PO_in_cell(
                    ax=ax_true_local,
                    matrix=true_local,
                    item_labels=items if items else [],
                    title=f"True PO (Assessor {assessor})"
                )

                # col 1 => iteration_idx_1 partial order for assessor
                ax_iter1_local = axes[i, 1]
                mat1_local = state1.get(assessor, None)
                if mat1_local is not None:
                    mat1_local = threshold_matrix(mat1_local, threshold)
                plot_PO_in_cell(
                    ax=ax_iter1_local,
                    matrix=mat1_local,
                    item_labels=items if items else [],
                    title=f"PO (Assr {assessor}, Iter {iteration_idx_1})"
                )

                # col 2 => iteration_idx_2 partial order for assessor
                ax_iter2_local = axes[i, 2]
                mat2_local = state2.get(assessor, None)
                if mat2_local is not None:
                    mat2_local = threshold_matrix(mat2_local, threshold)
                plot_PO_in_cell(
                    ax=ax_iter2_local,
                    matrix=mat2_local,
                    item_labels=items if items else [],
                    title=f"PO (Assr {assessor}, Iter {iteration_idx_2})"
                )

            plt.tight_layout()
            plt.show()

        # 4. Retrieve parameter values from the relevant traces
        param_rho   = results_dict.get("rho_trace", [])
        param_tau   = results_dict.get("tau_trace", [])
        param_noise = results_dict.get("prob_noise_trace", [])
        param_mth   = results_dict.get("mallow_theta_trace", [])
        param_beta  = results_dict.get("beta_trace", [])
        param_K     = results_dict.get("K_trace", [])

        def get_or_none(param_list, ix):
            if (ix < 0) or (ix >= len(param_list)):
                return None
            return param_list[ix]

        rho1 = get_or_none(param_rho,   idx1)
        tau1 = get_or_none(param_tau,   idx1)
        pn1  = get_or_none(param_noise, idx1)
        mth1 = get_or_none(param_mth,   idx1)
        beta1 = get_or_none(param_beta, idx1)
        K1 = get_or_none(param_K, idx1)
        rho2 = get_or_none(param_rho,   idx2)
        tau2 = get_or_none(param_tau,   idx2)
        pn2  = get_or_none(param_noise, idx2)
        mth2 = get_or_none(param_mth,   idx2)
        beta2 = get_or_none(param_beta, idx2)
        K2 = get_or_none(param_K, idx2)
        # 5. Log-likelihood retrieval
        ll_list = results_dict.get("log_likelihood_currents", [])
        llk1 = ll_list[iteration_idx_1] if iteration_idx_1 < len(ll_list) else None
        llk2 = ll_list[iteration_idx_2] if iteration_idx_2 < len(ll_list) else None

        # 6. Build final adjacency data: threshold + transitive_reduction for iteration1 & iteration2
        iteration1_final = {}
        iteration2_final = {}

        # We'll collect both global (0) and assessors
        all_keys = [0] + assessors

        for a in all_keys:
            mat_1 = state1.get(a, None)
            mat_2 = state2.get(a, None)

            if mat_1 is not None:
                bin_1 = (mat_1 >= threshold).astype(int)
                red_1 = BasicUtils.transitive_reduction(bin_1) 
                iteration1_final[a] = red_1

            if mat_2 is not None:
                bin_2 = (mat_2 >= threshold).astype(int)
                red_2 = BasicUtils.transitive_reduction(bin_2) 
                iteration2_final[a] = red_2

        # 7. Return a summary dictionary
        comparison_out = {
            "iteration1_index": iteration_idx_1,
            "iteration2_index": iteration_idx_2,
            "rho1": rho1, "tau1": tau1, "prob_noise1": pn1, "mallow_theta1": mth1,
            "beta1": beta1, "K1": K1,
            "rho2": rho2, "tau2": tau2, "prob_noise2": pn2, "mallow_theta2": mth2,
            "beta2": beta2, "K2": K2,
            "loglik1": llk1, "loglik2": llk2,
            "iteration1_final": iteration1_final,
            "iteration2_final": iteration2_final
        }

        return comparison_out

    @staticmethod
    def print_mcmc_comparison_table(comparison_out, rho_true, tau_true, prob_noise_true):
        # Extract iteration indices
        iterationA = comparison_out["iteration1_index"]
        iterationB = comparison_out["iteration2_index"]

        # Collect rows of data in a list of lists
        table_data = [
            [
                "Iteration 1",
                iterationA,
                comparison_out["rho1"],
                comparison_out["tau1"],
                comparison_out["prob_noise1"],
                comparison_out["mallow_theta1"],
                comparison_out["loglik1"]
            ],
            [
                "Iteration 2",
                iterationB,
                comparison_out["rho2"],
                comparison_out["tau2"],
                comparison_out["prob_noise2"],
                comparison_out["mallow_theta2"],
                comparison_out["loglik2"]
            ],
            [
                "True Params",
                "–",  # No specific iteration index for true params
                rho_true,
                tau_true,
                prob_noise_true,
                "–",  # If you have a true Mtheta, replace "–" with that
                "–"  # True log-likelihood typically not available; replace if you have it
            ]
        ]

        # Define the table headers
        headers = [
            "Label",
            "Iteration",
            "rho",
            "tau",
            "prob_noise",
            "mtheta",
            "log-likelihood"
        ]

        # Print the table in a nice grid format
        print("\n--- Comparison Output Summary (Tabular) ---")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))



    @staticmethod
    # Display the first few rows for verification.
    def plot_time_components_by_category(df: pd.DataFrame) -> None:
        """
        Plots per-iteration time trends for each timing component (PriorTime, LikelihoodTime,
        UpdateTime) broken down by update category.
        
        The function creates three vertically arranged subplots (one per component). 
        Each subplot plots a line (with markers) for each update category.

        Parameters:
        -----------
        df : pd.DataFrame
            A DataFrame containing the following columns:
            - "Iteration": iteration number
            - "UpdateCategory": update category (e.g. "rho", "tau", "noise", "U0", "Ua", "rho_tau")
            - "PriorTime": time spent on prior computations in that iteration
            - "LikelihoodTime": time spent on likelihood calculations in that iteration
            - "UpdateTime": time spent on the update branch in that iteration
        """
        # Set a clean seaborn style.
        sns.set(style="whitegrid", context="talk")
        
        # Filter out rows where UpdateCategory is missing.
        df = df[df["UpdateCategory"].notna()].copy()
        
        # Check if any columns contain dictionary values and convert them to strings
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, dict)).any():
                print(f"Warning: Column '{col}' contains dictionary values. Converting to strings.")
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        
        # Unique update categories.
        categories = sorted(df["UpdateCategory"].unique())
        
        # Define the three timing components to plot.
        components = ["PriorTime", "LikelihoodTime", "UpdateTime"]
        # Define distinct colors for each update category (modify as needed).
        color_map = {"rho": "blue", "tau": "green", "noise": "red", "U0": "purple", "Ua": "orange", "rho_tau": "brown", "K_dim": "cyan", "beta": "magenta"}
        
        # Create one subplot for each timing component.
        fig, axes = plt.subplots(len(components), 1, figsize=(14, 12), sharex=True)
        
        for comp, ax in zip(components, axes):
            for cat in categories:
                # Filter rows for this update category and sort by iteration.
                subset = df[df["UpdateCategory"] == cat].sort_values("Iteration")
                # Get a color for the category, or default to None.
                cat_color = color_map.get(cat, None)
                ax.plot(subset["Iteration"], subset[comp],
                        marker='o', markersize=1, linestyle='-', label=cat,
                        color=cat_color)
            ax.set_ylabel(f"{comp} (s)", fontsize=14)
            ax.legend(title="Update Category", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        axes[-1].set_xlabel("Iteration", fontsize=14)
        fig.suptitle("Per-Iteration Time Trends by Update Category", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()



        # ------------------ Plot 2: Stacked Bar Plot by Update Category ------------------
        # Group by update category and compute the mean (or total) timing for each component.
        grouped = df.groupby("UpdateCategory")[["PriorTime", "LikelihoodTime"]].mean()

        plt.figure(figsize=(10, 6))
        grouped.plot(kind="bar", stacked=True, color=["lightcoral", "lightskyblue"],
                    edgecolor='black')
        plt.xlabel("Update Category", fontsize=12)
        plt.ylabel("Average Time per Iteration (seconds)", fontsize=12)
        plt.title("Average Timing Breakdown per Update Category", fontsize=14)
        plt.legend(title="Component", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def print_comparison_summary(comparison_out):
        """
        Print a formatted summary of the output from a comparison of two MCMC iterations.
        Includes parameter values, log-likelihoods, and partial order adjacency matrices.
        """
        print("\n========== Comparison Output Summary ==========")
        print(f"Iterations: {comparison_out['iteration1_index']} vs {comparison_out['iteration2_index']}\n")

        print("---- Iteration 1 Parameters ----")
        print(f"rho         = {comparison_out['rho1']}")
        print(f"tau         = {comparison_out['tau1']}")
        print(f"prob_noise  = {comparison_out['prob_noise1']}")
        print(f"mallow_theta= {comparison_out['mallow_theta1']}")
        print(f"log-likelihood = {comparison_out['loglik1']:.4f}\n")

        print("---- Iteration 2 Parameters ----")
        print(f"rho         = {comparison_out['rho2']}")
        print(f"tau         = {comparison_out['tau2']}")
        print(f"prob_noise  = {comparison_out['prob_noise2']}")
        print(f"mallow_theta= {comparison_out['mallow_theta2']}")
        print(f"log-likelihood = {comparison_out['loglik2']:.4f}\n")

        print("---- Iteration 1 Partial Orders (transitive-reduced) ----")
        for a, adj in comparison_out["iteration1_final"].items():
            print(f"Assessor {a}, shape = {adj.shape}\n{adj}\n")

        print("---- Iteration 2 Partial Orders (transitive-reduced) ----")
        for a, adj in comparison_out["iteration2_final"].items():
            print(f"Assessor {a}, shape = {adj.shape}\n{adj}\n")

        print("=========================================================")



    @staticmethod
    def plot_update_acceptance_by_category(mcmc_results, desired_order=None, jitter_strength=0.08):
        """
        Summarize and plot MCMC update acceptance by category.

        Parameters:
        - mcmc_results: dict containing key "update_df" with a DataFrame of updates.
        - desired_order: list of category names in the order you'd like them plotted.
                        If None, it defaults to a common order.
        - jitter_strength: float controlling the amount of vertical jitter for visualization.
        """
        if desired_order is None:
            desired_order = ["rho", "tau", "rho_tau","K_dim","beta", "noise", "U0", "Ua"]

        # Create a mapping: category name → numeric value
        category_to_num = {cat: i for i, cat in enumerate(desired_order)}

        # Filter the update DataFrame
        update_df = mcmc_results["update_df"]
        update_df_filtered = update_df[update_df["category"].isin(desired_order)]

        print("\n--- MCMC Update Acceptance Rates by Category ---")
        for category in desired_order:
            cat_data = update_df_filtered[update_df_filtered["category"] == category]
            if not cat_data.empty:
                acceptance_rate = cat_data["accepted"].mean()
                print(f"{category:10s}: {acceptance_rate * 100:.2f}%")
            else:
                print(f"{category:10s}: No updates recorded.")

        # Plotting
        plt.figure(figsize=(25, 6))
        accepted_plotted = set()
        rejected_plotted = set()

        for category in desired_order:
            cat_data = update_df_filtered[update_df_filtered["category"] == category]
            if cat_data.empty:
                continue

            numeric_cat = category_to_num[category]
            accepted = cat_data[cat_data["accepted"]]
            rejected = cat_data[~cat_data["accepted"]]

            if not accepted.empty:
                y_jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(accepted))
                label = "Accepted" if "Accepted" not in accepted_plotted else None
                accepted_plotted.add("Accepted")
                plt.scatter(accepted["iteration"], numeric_cat + y_jitter,
                            color="green", marker="o", s=20, alpha=0.8, label=label)

            if not rejected.empty:
                y_jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(rejected))
                label = "Rejected" if "Rejected" not in rejected_plotted else None
                rejected_plotted.add("Rejected")
                plt.scatter(rejected["iteration"], numeric_cat + y_jitter,
                            color="red", marker="x", s=20, alpha=0.8, label=label)

        # Format plot
        plt.yticks(ticks=list(category_to_num.values()), labels=desired_order, fontsize=12)
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Update Category", fontsize=14)
        plt.title("Update Category by Iteration (Green=Accepted, Red=Rejected)", fontsize=16)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()





    @staticmethod
    def compare_and_visualize_global(
        h_true_global: np.ndarray,
        h_inferred_global: np.ndarray,
        index_to_item_global: Dict[int, int],
        global_Ma_list: List[str],
        do_transitive_reduction: bool = True
    ) -> None:

        h_true_plot = BasicUtils.transitive_reduction(h_true_global)
        h_inferred_plot = BasicUtils.transitive_reduction(h_inferred_global)

        PO_plot.visualize_partial_order(
            final_h=h_true_plot,
            Ma_list=global_Ma_list,
            title='True Global Partial Order'
        )
        PO_plot.visualize_partial_order(
            final_h=h_inferred_plot,
            Ma_list=global_Ma_list,
            title='Inferred Global Partial Order'
        )

        # Compute and print missing and redundant relationships.
        missing_relationships = BasicUtils.compute_missing_relationships(h_true_plot,h_inferred_plot, index_to_item_global)
        redundant_relationships = BasicUtils.compute_redundant_relationships(h_true_plot, h_inferred_plot, index_to_item_global)

        if missing_relationships:
            print("\nMissing (true PO edges not in inferred PO):")
            for i, j in missing_relationships:
                print(f"{i} < {j}")
        else:
            print("\nNo missing relationships in global partial order.")

        if redundant_relationships:
            print("\nRedundant (inferred PO edges not in true PO):")
            for i, j in redundant_relationships:
                print(f"{i} < {j}")
        else:
            print("\nNo redundant relationships in global partial order.")

    @staticmethod
    def compare_and_visualize_assessor(
        assessor: int,
        Ma_list: List[str],
        h_true_a: np.ndarray,
        h_inferred_a: np.ndarray,
        index_to_item_local: Dict[int, int],
        do_transitive_reduction: bool = True
    ) -> None:
        """
        Compare and visualize the partial orders for a specific assessor, printing missing and redundant edges.

        Parameters
        ----------
        assessor : int
            The assessor's identifier.
        Ma_list : List[str]
            List of item labels for the assessor.
        h_true_a : np.ndarray
            True local partial order adjacency matrix for the assessor.
        h_inferred_a : np.ndarray
            Inferred local partial order adjacency matrix for the assessor.
        index_to_item_local : Dict[int, int]
            Mapping from local indices to items for the assessor.
        do_transitive_reduction : bool, optional
            Whether to apply transitive reduction (default is True).
        """


        h_true_plot = BasicUtils.transitive_reduction(h_true_a)
        h_inferred_plot = BasicUtils.transitive_reduction(h_inferred_a)



        # Visualize local partial orders for the assessor.
        PO_plot.visualize_partial_order(
            final_h=h_true_plot,
            Ma_list=Ma_list,
            title=f"True Local Partial Order (Assessor={assessor})"
        )
        PO_plot.visualize_partial_order(
            final_h=h_inferred_plot,
            Ma_list=Ma_list,
            title=f"Inferred Local Partial Order (Assessor={assessor})"
        )

        # Compute and print missing and redundant relationships.
        missing_relationships = BasicUtils.compute_missing_relationships(h_true_a, h_inferred_a, index_to_item_local)
        redundant_relationships = BasicUtils.compute_redundant_relationships(h_true_a, h_inferred_a, index_to_item_local)

        if missing_relationships:
            print(f"\nMissing edges for assessor {assessor}:")
            for i, j in missing_relationships:
                print(f"{i} < {j}")
        else:
            print(f"\nNo missing edges for assessor {assessor}.")

        if redundant_relationships:
            print(f"\nRedundant edges for assessor {assessor}:")
            for i, j in redundant_relationships:
                print(f"{i} < {j}")
        else:
            print(f"\nNo redundant edges for assessor {assessor}.")

    @staticmethod
    def plot_joint_parameters(mcmc_results):
        """
        Given an mcmc_results dictionary from the HPO simulation, this function
        creates scatter plots of:
        - p (i.e. prob_noise) versus rho
        - p (i.e. prob_noise) versus tau
        so you can inspect the joint behavior of these parameters across MCMC iterations.
        
        Parameters:
        -----------
        mcmc_results : dict
            Dictionary output from mcmc_simulation_hpo, expected to contain:
            - "rho_trace": list of rho values per iteration
            - "tau_trace": list of tau values per iteration
            - "prob_noise_trace": list of noise probability values per iteration (interpreted as p)
        """
        # Extract traces
        rho_trace = mcmc_results["rho_trace"]
        tau_trace = mcmc_results["tau_trace"]
        prob_noise_trace = mcmc_results["prob_noise_trace"]

        # Create a figure with two subplots.
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))

        # Plot p vs. rho
        axes[0].scatter(prob_noise_trace, rho_trace, alpha=0.6 , color='navy')
        axes[0].set_xlabel("p (noise probability)", fontsize=12)
        axes[0].set_ylabel("rho", fontsize=12)
        axes[0].set_title("Joint Behavior: p vs. rho", fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Plot p vs. tau
        axes[1].scatter(prob_noise_trace, tau_trace, alpha=0.6, color='darkgreen')
        axes[1].set_xlabel("p (noise probability)", fontsize=12)
        axes[1].set_ylabel("tau", fontsize=12)
        axes[1].set_title("Joint Behavior: p vs. tau", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.5)


        axes[2].scatter(rho_trace, tau_trace, alpha=0.6, color='darkred')
        axes[2].set_xlabel("rho)", fontsize=12)
        axes[2].set_ylabel("tau", fontsize=12)
        axes[2].set_title("Joint Behavior: p vs. tau", fontsize=14)
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()


    @staticmethod

    def plot_u0_ua_diagnostics(results, assessors, M0, K):
        """
        Present posterior diagnostics for U0 and Ua by displaying clean, informative plots.

        Parameters
        ----------
        results : dict
            Output from mcmc_simulation_hpo(...). Must include:
                - "U0_trace": list of (n_global x K) arrays
                - "Ua_trace": list of assessor->(M_a x K) dicts
        assessors : list[int]
            List of assessor IDs.
        M0 : list[int]
            Global item indices.
        K : int
            Dimension of latent space.
        """
        sns.set(style="whitegrid", font_scale=1.1)

        # ─────────────────────────────────────────────────────────────
        # U0 Posterior Diagnostics
        # ─────────────────────────────────────────────────────────────
        U0_trace = results.get("U0_trace", [])


        U0_stack = np.stack(U0_trace)  # (n_samples, n_global, K)
        U0_flat = U0_stack.reshape(-1, K)

        print(f"[U0] Total samples: {U0_flat.shape[0]} | K = {K}")

        # 1. Histogram per U0 dimension
        fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 4))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for k in range(K):
            sns.histplot(U0_flat[:, k], bins=100, stat="density", ax=axes[k],
                        color="dodgerblue", edgecolor=None, alpha=0.6)
            axes[k].set_title(f"U0: dim {k}", fontsize=13)
            axes[k].set_xlabel("Latent Value")
            axes[k].set_ylabel("Density")

        plt.suptitle("Posterior Marginals of U0 (Flattened)", fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # 2. Correlation matrix
        corr_matrix = np.corrcoef(U0_flat.T)
        plt.figure(figsize=(4, 4))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, center=0,
                    cbar_kws={"shrink": 0.6})
        plt.title("Correlation Matrix Across U0 Dimensions", fontsize=12)
        plt.tight_layout()
        plt.show()

        # ─────────────────────────────────────────────────────────────
        # Ua Posterior Diagnostics
        # ─────────────────────────────────────────────────────────────
        Ua_trace = results.get("Ua_trace", [])


        for a in assessors:
            all_rows = [Ua[a] for Ua in Ua_trace if a in Ua]
            if not all_rows:
                print(f"[Ua] Assessor {a} not found in Ua trace.")
                continue

            arr_flat = np.concatenate(all_rows, axis=0)  # shape: (n_total_items, K)
            print(f"[Ua] Assessor {a}: {arr_flat.shape[0]} total rows.")

            fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 4))
            axes = axes if isinstance(axes, np.ndarray) else [axes]

            for k in range(K):
                sns.histplot(arr_flat[:, k], bins=100, stat="density", ax=axes[k],
                            color="mediumseagreen", edgecolor=None, alpha=0.6)
                axes[k].set_title(f"Ua: assessor={a}, dim {k}", fontsize=13)
                axes[k].set_xlabel("Latent Value")
                axes[k].set_ylabel("Density")

            plt.suptitle(f"Posterior Marginals for Assessor {a} (Flattened)", fontsize=15, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
