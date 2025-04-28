#!/usr/bin/env python3
"""
Multi-Agent Disaster Recovery Example

This example demonstrates the use of Hierarchical Bayesian Partial Order Planning
for a disaster recovery scenario with multiple agents:
- Two robots: R1 and R2
- Two disaster locations: L1 and L2
- A hospital H for delivering survivors
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bayesian_pop.models.hierarchical_planner import HierarchicalPartialOrderPlanner

def create_disaster_recovery_scenario():
    """Create a multi-agent disaster recovery scenario."""
    
    # Define agents
    agents = ["Robot1", "Robot2"]
    
    # Define actions for each agent
    agent_actions = {
        "Robot1": [
            "ClearDebris(L1)",    # 0
            "SearchSurvivor(L1)",  # 1
            "Transport(L1->H)"     # 2
        ],
        "Robot2": [
            "ClearDebris(L2)",    # 0
            "SearchSurvivor(L2)",  # 1
            "Transport(L2->H)"     # 2
        ]
    }
    
    # Define mandatory arcs for each agent
    mandatory_arcs = {
        "Robot1": [
            (0, 1),  # ClearDebris(L1) before SearchSurvivor(L1)
            (1, 2)   # SearchSurvivor(L1) before Transport(L1->H)
        ],
        "Robot2": [
            (0, 1),  # ClearDebris(L2) before SearchSurvivor(L2)
            (1, 2)   # SearchSurvivor(L2) before Transport(L2->H)
        ]
    }
    
    # Define cross-agent constraints
    # For example, if there's only one transport vehicle that both robots share:
    cross_agent_arcs = [
        ("Robot1", 2, "Robot2", 2)  # Robot1's Transport must finish before Robot2's Transport
    ]
    
    # Create the hierarchical planner
    planner = HierarchicalPartialOrderPlanner(
        agents=agents,
        agent_actions=agent_actions,
        mandatory_arcs=mandatory_arcs,
        cross_agent_arcs=cross_agent_arcs,
        K=2,  # Dimension of latent space
        noise_option="queue_jump",
        random_seed=42
    )
    
    # Initialize parameters
    planner.initialize_parameters()
    
    # Set different success probabilities for different actions
    # Adjust the global pi array, which is indexed by global action indices
    for agent, actions in agent_actions.items():
        for i, action in enumerate(actions):
            global_idx = planner.agent_action_to_global[(agent, i)]
            
            if "ClearDebris" in action:
                planner.planner.pi[global_idx] = 0.7  # Clearing debris is harder
            elif "SearchSurvivor" in action:
                planner.planner.pi[global_idx] = 0.8  # Searching is moderately hard
            elif "Transport" in action:
                planner.planner.pi[global_idx] = 0.9  # Transport is easier
    
    return planner

def visualize_multi_agent_plan(planner, agent_executions=None, title="Multi-Agent Partial Order Plan"):
    """
    Visualize the multi-agent partial order plan.
    
    Parameters:
    -----------
    planner: HierarchicalPartialOrderPlanner
        The planner containing the partial order
    agent_executions: dict, optional
        Dictionary mapping agent to execution sequence (for highlighting)
    title: str
        Title for the plot
    """
    G = nx.DiGraph()
    
    # Add nodes for each agent's actions
    node_colors = []
    for agent in planner.agents:
        for i, action in enumerate(planner.agent_actions[agent]):
            node_id = f"{agent}_{i}"
            G.add_node(node_id, label=f"{agent}: {action}")
            
            # Set node color based on agent
            if agent == "Robot1":
                node_colors.append("lightblue")
            else:
                node_colors.append("lightgreen")
    
    # Add edges for mandatory arcs within each agent
    for agent, arcs in planner.mandatory_arcs.items():
        for i, j in arcs:
            G.add_edge(f"{agent}_{i}", f"{agent}_{j}", color="blue", style="solid")
    
    # Add edges for cross-agent arcs
    for agent1, i, agent2, j in planner.cross_agent_arcs:
        G.add_edge(f"{agent1}_{i}", f"{agent2}_{j}", color="red", style="dashed")
    
    # Get additional arcs inferred from the model
    cross_agent_constraints = planner.get_cross_agent_constraints()
    for agent1, i, agent2, j in cross_agent_constraints:
        # Skip if it's already a mandatory arc
        if (agent1, i, agent2, j) in planner.cross_agent_arcs:
            continue
        G.add_edge(f"{agent1}_{i}", f"{agent2}_{j}", color="purple", style="dotted")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors)
    
    # Draw edges with different styles
    # Regular edges (within agents)
    edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style') == 'solid']
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color="blue", arrows=True)
    
    # Cross-agent mandatory edges
    edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style') == 'dashed']
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color="red", arrows=True, style="dashed")
    
    # Inferred cross-agent edges
    edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style') == 'dotted']
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.5, edge_color="purple", arrows=True, style="dotted")
    
    # Draw labels
    labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # Add a legend
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="blue", lw=2, label="Mandatory (within agent)"),
            plt.Line2D([0], [0], color="red", lw=2, label="Mandatory (cross-agent)", linestyle="dashed"),
            plt.Line2D([0], [0], color="purple", lw=1.5, label="Inferred constraint", linestyle="dotted"),
            plt.Line2D([0], [0], marker='o', color='lightblue', label='Robot1', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='lightgreen', label='Robot2', markersize=10, linestyle='None')
        ]
    )
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('multi_agent_disaster_recovery_plan.png')
    plt.show()

def print_agent_executions(planner, agent_executions, agent_outcomes=None):
    """
    Print the execution sequences for each agent.
    
    Parameters:
    -----------
    planner: HierarchicalPartialOrderPlanner
        The planner
    agent_executions: dict
        Dictionary mapping agent to execution sequence
    agent_outcomes: dict, optional
        Dictionary mapping agent to execution outcomes
    """
    print("\nExecution Sequences:")
    print("-------------------")
    
    for agent, execution in agent_executions.items():
        print(f"\n{agent}:")
        for i, action_idx in enumerate(execution):
            action_name = planner.agent_actions[agent][action_idx]
            if agent_outcomes:
                outcome = "Success" if agent_outcomes[agent][i] else "Failure"
                print(f"  {i+1}. {action_name} - {outcome}")
            else:
                print(f"  {i+1}. {action_name}")

def main():
    """Main function demonstrating the hierarchical partial order planning."""
    
    # Create the scenario
    planner = create_disaster_recovery_scenario()
    
    # Visualize the initial plan
    visualize_multi_agent_plan(planner, title="Initial Multi-Agent Plan")
    
    # Generate some sample executions
    print("Generating sample executions...")
    
    all_agent_executions = []
    all_agent_outcomes = []
    
    for i in range(5):
        agent_executions, agent_outcomes = planner.sample_multi_agent_execution_with_outcomes(p_noise=0.2)
        all_agent_executions.append(agent_executions)
        all_agent_outcomes.append(agent_outcomes)
        
        print(f"\nSample Execution {i+1}:")
        print_agent_executions(planner, agent_executions, agent_outcomes)
    
    # Convert to format needed for MCMC update
    observed_orders = {agent: [] for agent in planner.agents}
    success_outcomes = {agent: [] for agent in planner.agents}
    
    for agent_executions, agent_outcomes in zip(all_agent_executions, all_agent_outcomes):
        for agent in planner.agents:
            observed_orders[agent].append(agent_executions[agent])
            success_outcomes[agent].append(agent_outcomes[agent])
    
    # Run MCMC to update the model
    print("\nRunning MCMC to update the model...")
    trace = planner.mcmc_update(
        observed_orders=observed_orders,
        success_outcomes=success_outcomes,
        n_iterations=1000,
        dr=0.95,
        dtau=0.95
    )
    
    # Get and visualize the updated plan
    updated_plan = planner.get_max_posterior_plan()
    visualize_multi_agent_plan(planner, title="Updated Multi-Agent Plan")
    
    # Print the inferred success probabilities
    print("\nInferred Action Success Probabilities:")
    for agent in planner.agents:
        print(f"\n{agent}:")
        for i, action in enumerate(planner.agent_actions[agent]):
            global_idx = planner.agent_action_to_global[(agent, i)]
            print(f"  {action}: {planner.planner.pi[global_idx]:.2f}")
    
    # Plot some of the MCMC traces
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(trace['rho'])
    plt.title('Trace of rho')
    plt.xlabel('Iteration / 10')
    plt.ylabel('rho')
    
    plt.subplot(1, 3, 2)
    plt.plot(trace['tau'])
    plt.title('Trace of tau')
    plt.xlabel('Iteration / 10')
    plt.ylabel('tau')
    
    plt.subplot(1, 3, 3)
    plt.plot(trace['prob_noise'])
    plt.title('Trace of noise probability')
    plt.xlabel('Iteration / 10')
    plt.ylabel('prob_noise')
    
    plt.tight_layout()
    plt.savefig('mcmc_traces.png')
    plt.show()
    
    # Print any cross-agent constraints that were inferred
    cross_agent_constraints = planner.get_cross_agent_constraints()
    if cross_agent_constraints:
        print("\nInferred Cross-Agent Constraints:")
        for agent1, i, agent2, j in cross_agent_constraints:
            action1 = planner.agent_actions[agent1][i]
            action2 = planner.agent_actions[agent2][j]
            print(f"  {agent1}'s {action1} must precede {agent2}'s {action2}")
    
    print("\nHierarchical Bayesian Partial Order Planning complete!")

if __name__ == "__main__":
    main() 