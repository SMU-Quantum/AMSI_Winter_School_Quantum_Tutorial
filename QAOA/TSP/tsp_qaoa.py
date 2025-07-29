# Import necessary libraries
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (saves to file)
import matplotlib.pyplot as plt
import networkx as nx
from docplex.mp.model import Model
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

# --- Configuration ---
# Set up output directory for plots
output_dir = "tsp_results"
os.makedirs(output_dir, exist_ok=True)

# Set the random seed for reproducibility
seed = 135
# Number of cities
n = 3
num_qubits = n**2

# Configure Qiskit backend
backend = AerSimulator(method='automatic')
estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend)

# --- Generate TSP Instance ---
print("=== Generating TSP Instance ===")
tsp = Tsp.create_random_instance(n, seed=seed)
adj_matrix = nx.to_numpy_array(tsp.graph)

print(f"Number of cities: {n}")
print(f"Number of qubits required (n^2): {num_qubits}")
print("\nCity coordinates:")
for idx, coord in enumerate([tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]):
    print(f"  City {idx}: ({coord[0]:.2f}, {coord[1]:.2f})")

print("\nDistance (adjacency) matrix:")
print(adj_matrix)

# --- Solve Classically (CPLEX) ---
print("\n=== Solving Classically (CPLEX) ===")
# Build the model
m = Model(name='TSP_3cities')
# Suppress CPLEX output
m.set_log_output(False)

# Decision variables
x = {}
for i in range(n):
    for j in range(n):
        if i != j:
            x[i,j] = m.binary_var(name=f"x_{i}_{j}")

# MTZ “u” variables to eliminate subtours
u = {i: m.continuous_var(lb=0, ub=n-1, name=f"u_{i}") for i in range(n)}

# Objective: minimize total travel distance
m.minimize(m.sum(adj_matrix[i,j] * x[i,j] for (i,j) in x))

# Degree constraints: leave each city once, enter each city once
for i in range(n):
    m.add_constraint(m.sum(x[i,j] for j in range(n) if j!=i) == 1, ctname=f"out_{i}")
    m.add_constraint(m.sum(x[j,i] for j in range(n) if j!=i) == 1, ctname=f"in_{i}")

# MTZ subtour-elimination constraints (for i≠0, j≠0)
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            m.add_constraint(u[i] - u[j] + n * x[i,j] <= n - 1, ctname=f"mtz_{i}_{j}")

# Solve
sol = m.solve()

if not sol:
    print("❌ No classical solution found.")
    exit()
else:
    # Extract the tour
    tour = [0]
    current = 0
    for _ in range(n-1):
        for j in range(n):
            if current != j and sol.get_value(x[current,j]) > 0.5:
                tour.append(j)
                current = j
                break
    tour.append(0) # return to start
    classical_cost = m.objective_value
    print("✅ Classical solution found.")
    print(f"Optimal tour: {tour}")
    print(f"Optimal cost: {classical_cost:.4f}")

# --- Visualize Classical Solution ---
def visualize_tsp_solution(G, pos, tour, title="TSP Solution Visualization", filename="tsp_solution.png"):
    """
    Visually highlights the TSP tour and saves the plot.
    """
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600, alpha=0.9, edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=[tour[0]], node_shape='s', node_color='orange', node_size=700, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', style='dashed', alpha=0.5, width=1)

    # Tour edges (circular: back to start)
    tour_edges = [(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))]
    nx.draw_networkx_edges(G, pos, edgelist=tour_edges, edge_color='red', width=2.5, alpha=0.9)

    # Edge labels
    edge_labels = {e: f"{adj_matrix[e[0], e[1]]:.1f}" for e in tour_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='darkred')

    total_cost = sum(adj_matrix[e[0], e[1]] for e in tour_edges)
    plt.title(f"{title}\nTotal Tour Cost: {total_cost:.2f}", fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close() # Important: Close the figure to free memory
    print(f"✅ Classical solution plot saved to: {os.path.join(output_dir, filename)}")

G = tsp.graph
pos = {i: G.nodes[i]["pos"] for i in G.nodes()}
visualize_tsp_solution(G, pos, tour, title="Optimal Classical TSP Tour", filename="classical_tsp_solution.png")


# --- Convert to QUBO ---
print("\n=== Converting Problem to QUBO ===")
problem = tsp.to_quadratic_program()
# print(problem.prettyprint()) # Optional: Uncomment to see QP

converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)
# print(qubo.export_as_lp_string()) # Optional: Uncomment to see QUBO

num_vars = qubo.get_num_vars()
print(f"Number of variables in QUBO: {num_vars}")

# --- Convert QUBO to Ising ---
qubitOp, offset = qubo.to_ising()
print("✅ QUBO converted to Ising Hamiltonian.")
# print("Offset:", offset) # Usually not needed for final output
# print("Ising Hamiltonian:") # Very verbose
# print(str(qubitOp))

# --- Set up QAOA ---
print("\n=== Setting up QAOA ===")
reps = 8 # Reduced for faster execution; increase if needed
circuit = QAOAAnsatz(cost_operator=qubitOp, reps=reps)
circuit = circuit.decompose(reps=3) # Decompose for clarity
# circuit.draw('mpl', filename=os.path.join(output_dir, 'qaoa_circuit.png')) # Optional: Save circuit diagram
# print(circuit.draw()) # Optional: Print text diagram

initial_gamma = np.pi
initial_beta = np.pi/2
init_params = [initial_gamma, initial_beta] * reps
print(f"Initial QAOA parameters: {init_params}")
print(f"QAOA reps: {reps}")

# --- Define Cost Function for Optimization ---
objective_func_vals = [] # Store the objective function values

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    objective_func_vals.append(cost)
    return cost

# --- Run QAOA Optimization ---
print("\n=== Running QAOA Optimization ===")
result = minimize(
    cost_func_estimator,
    init_params,
    args=(circuit, qubitOp, estimator),
    method="COBYLA", # 'L-BFGS-B', 'SLSQP' are alternatives
    options={'maxiter': 5000, 'disp': True}, # Reduced maxiter for demo; increase if needed
    tol=1e-6 # Slightly relaxed tolerance
)

if not result.success:
    print("⚠️  QAOA optimization did not converge successfully.")
else:
    print("✅ QAOA optimization completed.")
    print(f"Optimization message: {result.message}")
    print(f"Final cost (energy): {result.fun:.4f}")

# --- Plot QAOA Convergence ---
def plot_convergence(objective_vals, filename="qaoa_convergence.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(objective_vals, marker='o', linestyle='-', color='royalblue', markersize=4)
    plt.title("QAOA Optimization Convergence", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost (Energy)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ QAOA convergence plot saved to: {os.path.join(output_dir, filename)}")

plot_convergence(objective_func_vals, filename="qaoa_convergence.png")

# --- Analyze QAOA Results ---
print("\n=== Analyzing QAOA Results ===")
# Assign the optimal parameters to the ansatz
optimal_circuit = circuit.assign_parameters(result.x)
optimal_circuit.measure_all()

# Run the sampler job
pub = (optimal_circuit,)
job = sampler.run([pub], shots=int(1e6)) # Reduced shots for speed; increase if needed

# Get the counts
try:
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
except Exception as e:
    print(f"❌ Error retrieving measurement results: {e}")
    exit()

# Normalize the counts to get a probability distribution
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}

# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
if not keys:
    print("❌ No measurement results found.")
    exit()

most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, num_vars)
most_likely_bitstring.reverse()
print(f"Most likely measured bitstring: {most_likely_bitstring}")

# Interpret the result
try:
    result_tsp = converter.interpret(most_likely_bitstring)
    result_value = problem.objective.evaluate(result_tsp)
    is_feasible = problem.get_feasibility_info(result_tsp)[0]
    print(f"Interpreted TSP solution: {result_tsp}")
    print(f"Solution value (distance): {result_value:.4f}")
    print(f"Is solution feasible? {is_feasible}")

    # --- Final Comparison ---
    print("\n" + "="*50)
    print("🏁 FINAL COMPARISON")
    print("="*50)
    print(f"Classical (Best Known) Solution Value: {classical_cost:.4f}")
    print(f"QAOA (Quantum) Solution Value:         {result_value:.4f}")
    print(f"QAOA Solution Feasible:                {is_feasible}")

    if is_feasible and classical_cost != 0:
        optimality_gap = 100 * abs(classical_cost - result_value) / classical_cost
        print(f"Optimality Gap:                        {optimality_gap:.2f}%")
        if abs(result_value - classical_cost) < 1e-3: # Tolerance for numerical errors
            print("✅ Quantum solution matches the classical optimum!")
        else:
            print("ℹ️  Quantum solution is suboptimal compared to classical.")
    elif not is_feasible:
        print("❌ Quantum solution is not feasible.")
    print("="*50)

except Exception as e:
    print(f"❌ Error interpreting QAOA result: {e}")


# --- Show Top 4 Most Probable Solutions ---
print("\n=== Top 4 Most Probable Measured Solutions ===")
top_k = 4
top_indices = np.argsort(np.abs(values))[::-1][:top_k]
col_widths = {"Rank": 5, "Prob": 10, "TSP Sol": 20, "Value": 10, "Feasible": 10, "Gap (%)": 12}
header = (
    f"{'Rank':<{col_widths['Rank']}} | "
    f"{'Prob':<{col_widths['Prob']}} | "
    f"{'TSP Sol':<{col_widths['TSP Sol']}} | "
    f"{'Value':<{col_widths['Value']}} | "
    f"{'Feasible':<{col_widths['Feasible']}} | "
    f"{'Gap (%)':<{col_widths['Gap (%)']}}"
)
print(header)
print("-" * len(header))
for rank, idx in enumerate(top_indices, 1):
    prob = values[idx]
    bitstring = to_bitstring(keys[idx], num_vars)
    bitstring.reverse()
    try:
        interpreted_sol = converter.interpret(bitstring)
        sol_value = problem.objective.evaluate(interpreted_sol)
        sol_feasible = problem.get_feasibility_info(interpreted_sol)[0]

        if sol_feasible and classical_cost != 0:
             gap = 100 * abs(classical_cost - sol_value) / classical_cost
             gap_str = f"{gap:8.2f}"
        else:
             gap_str = "N/A"

        print(
            f"{rank:<{col_widths['Rank']}} | "
            f"{prob:<{col_widths['Prob']}.4f} | "
            f"{str(interpreted_sol):<{col_widths['TSP Sol']}} | "
            f"{sol_value:<{col_widths['Value']}.2f} | "
            f"{str(sol_feasible):<{col_widths['Feasible']}} | "
            f"{gap_str}"
        )
    except Exception:
        print(
            f"{rank:<{col_widths['Rank']}} | "
            f"{prob:<{col_widths['Prob']}.4f} | "
            f"{'Error interpreting':<{col_widths['TSP Sol']}} | "
            f"{'N/A':<{col_widths['Value']}} | "
            f"{'False':<{col_widths['Feasible']}} | "
            f"{'N/A':<{col_widths['Gap (%)']}}"
        )
print("-" * len(header))

print(f"\n📁 All plots saved to the '{output_dir}' directory.")
