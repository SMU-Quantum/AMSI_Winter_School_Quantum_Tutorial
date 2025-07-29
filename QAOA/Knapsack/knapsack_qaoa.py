# knapsack_qaoa.py
#
# Description:
# This script solves the Knapsack problem using the Quantum Approximate Optimization
# Algorithm (QAOA) with Qiskit. It demonstrates problem setup, conversion to an
# Ising Hamiltonian, QAOA execution, and result interpretation.
#
# To Run:
# Ensure you have a Python environment with the following libraries installed:
# - qiskit
# - qiskit_optimization
# - qiskit_aer
# - matplotlib
# - numpy
# - scipy
# Then, execute the script from your terminal: python knapsack_qaoa.py

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- Quantum and Optimization Imports ---
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_optimization.algorithms import CplexOptimizer

# --- Setup Output Directory for Plots ---
if not os.path.exists("qaoa_plots"):
    os.makedirs("qaoa_plots")

def print_section(title):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60 + "\n")

# --- 1. Problem Definition ---
print_section("1. Defining the Knapsack Problem")

# Set the random seed for reproducibility
np.random.seed(135)

# Parameters for the knapsack problem
num_items = 4
weights = np.random.randint(1, 10, size=num_items)
values = np.random.randint(10, 50, size=num_items)
capacity = int(0.6 * np.sum(weights))

print(f"Number of items: {num_items}")
print(f"Item weights: {weights}")
print(f"Item values:  {values}")
print(f"Knapsack capacity: {capacity}\n")

# Create the Knapsack problem instance and convert it to a Quadratic Program
knapsack = Knapsack(values.tolist(), weights.tolist(), capacity)
problem = knapsack.to_quadratic_program()
print("Quadratic Program Formulation:\n")
print(problem.prettyprint())

# --- 2. Classical Benchmark Solution ---
print_section("2. Solving Classically with CPLEX")

optimizer = CplexOptimizer()
result_classical = optimizer.solve(problem)
print("Classical Solution using CPLEX Optimizer:")
print(f"  - Optimal value: {result_classical.fval}")
print(f"  - Solution vector: {result_classical.x}\n")

# --- 3. Convert Problem for Quantum Solver (QUBO to Ising) ---
print_section("3. Converting to QUBO and Ising Hamiltonian")

converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)
num_vars = qubo.get_num_vars()
print(f"Number of variables in QUBO (including slack variables): {num_vars}\n")

qubitOp, offset = qubo.to_ising()
print(f"Ising Hamiltonian Offset: {offset:.4f}")
print("Ising Hamiltonian (Qubit Operator):\n")
print(str(qubitOp))

# --- 4. QAOA Setup ---
print_section("4. Setting up the QAOA Algorithm")

# Backend simulator
backend = AerSimulator(method="automatic")
estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend)

# Define the QAOA ansatz circuit
reps = 1  # Number of QAOA layers
ansatz = QAOAAnsatz(cost_operator=qubitOp, reps=reps)
ansatz = ansatz.decompose()
print(f"QAOA Ansatz created with {reps} layer(s).")

# Save ansatz visualization
fig, ax = plt.subplots(figsize=(10, 6))
ansatz.draw("mpl", style="iqp", ax=ax)
ax.set_title("QAOA Ansatz Circuit", fontsize=16)
plt.savefig("qaoa_plots/qaoa_ansatz_circuit.png", dpi=300)
plt.close(fig)
print("Ansatz circuit diagram saved to: qaoa_plots/qaoa_ansatz_circuit.png\n")

# Store objective function values during optimization
objective_func_vals = []

# Cost function for the classical optimizer
def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    cost = result[0].data.evs[0]
    objective_func_vals.append(cost)
    print(f"  - Current Parameters: {np.round(params, 4)} | Energy: {cost:.6f}")
    return cost

# --- 5. Run QAOA ---
print_section("5. Running the QAOA Optimization")

# Initial parameters for QAOA (gamma and beta)
init_params = [np.pi, np.pi/2] * reps

print("Starting classical optimization of QAOA parameters...\n")
result = minimize(
    cost_func_estimator,
    init_params,
    args=(ansatz.decompose(), qubitOp, estimator),
    method="COBYLA",
    tol=1e-8
)
print("\nOptimization complete.")
print(f"  - Final QAOA Energy: {result.fun:.6f}")
print(f"  - Optimal Parameters: {np.round(result.x, 4)}\n")

# --- 6. Analyze QAOA Results ---
print_section("6. Analyzing QAOA Results")

# Plot the convergence
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(objective_func_vals, "o-")
ax.set_xlabel("Iteration Number", fontsize=14)
ax.set_ylabel("Cost (Energy)", fontsize=14)
ax.set_title("QAOA Convergence Plot", fontsize=16)
ax.grid(True, linestyle="--", alpha=0.7)
plt.savefig("qaoa_plots/qaoa_convergence.png", dpi=300)
plt.close(fig)
print("Convergence plot saved to: qaoa_plots/qaoa_convergence.png\n")

# --- 7. Sample the Solution ---
print_section("7. Sampling the Solution")

# Assign optimal parameters and sample
optimal_circuit = ansatz.assign_parameters(result.x)
optimal_circuit = optimal_circuit.decompose()
optimal_circuit.measure_all()

pub = (optimal_circuit,)
job = sampler.run([pub], shots=int(1e4))
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_bin.values())
final_distribution = {key: val/shots for key, val in counts_bin.items()}

# Analyze the top 4 most probable solutions
values = np.array(list(final_distribution.values()))
keys = list(final_distribution.keys())
top_4_indices = np.argsort(np.abs(values))[::-1][:4]

print("Top 4 Most Probable Solutions from QAOA:\n")
header = (
    f"{'Rank':<5} | {'Bitstring':<{num_vars+2}} | {'Probability':<11} | "
    f"{'Knapsack Solution':<18} | {'Value':<7} | {'Feasible':<8}"
)
print(header)
print("-" * len(header))

for rank, idx in enumerate(top_4_indices, 1):
    bitstring_rev = keys[idx]
    bitstring = [int(b) for b in bitstring_rev[::-1]]
    interpreted = converter.interpret(bitstring)
    value = problem.objective.evaluate(interpreted)
    feasible = problem.get_feasibility_info(interpreted)[0]
    
    print(
        f"{rank:<5} | {str(bitstring):<{num_vars+2}} | {values[idx]:<11.6f} | "
        f"{str(interpreted):<18} | {value:<7.1f} | {str(feasible):<8}"
    )

print("\n--- Script Finished: Knapsack QAOA Demonstration Complete ---")
