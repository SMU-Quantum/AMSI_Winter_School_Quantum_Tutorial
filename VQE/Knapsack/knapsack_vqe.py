# knapsack_vqe.py
#
# Description:
# This script solves the Knapsack problem using the Variational Quantum Eigensolver (VQE)
# algorithm with Qiskit. It demonstrates defining the problem, converting it for a
# quantum solver, running VQE, and interpreting the results.
#
# To Run:
# Ensure you have a Python environment with the following libraries installed:
# - qiskit
# - qiskit_optimization
# - qiskit_aer
# - matplotlib
# - numpy
# - scipy
# Then, execute the script from your terminal: python knapsack_vqe.py

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- Quantum and Optimization Imports ---
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import EfficientSU2
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_optimization.algorithms import CplexOptimizer

# --- Setup Output Directory for Plots ---
if not os.path.exists("vqe_plots"):
    os.makedirs("vqe_plots")

# --- 1. Problem Definition ---
print("="*40)
print("1. Defining the Knapsack Problem")
print("="*40)

# Set the random seed for reproducibility
np.random.seed(135)

# Parameters for the knapsack problem
num_items = 4
weights = np.random.randint(1, 10, size=num_items)
values = np.random.randint(10, 50, size=num_items)
capacity = int(0.6 * np.sum(weights))

print(f"Weights: {weights}")
print(f"Values: {values}")
print(f"Capacity: {capacity}\n")

# Create the Knapsack problem instance and convert it to a Quadratic Program
knapsack = Knapsack(values.tolist(), weights.tolist(), capacity)
problem = knapsack.to_quadratic_program()
print("Quadratic Program Formulation:")
print(problem.prettyprint())

# --- 2. Classical Benchmark Solution ---
print("\n" + "="*40)
print("2. Solving Classically with CPLEX")
print("="*40)

# Use a classical optimizer to find the exact solution for benchmarking
optimizer = CplexOptimizer()
result_classical = optimizer.solve(problem)
print(f"Optimal value (Classical): {result_classical.fval}")
print(f"Solution vector (Classical): {result_classical.x}\n")

# --- 3. Convert Problem for Quantum Solver (QUBO to Ising) ---
print("\n" + "="*40)
print("3. Converting to QUBO and Ising Hamiltonian")
print("="*40)

# Convert the problem to a Quadratic Unconstrained Binary Optimization (QUBO) problem
converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)
num_vars = qubo.get_num_vars()
print(f"Number of variables in QUBO (including slack variables): {num_vars}\n")

# Convert the QUBO to an Ising Hamiltonian for VQE
qubitOp, offset = qubo.to_ising()
print(f"Ising Hamiltonian Offset: {offset}")
print("Ising Hamiltonian (Qubit Operator):")
print(str(qubitOp))

# --- 4. VQE Setup ---
print("\n" + "="*40)
print("4. Setting up the VQE Algorithm")
print("="*40)

# Backend simulator for running quantum circuits
backend = AerSimulator(method="automatic")
estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend)

# Define the parameterized quantum circuit (ansatz)
reps = 1
ansatz = EfficientSU2(qubitOp.num_qubits, reps=reps, entanglement='linear', insert_barriers=True)
ansatz = ansatz.decompose()  # Decompose to visualize the circuit structure
num_params = ansatz.num_parameters
print(f"Ansatz: EfficientSU2 with {num_params} parameters.")

# Save ansatz visualization
fig, ax = plt.subplots(figsize=(10, 6))
ansatz.decompose().draw("mpl", style="iqp", ax=ax)
ax.set_title("VQE Ansatz Circuit (EfficientSU2)", fontsize=16)
plt.savefig("vqe_plots/ansatz_circuit.png", dpi=300)
plt.close(fig)
print("Saved ansatz circuit diagram to vqe_plots/ansatz_circuit.png")

# Dictionary to store the history of the optimization process
cost_history_dict = {"iters": 0, "prev_vector": None, "cost_history": []}

# Cost function for the classical optimizer
def cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iteration: {cost_history_dict['iters']} | Current Energy: {energy:.4f}")
    return energy

# --- 5. Run VQE ---
print("\n" + "="*40)
print("5. Running the VQE Optimization")
print("="*40)

# Generate a random starting point for the optimizer
x0 = 2 * np.pi * np.random.random(num_params)

# Use SciPy's COBYLA optimizer to find the minimum energy
res = minimize(
    cost_func,
    x0,
    args=(ansatz.decompose(), qubitOp, estimator),
    method="cobyla",
    options={"maxiter": 1000, "disp": True, "tol": 1e-6}
)
print("\nOptimization finished.")
print(f"Final Energy (VQE): {res.fun:.4f}\n")

# --- 6. Analyze VQE Results ---
print("\n" + "="*40)
print("6. Analyzing VQE Results")
print("="*40)

# Plot the convergence of the VQE algorithm
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"], "o-")
ax.set_xlabel("Iteration Number", fontsize=14)
ax.set_ylabel("Energy (Cost Function Value)", fontsize=14)
ax.set_title("VQE Convergence Plot", fontsize=16)
ax.grid(True, linestyle="--", alpha=0.7)
plt.savefig("vqe_plots/vqe_convergence.png", dpi=300)
plt.close(fig)
print("Saved convergence plot to vqe_plots/vqe_convergence.png\n")

# --- 7. Sample the Solution from the Optimized Circuit ---
print("\n" + "="*40)
print("7. Sampling the Solution")
print("="*40)

# Assign the optimal parameters to the ansatz and add measurements
optimal_circuit = ansatz.assign_parameters(res.x)
optimal_circuit = optimal_circuit.decompose()
optimal_circuit.measure_all()

# Run the sampler to get measurement counts
pub = (optimal_circuit,)
job = sampler.run([pub], shots=int(1e4))
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_bin.values())
final_distribution = {key: val/shots for key, val in counts_bin.items()}

# Function to convert integer to bitstring
def to_bitstring(integer, num_bits):
    return [int(digit) for digit in np.binary_repr(integer, width=num_bits)]

# Analyze the top 4 most probable solutions
values = np.array(list(final_distribution.values()))
keys = list(final_distribution.keys())
top_4_indices = np.argsort(np.abs(values))[::-1][:4]

print("Top 4 Most Probable Solutions from VQE:\n")
header = f"{'Rank':<5} | {'Bitstring':<{num_vars+2}} | {'Probability':<11} | {'Knapsack Solution':<18} | {'Value':<7} | {'Feasible':<8}"
print(header)
print("-" * len(header))

for rank, idx in enumerate(top_4_indices, 1):
    bitstring_rev = keys[idx]
    # Reverse the bitstring to match Qiskit's convention
    bitstring = [int(b) for b in bitstring_rev[::-1]]
    interpreted = converter.interpret(bitstring)
    value = problem.objective.evaluate(interpreted)
    feasible = problem.get_feasibility_info(interpreted)[0]
    
    print(
        f"{rank:<5} | {str(bitstring):<{num_vars+2}} | {values[idx]:<11.6f} | "
        f"{str(interpreted):<18} | {value:<7.1f} | {str(feasible):<8}"
    )

print("\n--- Script Finished ---")