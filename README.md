# 2025-08 UNSW Quantum Optimization
Tutorial on Quantum Optimization

## Installation 

To set up the repository, follow these steps:

1. **Clone the repository:**
        
        git clone https://github.com/SMU-Quantum/2025-08_UNSW_Quantum_Optimization.git
        cd 2025-08_UNSW_Quantum_Optimization

2. **Set up a virtual environment (recommended):**

        python3 -m venv venv
        venv\Scripts\activate  # on linux, or macos use `source venv/bin/activate`

3. **Install Dependencies:**

        pip install -r requirements.txt


Ensure that your system has Python 3.9 or higher installed.





---



## Quantum Computers Can Evaluate All Possible Solutions at the Same Time...

> **"Quantum Computers can evaluate all possible solutions at the same time..."**
>
> → **A common misunderstanding, particularly for combinatorial optimization!**

### Explanation

Suppose the equal superposition state:

$$|+\rangle_n = H^{\otimes n}|0\rangle_n = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle_n $$

Furthermore, suppose a function $f: \{0, \dots, 2^n - 1\} \to \mathbb{R}$ and a corresponding quantum operation:

$$ F: |x\rangle_n |0\rangle_m \mapsto |x\rangle_n |f(x)\rangle_m $$

Thus, we get:
$$ F|+\rangle_n = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle_n |f(x)\rangle_m $$

And the quantum computer evaluates the exponential number of solutions in parallel!

Doesn't it?

Unfortunately:
- It remains an exponential number of possible solutions.
- The quantum computer does NOT tell us which one achieves the optimum.
- If we measure the qubits in the state above, we just sample every possible solution with the same probability of $ \frac{1}{2^n} $, as we could easily do classically.

---

### Key Takeaway

→ To get a quantum computer to tell us the optimal solution of a combinatorial optimization problem with probability higher than $ \frac{1}{2^n} $, we have to work a bit harder...