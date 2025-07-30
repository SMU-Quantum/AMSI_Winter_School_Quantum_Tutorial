# 2025-08 UNSW Quantum Optimization
Tutorial on Quantum Optimization

This document outlines the steps to set up the development environment for the project. Due to dependencies such as **CPLEX**, which currently supports up to Python 3.10, it is essential to use **Python 3.10** or earlier.

(or you can use [uv]())

---

## ✅ Prerequisites

Ensure that **Python 3.10** is installed on your system.

To verify your Python version, run:

    python --version
    # or
    python3 --version

If Python 3.10 is not installed, follow the instructions below based on your operating system:

- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Use `brew install python@3.10` (if available)
- **Linux (Debian/Ubuntu)**: Install via `apt` (see troubleshooting section)

> ⚠️ **Note**: Avoid using Python versions newer than 3.10 to ensure compatibility with CPLEX and other dependencies.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

Begin by cloning the project repository:

    git clone https://github.com/SMU-Quantum/2025-08_UNSW_Quantum_Optimization.git
    cd 2025-08_UNSW_Quantum_Optimization

### 2. Create and Activate a Virtual Environment

Using a virtual environment is strongly recommended to isolate project dependencies.

#### Create the virtual environment:

    python3.10 -m venv venv

> On **Windows**, if `python3.10` is not recognized, use:  
> 
>     py -3.10 -m venv venv

#### Activate the virtual environment:

| OS       | Activation Command               |
|---------|----------------------------------|
| Windows | `venv\Scripts\activate`          |
| macOS   | `source venv/bin/activate`       |
| Linux   | `source venv/bin/activate`       |

After activation, you should see `(venv)` prefixed in your shell prompt.

### 3. Verify Python Version

Confirm that the correct Python interpreter is being used:

    python --version

Expected output:
    
    Python 3.10.x

### 4. Install Project Dependencies

Install all required packages from the `requirements.txt` file:

    pip install -r requirements.txt

> Ensure the virtual environment is active before running this command.

---

## ⚠️ Troubleshooting

### ❌ `python3.10` command not found

#### Ubuntu/Debian Linux

Install Python 3.10 and required components:

    sudo apt update
    sudo apt install python3.10 python3.10-venv python3.10-dev

Then create the virtual environment:

    python3.10 -m venv venv

#### macOS (with Homebrew)

Install Python 3.10 using Homebrew:

    brew install python@3.10

Due to Homebrew's path structure, you may need to use the full path:

    # Intel Mac
    /usr/local/bin/python3.10 -m venv venv

    # Apple Silicon (M1/M2/M3)
    /opt/homebrew/bin/python3.10 -m venv venv

#### Windows

Use the Python Launcher for Windows:

    py -3.10 -m venv venv

This method reliably targets Python 3.10 even when multiple versions are installed.

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
- If we measure the qubits in the state above, we just sample every possible solution with the same probability of $\frac{1}{2^n}$, as we could easily do classically.

---

### Key Takeaway

→ To get a quantum computer to tell us the optimal solution of a combinatorial optimization problem with probability higher than $\frac{1}{2^n}$, we have to work a bit harder...