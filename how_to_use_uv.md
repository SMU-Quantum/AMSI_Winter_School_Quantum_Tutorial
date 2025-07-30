# uv: Ultra-Fast Python Packaging Tool

`uv` is a high-performance Python package installer and resolver, built in Rust by [Astral](https://github.com/astral-sh/uv). It replaces `pip`, `pip-tools`, and `virtualenv` with a single, blazing-fast tool.

Designed for modern workflows, `uv` accelerates dependency resolution, virtual environment creation, and package installation — often **10–100x faster** than traditional tools.

For more information, visit the [official repository](https://github.com/astral-sh/uv).

---

## ✅ Supported Python Versions

As of June 2025, `uv` supports Python 3.7 through **Python 3.13**. However, due to dependency constraints (e.g., **CPLEX**), it is recommended to use **Python 3.10 or earlier**.

| Version | Status       | Support Ends | Notes                                   |
|--------|--------------|--------------|-----------------------------------------|
| 3.14   | Pre-release  | 2030-10      | Not yet supported by `uv` stable          |
| 3.13   | Bugfix       | 2029-10      | Supported, but not compatible with CPLEX |
| 3.12   | Security     | 2028-10      | Supported, not CPLEX-compatible           |
| 3.11   | Security     | 2027-10      | Supported, not CPLEX-compatible           |
| 3.10   | Security     | 2026-10      | ✅ Recommended — compatible with CPLEX    |
| 3.9    | Security     | 2025-10      | Compatible, nearing end of life           |
| 3.8    | End of Life  | 2024-10      | ❌ No longer supported                    |

> ⚠️ **Important**: This project depends on **CPLEX**, which requires **Python ≤ 3.10**. Use **Python 3.10.18** (latest as of June 3, 2025).

To verify your Python version:

    python --version
    # or
    python3 --version

If Python 3.10 is not installed, see OS-specific instructions below.

---

## 📦 Installation

`uv` can be installed via multiple methods. Choose the one that best fits your system.

### Option 1: Standalone Installer (Recommended)

Downloads a precompiled binary for maximum speed.

#### macOS/Linux:

    curl -LsSf https://install.python-uv.com | sh

Installs to `~/.local/bin/uv`. Ensure this directory is in your `PATH`.

#### Windows (PowerShell):

    irm https://install.python-uv.com | iex

Installs to `%APPDATA%\uv\bin\uv.exe` and adds it to user `PATH`.

> 🔄 After installation, restart your shell or reload environment variables.

---

### Option 2: Install via pip

If you already have Python:

    pip install uv

> ✅ Works on all platforms.  
> ⚠️ Slightly slower than the standalone binary.

---

### Option 3: Package Managers

#### macOS (Homebrew):

    brew install uv

#### Linux (Debian testing or Ubuntu via third-party repo):

    sudo apt install uv

#### Windows (winget):

    winget install astral-sh.uv

---

## 🛠️ Project Setup with uv

### 1. Clone the Repository

    git clone https://github.com/SMU-Quantum/2025-08_UNSW_Quantum_Optimization.git
    cd 2025-08_UNSW_Quantum_Optimization

### 2. Create a Virtual Environment

`uv` includes built-in virtual environment management.

#### Create a venv with Python 3.10:

    uv venv venv --python 3.10

> If `python3.10` is not found, ensure it's installed and discoverable.

#### Alternative: Use full path (macOS/Homebrew):

    /opt/homebrew/bin/python3.10 -m venv venv   # Apple Silicon
    # or
    /usr/local/bin/python3.10 -m venv venv      # Intel

#### On Windows with Python Launcher:

    uv venv venv --python py:3.10

This uses the `py` launcher to locate Python 3.10.

---

### 3. Activate the Virtual Environment

| OS       | Command                          |
|---------|----------------------------------|
| Windows | `venv\Scripts\activate`          |
| macOS   | `source venv/bin/activate`       |
| Linux   | `source venv/bin/activate`       |

After activation, your prompt should display `(venv)`.

---

### 4. Verify Python Version

Ensure the correct interpreter is active:

    python --version

Expected output:

    Python 3.10.18

---

## 📥 Install Dependencies with `uv pip`

`uv` provides a `pip`-compatible subcommand for installing packages.

### Install from requirements.txt

    uv pip sync requirements.txt

> ✅ This resolves and installs all dependencies **instantly**, similar to `pip install -r requirements.txt`, but faster.

Alternatively, to install without syncing (i.e., allow extra packages):

    uv pip install -r requirements.txt

### Install a Single Package

    uv pip install numpy

### Install in Editable Mode

    uv pip install -e .

Supports development workflows with local packages.

### Compile a requirements.in File

If you have a `requirements.in` (unpinned), resolve and pin dependencies:

    uv pip compile requirements.in -o requirements.txt

> ✅ Replaces `pip-compile` from `pip-tools` with superior performance.

---

## 🔁 Advanced Usage

### Run a Script Directly (No venv needed)

    uv run python script.py

Automatically creates an isolated environment and runs the script.

### List Installed Packages

    uv pip list

### Show a Package

    uv pip show package_name

### Upgrade pip in the Virtual Environment

    uv pip install --upgrade pip

---

## ⚠️ Troubleshooting

### ❌ `uv: command not found`

Ensure the install directory is in your `PATH`:
- **macOS/Linux**: `~/.local/bin`
- **Windows**: `%APPDATA%\uv\bin`

Add to shell config (Linux/macOS):

    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc

---

### ❌ Python 3.10 Not Found

#### Ubuntu/Debian:

    sudo apt update
    sudo apt install python3.10 python3.10-venv python3.10-dev

Then:

    uv venv venv --python 3.10

#### macOS (Homebrew):

    brew install python@3.10

Then use full path:

    uv venv venv --python /opt/homebrew/bin/python3.10

#### Windows:

Use the Python Launcher:

    uv venv venv --python py:3.10

> 💡 List available Python versions:
>
>     py -0

---

### ❌ Certificate or Sigstore Verification Error

Starting with Python 3.10.7, releases are signed using **Sigstore**.

- **Windows**: Executables are Authenticode-signed by *Python Software Foundation*.
- **macOS**: Installer packages are signed with Apple Developer ID `BMM5U3QVKW`.

Ensure your system trusts these signatures. For air-gapped environments, download from trusted mirrors or verify manually.

See: [How to verify your downloaded files are genuine](https://www.python.org/downloads/)

---

## ✅ Why Use uv?

| Feature                | Benefit                                      |
|------------------------|----------------------------------------------|
| Blazing-fast installs  | Up to 100x faster than `pip`                 |
| Built-in virtualenv    | No need for `python -m venv`                 |
| `pip-tools` alternative| `uv pip compile` replaces `pip-compile`      |
| Offline caching        | Reinstall from cache without internet        |
| Cross-platform         | Consistent on Windows, macOS, Linux          |
| Low memory usage       | Efficient resolver design                    |

---

## 📚 Official Resources

- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Migration Guide from pip](https://docs.astral.sh/uv/migration/pip/)
- [Release Notes](https://github.com/astral-sh/uv/releases)

---

## ✅ Summary

With `uv`, you can:
- Install and manage Python packages at unprecedented speed
- Create and manage virtual environments seamlessly
- Replace `pip`, `pip-tools`, and `virtualenv` with a single tool
- Ensure compatibility with Python 3.7–3.13, while targeting **3.10** for CPLEX
- Install dependencies using `uv pip install`, `uv pip sync`, and more

You are now ready to develop, test, and deploy with a modern, high-performance Python toolchain.