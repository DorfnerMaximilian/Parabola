# Parabola

Parabola is a backend for **CP2K** designed to compute

* vibrational normal modes and their energies
* linear electron–vibration coupling constants

Additionally, it offers several analysis tools:
* Symmetry considerations for Molecules and Periodic Crystals
* Code for Molecular Geometry Optimization
* Bandunfolding for Supercellcalculations

---

## Prerequisites

### Python

* Python ≥ 3.x
* Required packages:

  * `numpy`
  * `scipy`
  * `matplotlib`
  * `ctypes`

### C++

* C++ compiler with **OpenMP** support

---

## Platform Support

Currently tested on:

* **Linux (Ubuntu)**
* **macOS**

Other Linux distributions may work but are not officially tested.

---

## Getting Started

The only mandatory compilation step is building the C++ extension.
All Python dependencies and build steps are handled via **uv**.

---

## Virtual Environment Setup

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and sync the virtual environment

```bash
uv sync --extra dev
```

This will:

* create a virtual environment in `.venv`
* install all required Python dependencies
* build the Parabola package

---

### 3. Activate the virtual environment

Add the virtual environment to your `.bashrc`:

```bash
source /path/to/parabola/.venv/bin/activate
```

Then reload your shell:

```bash
source ~/.bashrc
```

Alternatively, you may activate it manually in each session:

```bash
source .venv/bin/activate
```

---

## Working with `uv`

### Add additional Python dependencies

```bash
uv add numpy scipy matplotlib
```

### Rebuild Parabola after modifying C++ code

After changing any C++ source files, rebuild the package:

```bash
uv sync --reinstall-package parabola
```

Then regenerate Python stubs for the C++ extension:

```bash
uv run pybind11-stubgen -o src parabola._extension
```

---

## Running Tests

To execute the full test suite:

```bash
python -m unittest
```

---

## Notes

* Always ensure your virtual environment is active before running Parabola.
* Rebuilding is required after **any** change to the C++ code.
* OpenMP must be available during compilation for optimal performance.
