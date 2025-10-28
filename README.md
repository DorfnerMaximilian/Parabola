# Parabola
Parabola is intended to be a backend for cp2k to compute the normal modes, their energies and the linear electron-vibration/-phonon coupling constants. It provides apart from that other useful tools that utilize the analysis of electron-phonon coupled systems. 
## Prerequisites
python3: numpy, scipy, matplotlib, ctypes
c++: OpenMp

## Getting started
Up to now this is tested for Linux Ubuntu
The only really mandatory thing is to compile the c++ part of the program.
Just install all requirements (via apt), and compile the file...

## Setting up virtual environment
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh 
source $HOME/.local/bin/env
```bash

# Now add uv to your PATH in your .bashrc
```bash
source path_to_your_parabola/.venv/bin/activate
```bash

# setting up virtual environment
uv sync --extra dev
```

## Additional Instructions for uv
# How To: add python modules
```bash
uv add numpy scipy matplotlib
```

# Haw TO: rebuild parabola after making changes to C++ code:
```bash
uv sync --reinstall-package parabola
uv run pybind11-stubgen -o src parabola._extension
```bash



