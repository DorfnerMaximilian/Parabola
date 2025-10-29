set -e

# --- Check if uv is installed ---
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed."
fi

# --- Sync environment with dev extras ---
echo "Syncing environment with 'uv sync --extra dev'..."
uv sync --extra dev

# --- Add virtual environment activation to .bashrc if missing ---
VIRTUAL_DIR="$(pwd)"
echo "Project directory: $VIRTUAL_DIR"

if ! grep -qF "$VIRTUAL_DIR/.venv/bin/activate" ~/.bashrc; then
    echo "Adding virtual environment activation to ~/.bashrc..."
    echo "source $VIRTUAL_DIR/.venv/bin/activate" >> ~/.bashrc
else
    echo "Virtual environment activation already present in ~/.bashrc."
fi

# --- Activate the virtual environment immediately ---
echo "Activating virtual environment..."
# shellcheck disable=SC1090
source "$VIRTUAL_DIR/.venv/bin/activate"

echo
echo "✅ Installation complete."
echo "Your virtual environment is now active."


