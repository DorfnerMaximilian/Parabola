#!/bin/bash
# --- Configuration ---
TARGET_DIR="../parabola_light"
SOURCE_DIR="."
MODULES_TO_COPY=(
"PhysConst.py"
"cp2k_util.py"
"coordinate_tools.py"
"Read.py"
"Write.py"
)
# --- End of Configuration ---

echo "🚀 Starting the packaging process..."

# Clean up any previous attempt
if [ -d "$TARGET_DIR" ]; then
    echo "🧹 Removing old '$TARGET_DIR' directory."
    rm -r "$TARGET_DIR"
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p "$TARGET_DIR/Modules"
mkdir -p "$TARGET_DIR/run"

# Create __init__.py files
echo "📝 Creating package initialization files..."

# Main package __init__.py (simplified for lightweight version)
cat > "$TARGET_DIR/__init__.py" << 'EOF'
# Lightweight parabola package
from .Modules import PhysConst
from .Modules import cp2k_util
from .Modules import coordinate_tools
from .Modules import Read
from .Modules import Write
EOF

# Modules package __init__.py
cat > "$TARGET_DIR/Modules/__init__.py" << 'EOF'
# Modules package
from . import PhysConst
from . import cp2k_util
from . import coordinate_tools
from . import Read
from . import Write
EOF

# Create cp2k_util/__main__.py for module execution
cat > "$TARGET_DIR/run/__main__.py" << 'EOF'
"""
Main entry point for parabola.cp2k_util module execution.
Usage: python3 -m parabola.run [arguments]
"""
import sys
import argparse
import os
from parabola_light.Modules import cp2k_util

def main():
    """Main function to handle command line execution."""
    parser = argparse.ArgumentParser(
        description="Parabola package runner for geometry optimization and CP2K calculations",
        prog="python3 -m parabola.run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m parabola.run                          # Run in current directory
  python3 -m parabola.run /path/to/calc            # Run in specific directory
  python3 -m parabola.run --cluster slurm          # Run on SLURM cluster
  python3 -m parabola.run --tol-max 1e-3 ./calc   # Custom convergence tolerance
        """
    )
    
    # Main arguments
    parser.add_argument(
        "path", 
        nargs="?",
        default="./",
        help="Path to calculation directory (default: current directory)"
    )
    
    # Execution environment
    parser.add_argument(
        "--cluster", 
        choices=["local", "slurm"],
        default="local",
        help="Execution environment (default: local)"
    )
    
    # Convergence criteria
    parser.add_argument(
        "--tol-max", 
        type=float,
        default=5e-4,
        help="Maximum force tolerance (default: 5e-4)"
    )
    parser.add_argument(
        "--tol-rms", 
        type=float,
        default=1e-4,
        help="RMS force tolerance (default: 1e-4)"
    )
    
    # Output control
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress optimization progress output"
    )
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        return
    
    # Validate path
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Display run information
    if args.verbose:
        print("=" * 60)
        print("PARABOLA GEOMETRY OPTIMIZATION")
        print("=" * 60)
        print(f"Working directory: {os.path.abspath(args.path)}")
        print(f"Cluster environment: {args.cluster}")
        print(f"Max force tolerance: {args.tol_max:.2e}")
        print(f"RMS force tolerance: {args.tol_rms:.2e}")
        print("=" * 60)
    
    # Run geometry optimization
    try:
        cp2k_util.geo_opt(
            cluster=args.cluster,
            path=args.path,
            tol_max=args.tol_max,
            tol_drm=args.tol_rms
        )
        
        if not args.quiet:
            print("\n✅ Geometry optimization completed successfully!")
            print(f"Results saved in: {os.path.join(args.path, 'Geo_Optimization')}")
            
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during optimization: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Copy specified modules
echo "📦 Copying specified modules..."
for module in "${MODULES_TO_COPY[@]}"; do
    SOURCE_FILE="$SOURCE_DIR/Modules/$module"
    if [ -f "$SOURCE_FILE" ]; then
        cp "$SOURCE_FILE" "$TARGET_DIR/Modules/"
        echo " -> Copied $module"
    else
        echo " -> ⚠️ Warning: Module '$module' not found at '$SOURCE_FILE'. Skipping."
    fi
done

echo ""
echo "✅ Success! Your lightweight package is ready in the '$TARGET_DIR' directory."
echo "Final structure:"
if command -v tree &> /dev/null; then
    tree "$TARGET_DIR"
else
    find "$TARGET_DIR" -type f | sort
fi
echo ""
echo "Usage: python3 -m parabola.cp2k_util [arguments]"
