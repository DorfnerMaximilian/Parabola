#!/usr/bin/env python3
"""
Main entry point for parabola.cp2k_util module execution.
Usage: python3 -m parabola.run [arguments]
"""
import sys
import argparse
import os
from parabola import cp2k_util

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
        default=6e-4,
        help="Maximum force tolerance (default: 6e-4)"
    )
    parser.add_argument(
        "--tol-rms", 
        type=float,
        default=3e-4,
        help="RMS force tolerance (default: 3e-4)"
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
