#!/usr/bin/env python3
"""
ProteinProcessIO - GUI Launcher
================================

Launch the PySide6-based desktop application for protein processing
design and NVIDIA Warp multiphysics simulation.

Usage:
    python run_gui.py

Requirements:
    - PySide6
    - pyvistaqt (optional, for 3D visualization)
    - matplotlib (optional, for result plots)
    - numpy
    - warp-lang (for GPU simulation)
"""

import sys
import os

# Add src to path for development (skip when frozen — PyInstaller handles it)
if not getattr(sys, 'frozen', False):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def check_dependencies():
    """Check for required dependencies and print status."""
    print("Checking dependencies...")

    deps = {
        "PySide6": False,
        "numpy": False,
        "pyvistaqt": False,
        "matplotlib": False,
        "warp": False,
    }

    try:
        import PySide6
        deps["PySide6"] = True
        print(f"  PySide6: OK (v{PySide6.__version__})")
    except ImportError:
        print("  PySide6: MISSING - Install with: pip install PySide6")

    try:
        import numpy as np
        deps["numpy"] = True
        print(f"  numpy: OK (v{np.__version__})")
    except ImportError:
        print("  numpy: MISSING - Install with: pip install numpy")

    try:
        import pyvistaqt
        deps["pyvistaqt"] = True
        print("  pyvistaqt: OK")
    except ImportError:
        print("  pyvistaqt: MISSING (optional) - Install with: pip install pyvistaqt")

    try:
        import matplotlib
        deps["matplotlib"] = True
        print(f"  matplotlib: OK (v{matplotlib.__version__})")
    except ImportError:
        print("  matplotlib: MISSING (optional) - Install with: pip install matplotlib")

    try:
        import warp as wp
        deps["warp"] = True
        wp.init()
        devices = wp.get_devices()
        cuda_devices = [d for d in devices if "cuda" in str(d).lower()]
        if cuda_devices:
            print(f"  warp: OK - CUDA available ({cuda_devices[0]})")
        else:
            print("  warp: OK - CPU mode only (no CUDA)")
    except ImportError:
        print("  warp: MISSING - Install with: pip install warp-lang")
    except Exception as e:
        print(f"  warp: Error - {e}")

    print()

    # Check critical dependencies
    if not deps["PySide6"]:
        print("ERROR: PySide6 is required. Please install it first.")
        return False

    if not deps["numpy"]:
        print("ERROR: numpy is required. Please install it first.")
        return False

    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("ProteinProcessIO")
    print("Protein Processing Design & Simulation")
    print("=" * 60)
    print()

    # Check dependencies (skip when frozen — all deps are bundled)
    if not getattr(sys, 'frozen', False):
        if not check_dependencies():
            sys.exit(1)

    print("Launching application...")
    print()

    # Import and launch
    try:
        from airclassifier.gui import launch_app
        launch_app()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure you're running from the project root directory")
        print("and the package is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
