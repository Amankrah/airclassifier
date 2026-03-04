#!/usr/bin/env python3
"""
Build script for ProteinProcessIO desktop application.

This script compiles the Python application into standalone executables
for Windows, macOS, and Linux using PyInstaller.

Usage:
    python build.py                 # Build for current platform
    python build.py --onefile       # Single executable (slower startup)
    python build.py --clean         # Clean build artifacts first
    python build.py --debug         # Build with console for debugging

Requirements:
    pip install pyinstaller
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_platform():
    """Get current platform name."""
    if sys.platform == 'win32':
        return 'windows'
    elif sys.platform == 'darwin':
        return 'macos'
    else:
        return 'linux'


def clean_build():
    """Remove previous build artifacts."""
    print("Cleaning previous build artifacts...")

    dirs_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.pyc', '*.pyo', '*.spec.bak']

    root = Path(__file__).parent

    for dir_name in dirs_to_clean:
        dir_path = root / dir_name
        if dir_path.exists():
            print(f"  Removing {dir_path}")
            shutil.rmtree(dir_path)

    # Clean __pycache__ recursively
    for pycache in root.rglob('__pycache__'):
        print(f"  Removing {pycache}")
        shutil.rmtree(pycache)

    print("Clean complete.\n")


def check_dependencies():
    """Check if required build tools are installed."""
    print("Checking build dependencies...")

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  PyInstaller: OK (v{PyInstaller.__version__})")
    except ImportError:
        print("  PyInstaller: MISSING")
        print("\n  Install with: pip install pyinstaller")
        return False

    # Check main dependencies
    deps = ['PySide6', 'numpy', 'warp']
    for dep in deps:
        try:
            __import__(dep)
            print(f"  {dep}: OK")
        except ImportError:
            print(f"  {dep}: MISSING")
            return False

    print()
    return True


def build_executable(onefile=False, debug=False):
    """Build the executable using PyInstaller."""
    platform = get_platform()
    print(f"Building ProteinProcessIO for {platform}...")
    print()

    # Construct PyInstaller command
    cmd = ['pyinstaller', 'proteinprocessio.spec']

    if onefile:
        cmd.append('--onefile')
        print("  Mode: Single executable (--onefile)")
    else:
        print("  Mode: Directory bundle")

    if debug:
        # Modify spec for debug mode (show console)
        print("  Debug: Console enabled")

    cmd.extend([
        '--noconfirm',  # Overwrite without asking
        '--clean',      # Clean cache
    ])

    print(f"  Command: {' '.join(cmd)}")
    print()

    # Run PyInstaller
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print("\nBuild FAILED!")
        return False

    print("\nBuild completed successfully!")
    print_output_location(onefile)
    return True


def print_output_location(onefile):
    """Print where the built executable is located."""
    platform = get_platform()
    dist_dir = Path(__file__).parent / 'dist'

    print("\n" + "=" * 60)
    print("OUTPUT LOCATION")
    print("=" * 60)

    if platform == 'macos':
        app_path = dist_dir / 'ProteinProcessIO.app'
        if app_path.exists():
            print(f"\nmacOS App Bundle:")
            print(f"  {app_path}")
            print("\nTo run:")
            print(f"  open {app_path}")
    elif platform == 'windows':
        if onefile:
            exe_path = dist_dir / 'ProteinProcessIO.exe'
        else:
            exe_path = dist_dir / 'ProteinProcessIO' / 'ProteinProcessIO.exe'
        print(f"\nWindows Executable:")
        print(f"  {exe_path}")
        print("\nTo run:")
        print(f"  {exe_path}")
    else:  # Linux
        if onefile:
            exe_path = dist_dir / 'ProteinProcessIO'
        else:
            exe_path = dist_dir / 'ProteinProcessIO' / 'ProteinProcessIO'
        print(f"\nLinux Executable:")
        print(f"  {exe_path}")
        print("\nTo run:")
        print(f"  {exe_path}")

    print()


def create_installer(skip_build=False):
    """Create platform-specific installer."""
    platform = get_platform()
    root = Path(__file__).parent

    if platform == 'windows':
        print("\n" + "=" * 60)
        print("CREATING WINDOWS INSTALLER")
        print("=" * 60)

        # Check if Inno Setup is installed
        inno_paths = [
            Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
            Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
        ]

        iscc_path = None
        for path in inno_paths:
            if path.exists():
                iscc_path = path
                break

        if iscc_path is None:
            print("\nInno Setup not found!")
            print("\nTo create a Windows installer:")
            print("  1. Download Inno Setup from: https://jrsoftware.org/isinfo.php")
            print("  2. Install it (default location)")
            print("  3. Run: python build.py --installer")
            print("\nOr manually compile installer.iss with Inno Setup Compiler")
            return False

        # Check if dist folder exists
        dist_folder = root / 'dist' / 'ProteinProcessIO'
        if not dist_folder.exists():
            print(f"\nError: {dist_folder} not found!")
            print("Run 'python build.py' first to create the executable.")
            return False

        # Create installer output directory
        installer_dir = root / 'dist' / 'installer'
        installer_dir.mkdir(exist_ok=True)

        # Run Inno Setup compiler
        iss_file = root / 'installer.iss'
        print(f"\nCompiling installer with Inno Setup...")
        print(f"  Script: {iss_file}")
        print(f"  Compiler: {iscc_path}")

        result = subprocess.run([str(iscc_path), str(iss_file)], cwd=root)

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("INSTALLER CREATED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nInstaller location:")
            print(f"  {installer_dir / 'ProteinProcessIO-1.0.0-Setup.exe'}")
            print("\nThis installer will:")
            print("  - Install to Program Files")
            print("  - Create Start Menu shortcut")
            print("  - Optionally create Desktop shortcut")
            print("  - Add to Add/Remove Programs")
            return True
        else:
            print("\nInstaller creation FAILED!")
            return False

    elif platform == 'macos':
        print("\nCreating macOS DMG...")
        dist_dir = root / 'dist'
        app_path = dist_dir / 'ProteinProcessIO.app'
        dmg_path = dist_dir / 'ProteinProcessIO-1.0.0.dmg'

        if not app_path.exists():
            print(f"Error: {app_path} not found!")
            return False

        cmd = [
            'hdiutil', 'create',
            '-volname', 'ProteinProcessIO',
            '-srcfolder', str(app_path),
            '-ov',
            str(dmg_path)
        ]
        result = subprocess.run(cmd)
        return result.returncode == 0

    else:  # Linux
        print("\nTo create Linux AppImage, use appimagetool")
        print("  https://appimage.github.io/")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build ProteinProcessIO desktop application'
    )
    parser.add_argument(
        '--onefile', '-o',
        action='store_true',
        help='Create single executable (slower startup)'
    )
    parser.add_argument(
        '--clean', '-c',
        action='store_true',
        help='Clean build artifacts before building'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Build with console window for debugging'
    )
    parser.add_argument(
        '--clean-only',
        action='store_true',
        help='Only clean, do not build'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check dependencies'
    )
    parser.add_argument(
        '--installer', '-i',
        action='store_true',
        help='Create installer after building (requires Inno Setup on Windows)'
    )
    parser.add_argument(
        '--installer-only',
        action='store_true',
        help='Only create installer (skip PyInstaller build)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ProteinProcessIO Build System")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        print("Missing dependencies. Please install them first.")
        sys.exit(1)

    if args.check:
        print("All dependencies OK!")
        sys.exit(0)

    # Clean if requested
    if args.clean or args.clean_only:
        clean_build()

    if args.clean_only:
        sys.exit(0)

    # Installer only mode
    if args.installer_only:
        success = create_installer(skip_build=True)
        sys.exit(0 if success else 1)

    # Build
    success = build_executable(onefile=args.onefile, debug=args.debug)

    if success:
        if args.installer:
            create_installer()
        else:
            print("\nTo create a Windows installer, run:")
            print("  python build.py --installer-only")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
