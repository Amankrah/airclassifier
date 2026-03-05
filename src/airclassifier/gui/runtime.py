"""
Runtime environment helpers for the GUI.

Used to tailor messages and behavior for packaged (installed) vs development runs.
"""

import sys


def is_packaged_app() -> bool:
    """True if the app is running as a frozen/packaged build (e.g. PyInstaller)."""
    return getattr(sys, "frozen", False) or getattr(sys, "import_frozen", False)


def pyvista_unavailable_message() -> str:
    """User-facing message when PyVista 3D is not available."""
    if is_packaged_app():
        return (
            "3D visualization is not available in this installation.\n\n"
            "Simulation will run without 3D view."
        )
    return (
        "PyVista not available.\n"
        "Install with: pip install pyvista pyvistaqt\n\n"
        "Simulation will run without 3D visualization."
    )
