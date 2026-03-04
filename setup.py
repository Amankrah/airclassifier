"""
Cyclone Air Classifier Simulation Package

GPU-accelerated particle separation simulation using NVIDIA Warp.
"""

from setuptools import setup, find_packages

setup(
    name="airclassifier",
    version="0.1.0",
    description="GPU-accelerated cyclone air classifier simulation using NVIDIA Warp",
    author="Emmanuel Amankrah Kwofie",
    author_email="emmanuel.kwofie@mail.mcgill.ca",
    url="https://www.eakwofie.com/",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "warp-lang>=1.11.0",
        "numpy>=2.0.0",
        "scipy>=1.15.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "pyglet>=2.0.0",
            "vtk>=9.0.0",
        ],
        "gui": [
            "PySide6>=6.5.0",
            "pyvista>=0.42.0",
            "pyvistaqt>=0.11.0",
            "vtk>=9.2.0",
            "h5py>=3.0.0",
        ],
        "ml": [
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "airclassifier=airclassifier.simulation.simulator:main",
        ],
        "gui_scripts": [
            "airclassifier-gui=airclassifier.gui:launch_app",
        ],
    },
)
