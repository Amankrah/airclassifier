# ProteinProcessIO

GPU-accelerated design and simulation of air classification systems for particle separation, powered by [NVIDIA Warp](https://github.com/NVIDIA/warp).

Built for dry fractionation of plant-based food powders — separating protein, starch, and fiber fractions from yellow peas, faba beans, and oats.

## Features

- **Interactive GUI** — PySide6 desktop application with real-time 3D viewport (PyVista), assembly configuration, simulation control, and results analysis
- **Parametric Geometry** — 40+ components (cyclones, blowers, dampers, hoppers, airlocks, screw feeders, zigzag classifiers, wheel classifiers, venturi eductors, duct sections, bag filters, and more) with automatic assembly and port-to-port connection
- **Physics-Based Simulation** — Lagrangian particle tracking with Schiller-Naumann drag, gravity, wall collisions, centrifugal separation, and SPH-based air flow, all GPU-accelerated on NVIDIA Warp
- **Two Classification Modes** — Full system (venturi + zigzag preclassification + wheel classifier + multi-stage cyclones + bag filter) or wheel-only (direct feed to wheel classifier)
- **Real Food Powders** — Material presets for yellow pea, faba bean, and oat whole flour with measured size distributions and density data for protein, starch, and fiber fractions
- **Multi-Pass Recirculation** — Recirculate collected fractions through the system with venturi attrition modelling
- **Animated 3D Preview** — Mechanical animations (blower spin-up, damper open/close, lid servo, wheel rotation) driven by subsidiary physics simulators during both preview and live simulation
- **Cinematic Camera** — Optional game-style camera modes (Orbit, Showcase tour, Flythrough) during simulation with mouse-pause override

## Screenshot

The GUI provides a 3D viewport with the assembled system, toolbar controls, and a simulation panel with real-time KPI cards:

```text
+-------------------------------------------------------------+
|  Menu Bar  |  File  Edit  View  Assembly  Simulation  Help  |
+-------------------------------------------------------------+
|  Toolbar: New | Open | Save | Configure Assembly | Build    |
+-------------------------------------------------------------+
|                                                             |
|              3D Viewport (PyVista)                           |
|   View: Isometric | Edges | Wireframe | Particles | ...    |
|                                                             |
+-------------------------------------------------------------+
|  Simulation  |  Control  |  Settings  |  Log                |
|  [Run]  [Pause]  [Stop]     Progress: ████████ 100%        |
|  Sim Time: 360.0 s          Active Particles: 0            |
|  Fines: 2,341                Coarse: 2,659                  |
|  Separation Efficiency: 46.8%                               |
+-------------------------------------------------------------+
```

## Project Structure

```text
airclassifier/
├── src/airclassifier/
│   ├── geometry/              # Parametric components and assemblies
│   │   ├── components/        #   40+ individual components
│   │   ├── assembly/          #   Feed, air, classification, complete system
│   │   ├── primitives/        #   Cylinder, cone, tube builders
│   │   ├── sdf.py             #   Signed distance fields for collision
│   │   └── mesh_generator.py  #   Triangle mesh generation
│   ├── fluid/                 # Flow field solvers
│   │   ├── solvers/           #   Navier-Stokes, pressure projection
│   │   ├── kernels/           #   Advection, diffusion, projection
│   │   └── turbulence/        #   k-epsilon models, wall functions
│   ├── particles/             # Material and particle systems
│   │   ├── material.py        #   Food powder material definitions
│   │   ├── particle_system.py #   GPU particle system (Warp)
│   │   ├── drag_models/       #   Stokes, Schiller-Naumann, Haider-Levenspiel
│   │   └── interactions/      #   Particle-particle, particle-wall collisions
│   ├── kinetics/              # Force calculations and trajectory
│   │   ├── forces/            #   Drag, gravity, centrifugal, virtual mass
│   │   ├── cut_size.py        #   Cut size calculations
│   │   └── separation_efficiency.py
│   ├── simulation/            # Simulation orchestration
│   │   ├── classification_flow_physics.py  # Main classifier simulation (Warp)
│   │   ├── air_flow_physics.py             # SPH air flow simulation
│   │   ├── feed_flow_physics.py            # Feed system material flow
│   │   ├── airclass_flow_physics.py        # Air + classification coupling
│   │   ├── feedclass_flow_physics.py       # Feed + classification coupling
│   │   └── cfd_dem_coupling.py             # Full CFD-DEM coupling
│   ├── gui/                   # Desktop application (PySide6 + PyVista)
│   │   ├── main_window.py     #   Main window with menus, toolbar, docks
│   │   ├── simulation_backend.py  # Backend bridging GUI ↔ simulation
│   │   ├── panels/            #   Simulation control, results, properties
│   │   ├── dialogs/           #   Assembly config, simulation settings
│   │   └── widgets/           #   3D viewport, animation controller,
│   │                          #   cinematic camera
│   ├── visualization/         # CLI plotting and rendering
│   ├── io/                    # VTK export, config I/O
│   └── utils/                 # Constants, units, validation
├── examples/                  # CLI scripts (see below)
├── tests/                     # Test suite (geometry, integration)
├── config/                    # YAML configuration files
├── geometry_exports/          # Pre-exported STL meshes
└── utility_docs/              # Technical reports and analysis
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd airclassifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install core dependencies
pip install -r requirements.txt

# Install GUI dependencies (PySide6, PyVista)
pip install -r requirements-gui.txt

# Install package in development mode
pip install -e .
```

### Requirements

| Dependency   | Version    | Purpose                                  |
| ------------ | ---------- | ---------------------------------------- |
| Python       | >= 3.10    | Runtime                                  |
| warp-lang    | >= 1.11.0  | GPU-accelerated simulation (NVIDIA Warp) |
| numpy        | >= 2.0.0   | Numerical computing                      |
| scipy        | >= 1.15.0  | Scientific computing                     |
| PySide6      | >= 6.5.0   | GUI framework (Qt6)                      |
| pyvista      | >= 0.42.0  | 3D visualization                         |
| pyvistaqt    | >= 0.11.0  | PyVista Qt integration                   |
| matplotlib   | >= 3.7.0   | Plotting                                 |
| vtk          | >= 9.2.0   | Visualization toolkit                    |

**Hardware**: NVIDIA GPU with CUDA support recommended. CPU mode is supported but slower.

## Quick Start

### GUI Application

```bash
python run_gui.py
```

Or from Python:

```python
from airclassifier.gui import launch_app
launch_app()
```

**Workflow:**

1. **Configure Assembly** (`Ctrl+Shift+A`) — set wheel RPM, diameter, mode (full system or wheel-only), enable/disable subsystems
2. **Build Full System** (`Ctrl+B`) — generates 3D geometry with animated components
3. **Run Simulation** (`F5`) — plays startup preamble (8 s), then runs classification physics
4. View results in the **Results** tab; export as CSV/JSON

### CLI Simulation

```bash
# Full-system classification with yellow pea flour (5000 particles, 360 s)
python examples/run_classification_flow.py \
    --blower-rpm 500 --wheel-rpm 3000 \
    --full-system --material yellow_pea \
    --particles 5000 --time 360

# Multi-pass recirculation (3 passes, recirculate cyclone-1 fraction)
python examples/run_classification_flow.py \
    --blower-rpm 500 --wheel-rpm 3000 \
    --full-system --material yellow_pea \
    --recirculate cy1 --passes 3 --attrition 0.10

# Wheel-only mode (no venturi/zigzag preclassification)
python examples/run_classification_flow.py \
    --wheel-only --wheel-rpm 4000 --material faba_bean

# Air flow SPH simulation (blower ramp-up, damper dynamics)
python examples/run_air_flow_physics.py \
    --rpm 500 --particles 1000 --time 10 --analyze

# Feed system material flow (gravity, drag, screw conveying)
python examples/run_physics_flow.py \
    --material yellow_pea --particles 5000 --time 180 --pouring

# Visualize assembled geometry (3D interactive viewer)
python examples/visualize_geometry.py --with-preclassification --animate-wheel

# Inspect assembly connections and port alignment
python examples/inspect_assembly.py --validate
```

## Classification System

### With Preclassification (Full System)

```text
Air Supply → Blower → Dampers → Ductwork
                                    ↓
Feed Hopper → Airlock → Screw → Deagglomerator
                                    ↓
                              Venturi Eductor (entrainment)
                                    ↓
                           Zigzag Classifier (pre-separation)
                              ↓              ↓
                          Coarse out    Fines → Wheel Classifier
                                               ↓           ↓
                                           Fine out    Coarse out
                                               ↓
                                    Multi-Cyclone System (3 stages)
                                         ↓         ↓
                                     Collected    Exhaust → Bag Filter
```

### Without Preclassification (Wheel-Only)

```text
Air + Solids → Three-Point Junction → Wheel Classifier → Cyclones → Bag Filter
```

### Physics

- **Venturi**: Bernoulli acceleration entrains solids into the air stream
- **Zigzag**: Counter-current air/gravity separation in baffled channel
- **Wheel Classifier**: Centrifugal force (1,000–5,000 g) at the blade tips provides fine cut (d50 ~ 25 um)
- **Cyclones**: Multi-stage centrifugal separation with grade efficiency
- **Drag**: Schiller-Naumann correlation (Re < 1000) or Haider-Levenspiel (non-spherical)
- **Collisions**: Inelastic particle-wall with restitution and friction
- **Air Flow**: SPH with Poly6 density, Spiky pressure gradient, and XSPH smoothing
- **Blower**: Fan affinity laws with VFD ramp (S-curve startup)

## Materials

Built-in food powder presets with measured size distributions:

| Material   | Source            | Fractions              | Typical d50 |
| ---------- | ----------------- | ---------------------- | ----------- |
| Yellow Pea | *Pisum sativum*   | protein, starch, fiber | 15–80 um    |
| Faba Bean  | *Vicia faba*      | protein, starch, fiber | 12–75 um    |
| Oat        | *Avena sativa*    | protein, starch, fiber | 20–90 um    |

```python
from airclassifier.particles import ParticleMaterial, create_whole_flour_population

# Create a realistic particle population
material, diameters, densities, sphericities, types = \
    create_whole_flour_population(source="yellow_pea", num_particles=10000)
```

## GUI Details

### Assembly Configuration

The **Configure Assembly** dialog lets you set:

- Assembly mode (full system with preclassification, or wheel-only)
- Wheel classifier parameters (RPM, diameter, number of blades)
- Venturi and zigzag geometry overrides
- Cyclone diameters
- Which subsystems to include (feed, air, exhaust)

### Simulation Settings

The **Settings** tab mirrors the CLI parameters:

- Time, dt, output interval
- Particle count and feed rate
- Material source and fraction
- Blower RPM or direct air flow rate
- Turbulence, restitution, friction
- Multi-pass recirculation (passes, fractions, attrition)
- Device (CUDA / CPU)

### 3D Viewport

- **Rendering**: Solid + edge overlay, wireframe, particle display, flow field arrows
- **Views**: Isometric, Front, Back, Left, Right, Top, Bottom
- **Animation**: Mechanical components animate during simulation (blower impeller, damper blades, hopper lid, screw feeder, wheel classifier) using subsidiary physics simulators
- **Cinematic Camera**: Toggle the *Cinematic* button for automatic camera movement:
  - **Orbit** — continuous rotation around the assembly
  - **Showcase** — guided tour of key viewpoints with smooth transitions
  - **Flythrough** — scripted spiral sweep
  - Mouse interaction temporarily pauses the cinematic camera (resumes after 4 s)

### Simulation Lifecycle

1. **Build Full System** — creates `CompleteClassifierAssembly` with all subsystems; separates static mesh from animated parts (no z-fighting)
2. **Run Simulation** — 8 s startup preamble (blower ramp, dampers open, lid opens, wheel spins up), then classification physics on a background thread
3. **Live Updates** — progress bar, KPI cards (sim time, active particles, fines, coarse, separation efficiency), physics-driven animation
4. **Shutdown** — 3 s shutdown animation (dampers close, lid closes, blower ramps down), then results displayed

## Examples

| Script                         | Description                                            |
| ------------------------------ | ------------------------------------------------------ |
| `run_classification_flow.py`   | Full classification simulation with separation stats   |
| `run_air_flow_physics.py`      | SPH air flow through blower, dampers, ductwork         |
| `run_physics_flow.py`          | Gravity-driven material flow through feed system       |
| `visualize_geometry.py`        | Interactive 3D geometry viewer with STL export         |
| `run_viz_simulation.py`        | Live 3D visualization of running simulations           |
| `inspect_assembly.py`          | Assembly connection validation and port alignment      |
| `cfd_cyclone.py`               | CFD-DEM coupled cyclone simulation                     |

## Coordinate System

All geometry uses **Y-up**:

- **X**: Horizontal (width)
- **Y**: Vertical (height) — up
- **Z**: Horizontal (depth)

No coordinate transforms are applied — geometry is authored in Y-up and rendered in Y-up.

## Author

Emmanuel Kwofie

## License

MIT License
