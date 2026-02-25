# GPU-Accelerated Multiphysics Simulation of a Multi-Stage Air Classification System Using NVIDIA Warp

**Technical Note — Multiphysics Simulation Section**

*Emmanuel Kwofie*

---

## Abstract

This technical note documents the multiphysics simulation framework for a dry air classification system designed for legume flour fractionation. The framework implements four tiers of physics fidelity — from analytical fan affinity laws to full CFD-DEM two-way coupling — unified under a common `start()` / `step()` / `get_results()` API. Three domain-specific simulators (feed system, air system, classification system) are coupled through a geometry-driven particle transfer protocol that preserves physical particle diameters across subsystem boundaries. All computationally intensive operations — Lagrangian particle tracking, Smoothed Particle Hydrodynamics (SPH), zone-based drag and separation physics, hash-grid collision detection, and Navier-Stokes pressure projection — execute as NVIDIA Warp GPU kernels with persistent memory allocation and batched kernel launches. The implementation is structured in Python with Warp for GPU compute and NumPy for geometry and host-side data; a dedicated **kinetics** layer provides the canonical force and efficiency laws (drag, gravity, centrifugal, grade efficiency), and **particle** and **fluid** layers supply Warp-accelerated implementations and material/flow-field data. The technical note traces the dependency from the public simulation API down to these physics and kinetics modules (Section 1.3), describes the roles of Python libraries and Warp (Section 2.2), and details the material database, drag correlations, Rankine vortex flow fields, and per-component separation models (zigzag, wheel, cyclone, bag filter). No simulation parameter is hardcoded; all values derive from the parametric geometry assemblies documented in the companion Geometry Modeling technical note.

---

## 1. Introduction

### 1.1 Physical Problem

Air classification separates milled legume flour into protein-enriched fines and starch-enriched coarse fractions by exploiting differences in particle aerodynamic behavior under controlled airflow. The process involves coupled phenomena across multiple length and time scales:

| Scale | Phenomenon | Characteristic Dimension |
|-------|-----------|------------------------|
| Equipment (m) | Blower performance, duct pressure drop, cyclone swirl | 0.1–2 m |
| Component (cm) | Zigzag deflector recirculation, wheel blade passage | 1–20 cm |
| Particle (μm) | Drag, gravity, centrifugal separation, wall collision | 2–500 μm |
| Fluid (μm–cm) | Turbulent eddies, boundary layers, vortex breakdown | 10 μm – 10 cm |

A simulation framework for this system must handle:

1. **Feed handling**: Gravity-driven hopper discharge, rotary airlock metering, screw conveying, and deagglomeration with particle–wall and particle–particle collisions.
2. **Air supply**: Centrifugal blower performance curves, filter pressure drops, damper flow characteristics, and duct friction losses.
3. **Classification**: Particle entrainment in a venturi, multi-stage gravity separation in a zigzag channel, centrifugal separation in a spinning wheel, cyclone collection via swirling flow, and bag filter impaction.
4. **Inter-system coupling**: Particles exiting the feed system must enter the classification system with correct physical diameters and velocities; air flow rates from the blower must set the classification air velocities.

### 1.2 Simulation Architecture

The framework is organized in five layers:

```
┌─────────────────────────────────────────────────────┐
│  Layer 5: Visualization (PyVista + live overlay)     │
├─────────────────────────────────────────────────────┤
│  Layer 4: System Simulators                          │
│  (AirSystemSimulator, FeedSystemSimulator,           │
│   ClassificationFlowPhysicsSimulator)                │
├─────────────────────────────────────────────────────┤
│  Layer 3: Physics Engines                            │
│  (FeedFlowPhysicsSimulator,                          │
│   AirFlowPhysicsSimulator, CFDDEMCoupler)            │
├─────────────────────────────────────────────────────┤
│  Layer 2: Particle & Fluid Modules                   │
│  (WarpParticleSystem, drag models, flow fields,      │
│   Navier-Stokes solver, turbulence models)           │
├─────────────────────────────────────────────────────┤
│  Layer 1: NVIDIA Warp GPU Backend                    │
│  (@wp.kernel, @wp.func, wp.array, wp.HashGrid)      │
└─────────────────────────────────────────────────────┘
```

### 1.3 Trace from Simulation API to Physics and Kinetics

The public simulation API exposes a small set of entry points: **system simulators** (e.g. air system, feed system, classification system) each with a uniform interface `start()`, `step()`, and `get_results()`. Understanding how these connect to the underlying physics and kinetics is essential for reproducibility and extension.

**Layer 4 — System simulators**

- **Air system**: The basic air simulator uses analytical fan affinity laws (flow rate, pressure rise, and power scale with RPM and damper positions); it does not call a separate physics engine. The advanced air simulator is the **air flow physics simulator**, which implements SPH and 1D hydraulics and is invoked when high-fidelity air flow is required.
- **Feed system**: The feed system simulator presented to the user is a thin wrapper. Each call to `start()`, `step()`, or `get_results()` is forwarded to an internal **feed flow physics simulator**. All particle motion, zone transitions, and collisions are implemented there.
- **Classification system**: The classification simulator is the **classification flow physics simulator** directly; there is no separate wrapper. It advances particles through venturi, zigzag, wheel, cyclones, and bag filter using zone-based logic and component-specific separation models.

**Layer 3 — Physics engines**

The physics engines implement the actual equations of motion and constitutive laws:

1. **Feed flow physics simulator**  
   - Maintains particle state (positions, velocities, diameters, zones) on the GPU.  
   - Each time step: run a single **physics kernel** that, per particle, evaluates gravity, drag (Schiller–Naumann or Haider–Levenspiel), zone-specific forces (hopper walls, airlock rotation, screw conveying, deagglomerator rotor), semi-implicit Euler integration, zone-boundary checks, and containment.  
   - Optionally runs a **particle–particle collision kernel** using a hash grid for neighbor search and impulse-based response.  
   - Geometry (hopper, airlock, feeder, deagglomerator, transition ducts) is extracted once from the feed system assembly and passed into the kernels as parameters; no geometry is hardcoded.

2. **Air flow physics simulator**  
   - Combines a 1D hydraulic model (blower curve vs. system resistance, Darcy–Weisbach, filter and damper losses) with an SPH particle phase.  
   - SPH kernels: build hash grid, compute density (Poly6) and pressure (Tait), compute forces (Spiky pressure, Laplacian viscosity), integrate with blower and boundary terms, apply XSPH smoothing.  
   - Blower and duct geometry are extracted from the air system assembly; operating point (flow rate, pressure) is computed from the hydraulic balance and used to steer SPH particles.

3. **Classification flow physics simulator**  
   - Maintains particles in many zones (venturi, zigzag stages, wheel, cyclones, bag filter, collection bins).  
   - Per-zone physics: venturi (continuity-based velocity), zigzag (terminal velocity vs. recirculation-zone velocity, turbulent dispersion), wheel (centrifugal vs. drag force balance, blade collision), cyclone (Rankine vortex field, Rosin–Rammler grade efficiency), bag filter (Stokes-number impaction).  
   - Drag and terminal velocity use the same correlations as the feed system (Schiller–Naumann, Haider–Levenspiel); centrifugal and gravity forces are consistent with the kinetics force laws.  
   - All zone dimensions, cross-sections, and cut sizes are taken from the classification assembly geometry.

4. **CFD–DEM coupler**  
   - Couples an Eulerian fluid solver (Navier–Stokes with projection, advection, diffusion) to a Lagrangian particle system.  
   - Interpolation: fluid velocity at particle position; scatter: particle drag force as momentum source to the grid (atomic add).  
   - Uses the fluid solvers (advection, diffusion, pressure Poisson, projection) and turbulence models; particle side uses the same drag and integration concepts as the other physics engines.

**Layer 2 — Particle, fluid, and kinetics modules**

- **Kinetics**  
  The kinetics layer provides the **force and efficiency laws** used across the framework:  
  - **Drag**: Reynolds number, drag coefficient (Stokes, Schiller–Naumann, Haider–Levenspiel), and drag force as a function of relative velocity, diameter, fluid properties, and sphericity. Implemented as Python functions for validation and as Warp device functions/kernels for GPU use.  
  - **Gravity**: gravity force and acceleration, terminal velocity (Stokes and intermediate Re).  
  - **Centrifugal**: centrifugal force and acceleration, separation number (centrifugal vs. drag).  
  - **Virtual mass**: optional added-mass force for unsteady motion.  
  - **Separation efficiency**: grade efficiency curves, theoretical cut sizes (e.g. Lapple, Barth), Rosin–Rammler form for cyclone collection.  
  These formulas are the single source of the physics; the physics engines either call them (e.g. for post-processing or CPU-side checks) or use Warp kernels that implement the same formulas on the GPU.

- **Particle module**  
  - **Material**: Particle density, sphericity, size distribution (monodisperse, log-normal, Rosin–Rammler, etc.), and food-powder presets (yellow pea, faba bean, oat; protein, starch, fiber, whole). Used to initialize particle diameters, densities, and types.  
  - **Particle system**: Warp-based particle state (positions, velocities, diameters, etc.), time integration (semi-implicit Euler, velocity Verlet), Reynolds number and drag coefficient/acceleration as Warp device functions, centrifugal acceleration helper. Used by physics engines that need a generic particle integrator; the feed and classification engines instead use their own zone-specific kernels that embed the same force laws.  
  - **Drag models**: Warp kernels for Stokes, Schiller–Naumann, and Haider–Levenspiel drag and terminal velocity; these mirror the kinetics drag API for GPU execution.  
  - **Interactions**: Wall collision handler (reflection with restitution and friction) and particle–particle collision handler (hash-grid neighbor search, impulse exchange). The feed flow physics uses inline containment and optional collision kernel; the interaction handlers can be used where a generic wall or collision response is sufficient.

- **Fluid module**  
  - **Flow field**: Analytical cyclone flow (e.g. Rankine combined vortex) with tangential, axial, and radial velocity profiles; parameters from cyclone geometry and inlet conditions. Exposed as a Warp-compatible velocity query so classification (and optionally other) physics can evaluate fluid velocity at particle positions.  
  - **Solvers**: Navier–Stokes (staggered grid, projection method: advection, diffusion, pressure Poisson, projection); used by the CFD–DEM coupler.  
  - **Turbulence**: Smagorinsky LES and k‑ε RANS; subgrid or eddy viscosity is fed into the fluid solver when turbulence is enabled.

**Layer 1 — NVIDIA Warp**

- All particle state (positions, velocities, diameters, zones, activity flags) is stored in `wp.array` on the chosen device (CPU or CUDA).  
- The heavy work (physics kernel, collision kernel, SPH kernels, containment, classification zone logic, fluid advection/diffusion/projection) is implemented as `@wp.kernel` and `@wp.func`.  
- Neighbor search for SPH and particle collisions uses `wp.HashGrid` (build each step, then query per particle).  
- Single synchronization per step: after all kernels for that step are launched, `wp.synchronize()` is called so CPU-side code (e.g. state machine, output, transfer preparation) sees consistent results.

This trace makes clear that **all kinetics (force and efficiency laws) live in the kinetics and particle/fluid layers**, and that the **physics engines assemble geometry, material, and these laws** into domain-specific kernels and stepping logic, with Warp providing the GPU execution and memory model.

### 1.4 Scope of This Note

This note covers:

- **Section 2**: The NVIDIA Warp computational backend, Python library integration, and GPU memory management strategy.
- **Section 3**: Material modeling — food powder properties, size distributions, and drag correlations.
- **Section 4**: The feed system physics engine — zone-based Lagrangian particle tracking through hopper, airlock, screw feeder, and deagglomerator.
- **Section 5**: The air system physics engine — SPH-based air flow simulation with blower performance coupling.
- **Section 6**: The classification physics engine — multi-zone particle transport with component-specific separation models.
- **Section 7**: Fluid dynamics infrastructure — Navier-Stokes solver, turbulence models, and CFD-DEM coupling.
- **Section 8**: Inter-system coupling — the particle transfer protocol and system-level orchestration.
- **Section 9**: Numerical methods — time integration, stability, and validation.

---

## 2. NVIDIA Warp Computational Backend

### 2.1 Role of Warp in the Framework

NVIDIA Warp is a Python framework for high-performance GPU simulation that provides:

- **JIT compilation**: Python functions decorated with `@wp.kernel` and `@wp.func` are compiled to CUDA PTX at first invocation and cached at `~/.cache/warp/<version>/`.
- **Automatic differentiation**: Enables gradient-based optimization (not used in current physics, but available for future inverse design).
- **Spatial data structures**: `wp.HashGrid` for O(1) neighbor queries in particle collision detection.
- **Mesh queries**: `wp.Mesh` with hardware-accelerated ray tracing for SDF-based wall collisions.
- **Device management**: Transparent CPU/CUDA execution via the `device` parameter.

### 2.2 Python Libraries and Package Roles

The multiphysics stack is implemented in Python and relies on a small set of libraries, each with a clear role:

| Role | Library / package | Use in the framework |
|------|-------------------|----------------------|
| **GPU compute** | NVIDIA Warp | All particle and SPH kernels, fluid advection/diffusion/projection, hash grid, device arrays; JIT compilation to CUDA PTX. |
| **Arrays and math** | NumPy | CPU-side geometry extraction (positions, radii, port vectors), assembly traversal, size-distribution sampling, result aggregation; conversion to/from Warp arrays when copying between device and host. |
| **Visualization** | PyVista (VTK) | Real-time 3D rendering of meshes and particle clouds; not used inside the simulation step. |
| **Material and constants** | Built-in + project constants | SI constants (gravity, π, air properties), material presets (density, sphericity, size bounds), Sutherland viscosity; used by kinetics, particle material, and fluid config. |
| **Configuration and I/O** | Standard library / YAML | Loading and saving run configs, output paths; optional YAML for preset configurations. |

No simulation parameter is hardcoded in the physics kernels; geometry is extracted from the assembly (NumPy-based) and fluid/material properties from the shared constants and config objects. Warp is used only for the numerically intensive kernels; control flow, state machines, and coupling logic remain in Python.

### 2.3 Device Detection and Fallback

All simulators detect GPU availability at initialization:

```python
wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"
```

This enables development on CPU-only machines with automatic GPU acceleration when available.

### 2.4 Persistent GPU Memory Allocation

A critical performance design is the **zero-allocation-per-step** strategy. All GPU arrays are pre-allocated at simulator initialization and reused across time steps:

```python
# Allocated once in __init__()
self.positions  = wp.zeros(n, dtype=wp.vec3,   device=device)
self.velocities = wp.zeros(n, dtype=wp.vec3,   device=device)
self.diameters  = wp.zeros(n, dtype=float,     device=device)
self.masses     = wp.zeros(n, dtype=float,     device=device)
self.zones      = wp.zeros(n, dtype=wp.int32,  device=device)
self.is_active  = wp.zeros(n, dtype=wp.int32,  device=device)
```

This avoids the overhead of repeated `cudaMalloc`/`cudaFree` calls and prevents GPU memory fragmentation during long simulations.

### 2.5 Kernel Launch Pattern

Each simulation time step follows a batched kernel launch pattern with a single synchronization point:

```python
def step(self):
    # 1. Launch physics kernel (async)
    wp.launch(physics_kernel, dim=n_particles, inputs=[...], device=device)

    # 2. Launch collision kernel (async, depends on hash grid)
    wp.launch(collision_kernel, dim=n_particles, inputs=[...], device=device)

    # 3. Launch containment kernel (async)
    wp.launch(containment_kernel, dim=n_particles, inputs=[...], device=device)

    # 4. Single synchronization (blocks until all kernels complete)
    wp.synchronize()

    # 5. CPU-side bookkeeping (cheap)
    self.state.time += self.config.dt
    self.state.step += 1
```

The asynchronous kernel launches allow the GPU to pipeline execution, while the single `wp.synchronize()` minimizes CPU–GPU synchronization overhead.

### 2.6 Hash Grid for Neighbor Search

Particle–particle collision detection uses Warp's built-in spatial hash grid:

```python
# Build hash grid from current positions
self.hash_grid = wp.HashGrid(dim_x, dim_y, dim_z)
wp.hash_grid_build(self.hash_grid, positions, n_active)

# In kernel: query neighbors within search radius
query = wp.hash_grid_query(grid, pos_i, search_radius)
while wp.hash_grid_query_next(query):
    j = wp.hash_grid_query_current(query)
    # Process neighbor j
```

Cell dimensions are set to approximately $2.5 \times h$ where $h$ is the smoothing length (for SPH) or the search radius (for DEM collisions), ensuring O(1) average query complexity.

### 2.7 Warp Kernel Inventory

The complete framework contains 40+ Warp kernels across all modules:

| Module | Kernels | `@wp.func` | `@wp.struct` |
|--------|---------|-----------|-------------|
| Particle system | 8 | 5 | 3 |
| Drag models | 2 | 6 | — |
| Wall collisions | 3 | 6 | — |
| Feed flow physics | 2 | 6 | — |
| Air flow physics (SPH) | 5 | 3 | — |
| Classification physics | 5 | 25+ | — |
| CFD-DEM coupling | 3 | — | — |
| Fluid kernels (advection) | 4 | 1 | — |
| Fluid kernels (diffusion) | 3 | — | — |
| Fluid kernels (projection) | 5 | — | — |
| Turbulence models | 4 | — | — |
| Geometry SDF | 3 | 6 | 1 |
| **Total** | **47+** | **58+** | **4** |

**Integration summary.** The multiphysics stack is a Python application that delegates all compute-heavy work to NVIDIA Warp: particle state lives in `wp.array`, and each time step runs a sequence of Warp kernels (physics, collisions, SPH, fluid, etc.) with one synchronization. Python orchestrates geometry extraction (from the assemblies), config and material setup, kernel launch arguments, and result gathering. The kinetics layer defines the force and efficiency formulas; the particle and fluid layers provide Warp implementations of those formulas and of integration/collision/flow; the physics engines (feed, air, classification, CFD–DEM) compose geometry, materials, and these building blocks into domain-specific simulators. Section 1.3 gives the full trace from the public API down to these layers.

---

## 3. Material Modeling

### 3.1 Food Powder Material Database

The `ParticleMaterial` class encapsulates the physical properties of food powder fractions relevant to air classification:

| Material | Density [kg/m³] | Sphericity | d₅₀ [μm] | Size Range [μm] |
|----------|----------------|------------|----------|-----------------|
| Yellow pea protein | 1350 | 0.72 | 10 | 2–30 |
| Yellow pea starch | 1500 | 0.85 | 30 | 12–80 |
| Yellow pea fiber | 1250 | 0.55 | 150 | 50–500 |
| Yellow pea whole flour | 1420 | 0.75 | 35 | 2–500 |
| Faba bean protein | 1380 | 0.70 | 12 | 2–35 |
| Faba bean starch | 1520 | 0.83 | 28 | 10–75 |
| Oat protein | 1310 | 0.68 | 15 | 3–40 |
| Oat starch | 1450 | 0.80 | 25 | 8–65 |

These values are stored as material presets (density, sphericity, size bounds) in the shared constants and material database used by the particle and kinetics layers.

### 3.2 Size Distribution Models

Six distribution types are supported via the `SizeDistributionType` enum:

1. **Monodisperse**: All particles at $d_{50}$.
2. **Uniform**: $U(d_\text{min}, d_\text{max})$.
3. **Normal**: $N(d_\text{mean}, \sigma)$, clipped to physical bounds.
4. **Log-normal**: $\ln(d) \sim N(\mu, \sigma)$ — the default for milled flour.
5. **Rosin-Rammler**: $F(d) = 1 - \exp\left[-(d/d_{63.2})^m\right]$ — standard sieve analysis model.
6. **Gates-Gaudin-Schuhmann**: $F(d) = (d/d_\text{max})^m$ — power-law for comminuted products.

The `sample_diameters(n, seed)` method draws $n$ random diameters from the configured distribution, clipped to $[d_\text{min}, d_\text{max}]$.

### 3.3 Whole Flour Population Model

The `create_whole_flour_population()` factory generates a multi-component particle population reflecting the true composition of milled legume flour:

```python
population = create_whole_flour_population(
    source="yellow_pea",
    num_particles=5000,
    seed=42
)
# Returns arrays: diameters, densities, types (0=protein, 1=starch, 2=fiber)
```

Component fractions are based on proximate analysis data (e.g., yellow pea: ~22% protein, ~55% starch, ~8% fiber by mass), with each fraction's size distribution sampled independently.

### 3.4 Fluid Configuration

The `FluidConfig` dataclass provides temperature- and pressure-dependent air properties:

```python
air = FluidConfig.air_at_stp()        # 20°C, 1 atm
air = FluidConfig.air_at_temperature(  # Sutherland's law
    temperature=60.0,  # °C
    pressure=101325.0  # Pa
)
```

Sutherland's viscosity law for air:

$$\mu(T) = \mu_\text{ref} \cdot \frac{T_\text{ref} + S}{T + S} \cdot \left(\frac{T}{T_\text{ref}}\right)^{3/2}$$

where $\mu_\text{ref} = 1.716 \times 10^{-5}$ Pa·s, $T_\text{ref} = 273.15$ K, $S = 110.4$ K.

### 3.5 Kinetics and Force Laws (Foundation for All Physics Engines)

The kinetics layer provides the **canonical force and efficiency relations** used by the particle and classification physics. These are implemented as Python functions (for unit tests and CPU-side checks) and, where needed, as Warp device functions or kernels for GPU use.

**Drag**

- **Reynolds number**: $\text{Re}_p = \rho_f |\mathbf{v}_f - \mathbf{v}_p| d_p / \mu$.
- **Stokes** ($\text{Re}_p < 0.1$): $C_D = 24/\text{Re}_p$; drag force $\mathbf{F}_d = 3\pi\mu d_p (\mathbf{v}_f - \mathbf{v}_p)$.
- **Schiller–Naumann** (spheres, $\text{Re}_p < 1000$): $C_D = (24/\text{Re}_p)(1 + 0.15\,\text{Re}_p^{0.687})$; used for near-spherical starch granules and default particle drag.
- **Haider–Levenspiel** (non-spherical): $C_D$ as a function of $\text{Re}_p$ and sphericity $\phi$ via correlation coefficients $A(\phi)$, $B(\phi)$, $C(\phi)$, $D(\phi)$; used for protein bodies and irregular shapes.
- **Drag force**: $\mathbf{F}_d = \frac{1}{2} C_D \rho_f A_p |\mathbf{v}_f - \mathbf{v}_p| (\mathbf{v}_f - \mathbf{v}_p)/|\mathbf{v}_f - \mathbf{v}_p|$ with $A_p = \pi d_p^2/4$. Acceleration for integration: $\mathbf{a}_d = \mathbf{F}_d / m_p$.

**Gravity and buoyancy**

- Gravity force $\mathbf{F}_g = m_p \mathbf{g}$; buoyancy-corrected acceleration $\mathbf{a}_g = \mathbf{g}(1 - \rho_f/\rho_p)$ so that terminal velocity and settling in ducts are consistent with the chosen drag model.

**Terminal velocity**

- Stokes: $v_t = d_p^2 (\rho_p - \rho_f) g / (18\mu)$.
- Intermediate Re: iterative or correlation-based terminal velocity is used for zigzag and other zones where $v_t$ is compared to a local air velocity to decide coarse vs. fines.

**Centrifugal**

- Centrifugal force $F_c = m_p \omega^2 r$ (outward); radial drag (e.g. Stokes) $F_d \propto d_p$ inward. The ratio $F_c/F_d$ (or equivalent separation number) determines the wheel classifier cut: particles with $F_c/F_d > 1$ go to coarse, others to fines. The same idea appears in cyclone collection with tangential velocity and residence time.

**Separation efficiency**

- **Grade efficiency** $\eta(d)$: probability that a particle of diameter $d$ is collected. Implemented as a modified Rosin–Rammler form $\eta(d) = 1 - \exp[-0.693(d/d_{50})^2]$ for cyclones.
- **Theoretical cut size** $d_{50}$: Lapple and Barth formulas relate $d_{50}$ to cyclone geometry (inlet dimensions, body diameter, vortex finder), number of turns, and inlet velocity; used to set the cyclone $d_{50}$ in the classification physics from assembly geometry and flow rate.
- **Wheel cut**: $d_{50,\text{wheel}} = \sqrt{18\mu v_r / (\Delta\rho \cdot \omega^2 R)}$ from force balance at the wheel rim; $v_r$ from continuity through blade passage area.

**Virtual mass** (optional)

- Added-mass force for unsteady particle motion in dense flows; available in the kinetics API for use in CFD–DEM or specialized models.

The feed flow physics kernel uses the same drag and gravity relations (inlined in the kernel for zone-specific air velocity); the classification physics uses the same drag, terminal velocity, centrifugal balance, and grade-efficiency forms so that cut sizes and separation curves are consistent across the stack and traceable to these kinetics formulas.

---

## 4. Feed System Physics Engine

### 4.1 Architecture

The feed flow physics simulator implements zone-based Lagrangian particle tracking through four equipment components (hopper, airlock, screw feeder, deagglomerator) connected by three transition regions. All physics parameters are extracted from the feed system assembly geometry via a single geometry-extraction function; dimensions, port positions, angular velocities, and screw pitch come from the assembly, so there are **no magic numbers** in the kernel.

### 4.2 Zone Model

Each particle carries an integer zone identifier that determines which physics and containment rules apply:

| Zone ID | Name | Equipment | Physics |
|---------|------|-----------|---------|
| 0 | HOPPER | Feed hopper | Gravity + cone wall collisions |
| 10 | TRANS_HOPPER_AIRLOCK | Cylindrical connector | Gravity descent, radial containment |
| 1 | AIRLOCK | Rotary airlock | Vane rotation (10% coupling) |
| 11 | TRANS_AIRLOCK_FEEDER | Conical reducer | Gravity + converging wall |
| 2 | FEEDER | Screw feeder | Helical advance + rotation (10% coupling) |
| 12 | TRANS_FEEDER_DEAGG | Cylindrical connector | Gravity descent |
| 3 | DEAGGLOMERATOR | Deagglomerator | High-speed rotor (30% coupling) |
| 4 | EXITED | Below deagg outlet | Free fall + floor collision |

### 4.3 Derived Parameters from Geometry

All operational parameters are computed from the assembly's physical dimensions and operating speeds:

**Angular velocities:**
$$\omega = \frac{2\pi \cdot \text{RPM}}{60}$$

**Screw feeder axial velocity:**
$$v_\text{axial} = \frac{p \cdot \text{RPM}}{60}$$

where $p$ is the screw pitch extracted from `ScrewFeederParams.screw_pitch`.

**Volumetric flow rates:**
$$\dot{V}_\text{airlock} = V_\text{pocket} \times n_\text{vanes} \times \frac{\text{RPM}}{60} \times f_\text{fill}$$

**Transition connector geometry** (lengths, diameters, positions) is computed from the actual port positions of mating components.

### 4.4 Main Physics Kernel

The `physics_flow_kernel` is a single Warp kernel (~830 lines) that processes all particles in parallel. For each active particle, it:

1. **Identifies the current zone** from the particle's zone ID.
2. **Computes gravity acceleration** with buoyancy: $a_g = g(1 - \rho_f/\rho_p)$.
3. **Computes drag acceleration** using Schiller-Naumann: $C_D = \frac{24}{\text{Re}}(1 + 0.15\,\text{Re}^{0.687})$.
4. **Applies zone-specific forces**:
   - **Hopper**: Height-dependent wall radius (cylinder → cone), wall collisions.
   - **Airlock**: 10% tangential coupling to vane rotation, radial containment.
   - **Feeder**: Axial conveying force proportional to feeder RPM, 10% rotational coupling.
   - **Deagglomerator**: 30% coupling to high-speed rotor (scaled by $r/r_\text{rotor}$), capped at 100 m/s².
5. **Integrates velocity and position** (semi-implicit Euler).
6. **Checks zone transition criteria** based on spatial position relative to component boundaries.
7. **Enforces containment** — hard radial/axial bounds per zone with inelastic wall reflection.

### 4.5 Zone Transition Logic

Zone transitions are triggered by spatial proximity to outlet boundaries:

**Hopper → Trans_Hopper_Airlock (Zone 0 → 10):**
- Particle position $y < y_\text{outlet} + 5r_p$ AND radial distance $r < r_\text{outlet} + 0.5r_p$.
- Discharge gate must be open (`discharge_open = 1`).

**Trans_Airlock_Feeder → Feeder (Zone 11 → 2):**
- Particle position $y \leq y_\text{trans\_end} + 3r_p$.
- Initial velocity set to 30% of feeder axial speed with 50% damping on vertical and lateral components.

**Feeder → Trans_Feeder_Deagg (Zone 2 → 12):**
- Particle in outlet region: bottom 20% of trough AND within outlet radius.
- Velocity reset: 10% axial, minimum −0.5 m/s vertical, 10% lateral.

**Deagglomerator → Exited (Zone 3 → 4):**
- Particle position $y < y_\text{deagg\_outlet}$ AND within outlet radius.
- Tangential velocity from rotor preserved at 50%.

### 4.6 Transition Connector Physics

The three transition regions model the short ducts connecting equipment:

1. **Hopper → Airlock** (Zone 10): 15 mm cylindrical, vertical descent.
2. **Airlock → Feeder** (Zone 11): 120 mm conical reducer with 12° half-angle. The wall normal accounts for the cone inclination:
   $$\hat{n}_\text{wall} = \cos(\alpha)\,\hat{r} - \sin(\alpha)\,\hat{y}$$
3. **Feeder → Deagglomerator** (Zone 12): 20 mm cylindrical, vertical descent.

### 4.7 Particle Collision Detection

The `particle_collision_kernel` implements impulse-based collision response using the hash grid:

For each particle pair $(i, j)$ where distance $d_{ij} < r_i + r_j$ and approaching ($v_n < 0$):

$$j_\text{impulse} = \frac{-(1 + e)\,v_n}{1/m_i + 1/m_j}$$

$$\Delta\mathbf{v}_i = \frac{j_\text{impulse}}{m_i}\,\hat{\mathbf{n}}_{ij}$$

where $e$ is the coefficient of restitution and $\hat{\mathbf{n}}_{ij}$ is the collision normal.

### 4.8 Simulation Workflow (State Machine)

The feed system follows a five-phase workflow:

```
IDLE → POURING → SETTLING → FLOWING → COMPLETED
```

1. **IDLE**: System initialized, lid closed.
2. **POURING**: Lid opens to target angle (smooth angular velocity animation), particles generated in a circular stream above the hopper at $v_0 = -\sqrt{2g \cdot h_\text{pour}}$, with log-normal diameter distribution.
3. **SETTLING**: Lid closes, particles settle under gravity with collision damping. Transition to FLOWING when mean velocity < 0.05 m/s or maximum settling time reached.
4. **FLOWING**: Discharge gate opens, components ramp to operating RPM, particles flow through airlock → feeder → deagglomerator.
5. **COMPLETED**: All particles either exited or deactivated.

### 4.9 Dual Particle Scale Architecture

The feed system maintains two diameter scales for each particle:

- **Visual diameter** (~15 mm): Sized so that 5000 particles visually fill the hopper volume. Used for PyVista rendering.
- **Physical diameter** (~50 μm): Actual flour particle size used for drag, terminal velocity, and classification physics.

A linear scale factor bridges them: $d_\text{visual} = d_\text{physical} \times f_\text{scale}$.

When particles transfer to the classification system, **physical diameters are used**:

```python
transfer_data = feed_sim.get_particle_data_for_transfer()
# transfer_data['diameters'] contains PHYSICAL (μm-scale) diameters
classification_sim.inject_particles_from_feed(transfer_data)
```

---

## 5. Air System Physics Engine

### 5.1 Two Fidelity Levels

The air system offers two simulation approaches:

| Level | Simulator | Physics | GPU |
|-------|-----------|---------|-----|
| Basic | `AirSystemSimulator` | Fan affinity laws | No |
| Advanced | `AirFlowPhysicsSimulator` | SPH + 1D hydraulics | Yes |

### 5.2 Fan Affinity Laws (Basic Simulator)

The basic simulator implements classical affinity laws for centrifugal fans:

$$Q = Q_\text{design} \cdot \frac{N}{N_\text{design}} \cdot \prod_i \phi_i$$

$$\Delta p = \Delta p_\text{design} \cdot \left(\frac{N}{N_\text{design}}\right)^2$$

$$W = W_\text{design} \cdot \left(\frac{N}{N_\text{design}}\right)^3$$

where $Q$ is volumetric flow rate, $N$ is blower RPM, $\phi_i$ is the $i$-th damper position (0–1), $\Delta p$ is pressure rise, and $W$ is shaft power. Blower RPM ramps linearly during startup/shutdown.

### 5.3 Blower Performance Curve (Advanced Simulator)

The advanced simulator computes the operating point as the intersection of the blower characteristic curve and the system resistance curve:

**Blower curve** (parabolic approximation):
$$\Delta p_\text{blower}(Q) = \Delta p_\text{shutoff} \cdot \left(1 - \frac{Q^2}{Q_\text{max}^2}\right)$$

where $\Delta p_\text{shutoff} = \Delta p_\text{design} \cdot (N/N_\text{design})^2$ and $Q_\text{max} = Q_\text{design} \cdot (N/N_\text{design})$.

**System resistance curve:**
$$\Delta p_\text{system}(Q) = K_\text{total} \cdot Q^2$$

where $K_\text{total}$ is the sum of all component loss coefficients.

**Operating point** (analytical intersection):
$$Q_\text{op} = \sqrt{\frac{\Delta p_\text{shutoff}}{K_\text{total} + \Delta p_\text{shutoff}/Q_\text{max}^2}}$$

This is solved iteratively (6 iterations with 0.5 relaxation) to account for flow-dependent losses.

### 5.4 Component Pressure Drop Models

**Duct friction** (Darcy-Weisbach with Swamee-Jain friction factor):

$$\Delta p = f \cdot \frac{L}{D_h} \cdot \frac{\rho v^2}{2}, \quad f = \frac{0.25}{\left[\log_{10}\!\left(\frac{\epsilon}{3.7D} + \frac{5.74}{\text{Re}^{0.9}}\right)\right]^2}$$

**Filter** (media resistance + dynamic losses):

$$\Delta p_\text{filter} = R_\text{media} \cdot v_\text{face} + \frac{1}{2}\rho v_\text{face}^2 \cdot K_\text{entry}$$

where $R_\text{media}$ depends on efficiency class (50 Pa·s/m for G4 through 2000 Pa·s/m for HEPA).

**Damper** (butterfly valve characteristic):

$$K_\text{damper}(\phi) = 0.3 + 50 \cdot (1 - \sin^2(\phi \cdot \pi/2))$$

with $K = 1000$ for $\phi < 0.05$ (nearly closed) and $K = 0.3$ for $\phi > 0.95$ (fully open).

**Venturi system resistance:**

$$K_\text{venturi} = \frac{\rho}{2}\left(\frac{1}{A_\text{throat}^2} - \frac{1}{A_\text{inlet}^2}\right)$$

### 5.5 SPH Air Flow Simulation

The advanced simulator uses Smoothed Particle Hydrodynamics to visualize and compute 3D air flow through the duct system.

#### 5.5.1 SPH Formulation

**Density estimation** (Poly6 kernel):

$$\rho_i = \sum_j m_j \, W_\text{poly6}(|\mathbf{r}_i - \mathbf{r}_j|, h)$$

$$W_\text{poly6}(r, h) = \frac{315}{64\pi h^9}(h^2 - r^2)^3, \quad r < h$$

**Pressure** (Tait equation of state for weakly compressible flow):

$$p = \frac{c_s^2 \rho_0}{\gamma}\left[\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right], \quad \gamma = 7$$

where $c_s = 50$ m/s is the artificial speed of sound (chosen for stability, not physical accuracy).

**Pressure force** (Spiky kernel gradient, symmetric formulation):

$$\mathbf{F}_\text{pressure} = -\sum_j m_j \left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2}\right) \nabla W_\text{spiky}(\mathbf{r}_{ij}, h)$$

$$\nabla W_\text{spiky}(\mathbf{r}, h) = -\frac{45}{\pi h^6}(h - r)^2 \frac{\mathbf{r}}{r}$$

**Viscosity force** (Laplacian kernel):

$$\mathbf{F}_\text{viscosity} = \mu \sum_j \frac{m_j}{\rho_j}(\mathbf{v}_j - \mathbf{v}_i)\,\nabla^2 W_\text{visc}(\mathbf{r}_{ij}, h)$$

$$\nabla^2 W_\text{visc}(r, h) = \frac{45}{\pi h^6}(h - r)$$

**XSPH velocity smoothing** (coherent motion):

$$\mathbf{v}_i^* = \mathbf{v}_i + \varepsilon \sum_j \frac{2m_j}{\rho_i + \rho_j}(\mathbf{v}_j - \mathbf{v}_i)\,W(r_{ij}, h)$$

where $\varepsilon = 0.1$ is the XSPH factor.

#### 5.5.2 Boundary Containment

SPH particles are constrained to the physical duct geometry through five containment segments:

1. **Inlet duct**: Cylindrical containment in $YZ$ plane around duct centerline.
2. **90° Elbow**: Toroidal containment — particles follow the arc centerline with cylindrical cross-section.
3. **Vertical duct**: Cylindrical containment in $XY$ plane.
4. **Blower scroll**: Cylindrical containment in the impeller region, transitioning to rectangular at the outlet.
5. **Outlet duct**: Cylindrical containment downstream of the blower.

Out-of-bounds particles are respawned at the filter inlet using Warp's built-in RNG (`wp.rand_init`, `wp.randf`).

#### 5.5.3 Blower Acceleration Model

SPH particles inside the blower scroll receive centrifugal and tangential forces:

$$a_\text{centrifugal} = \omega^2 r \quad (\text{outward})$$
$$a_\text{tangential} = 0.6 \cdot \omega r \quad (\text{tangential drag from impeller})$$

Near the scroll outlet ($+X$ side), an additional outlet acceleration:

$$a_\text{outlet} = v_\text{tip} \cdot \frac{r}{R_\text{scroll}} \cdot (1 + 2\cos\theta_\text{outlet})$$

directs particles toward the discharge duct.

#### 5.5.4 Hydraulic-SPH Coupling

The 1D hydraulic solver (blower curve intersection) provides a target flow velocity:

$$v_\text{target} = \frac{Q_\text{operating}}{A_\text{duct}}$$

SPH particles are gently steered toward this velocity using a relaxation factor of 0.05:

$$a_\text{hydraulic} = 0.05 \cdot \frac{v_\text{target} - v_\text{SPH}}{\Delta t}$$

This ensures the SPH visualization is consistent with the global flow balance while allowing local flow features (eddies, secondary flows) to develop naturally.

#### 5.5.5 SPH Kernel Execution Sequence

Each time step launches five kernels in sequence:

```
1. wp.hash_grid_build()                  — Rebuild spatial hash
2. compute_sph_density_pressure_kernel   — Poly6 density → Tait pressure
3. compute_sph_forces_kernel             — Spiky pressure + Laplacian viscosity
4. integrate_sph_air_kernel              — Velocity integration + blower + boundaries
5. xsph_correction + apply_xsph_kernel  — Coherent velocity smoothing
```

---

## 6. Classification Physics Engine

### 6.1 Architecture

The `ClassificationFlowPhysicsSimulator` implements the most complex physics in the framework: multi-zone particle transport through up to 20 equipment zones with component-specific separation models. All geometry is extracted from the `ClassificationSystemAssembly` via `extract_geometry()`.

### 6.2 Zone Model

Particles carry an integer zone ID (0–99) that encodes both their spatial location and collection state:

**Active zones (particles in motion):**

| Zone | Equipment | Characteristic Physics |
|------|-----------|----------------------|
| 0–2 | Venturi (inlet, throat, divergent) | Continuity-driven acceleration |
| 10 | Duct (venturi → zigzag) | Terminal velocity gate |
| 20–22 | Zigzag (entry, stages, fines) | Deflector plate recirculation |
| 23 | Zigzag coarse outlet | Gravity descent |
| 34 | Wheel classifier housing | Centrifugal force balance |
| 35 | Wheel fines outlet | Axial exit through hub |
| 36 | Wheel coarse hopper | Conical gravity descent |
| 40–41 | Elbow + duct to cyclones | Inertial transport |
| 50–52 | Cyclone stages (primary, secondary, tertiary) | Rankine vortex + grade efficiency |
| 60–61 | Elbow + duct to bag filter | Expansion transport |
| 70 | Bag filter | Inertial impaction |

**Collection zones (particles at rest):**

| Zone | Collected Fraction | Product |
|------|-------------------|---------|
| 30 | Zigzag coarse | Starch-rich (>100 μm) |
| 37 | Wheel coarse | Starch-rich (25–50 μm) |
| 55 | Cyclone 1 dust | Mixed coarse |
| 56 | Cyclone 2 dust | Mixed intermediate |
| 57 | Cyclone 3 dust | Protein-rich fines |
| 75 | Bag filter dust | Ultrafine protein |
| 80 | Clean air exit | Escaped (< 1 μm) |

### 6.3 Venturi Entrainment Physics

Air velocity in the venturi follows the continuity equation:

$$v(x) = v_\text{inlet} \cdot \left(\frac{D_\text{inlet}}{D(x)}\right)^2$$

where $D(x)$ is the local diameter at axial position $x$, varying linearly through convergent, constant through throat, and linearly through divergent sections.

At the throat, the static pressure drop creates suction at the solids inlet, entraining particles from the feed system.

### 6.4 Zigzag Classifier Separation Model

The zigzag classifier separates particles through competition between gravity settling and aerodynamic drag at each deflector stage.

#### 6.4.1 Deflector Plate Zones

Each stage creates three flow regions behind the deflector plate:

1. **Throat zone**: Constricted passage between plate tip and opposite wall.
   $$v_\text{throat} = v_\text{bulk} \cdot \frac{w_\text{channel}}{w_\text{throat}}$$

2. **Separation zone**: Recirculation region behind the plate where velocity is reduced.
   $$v_\text{zone} = v_\text{bulk} \cdot \eta_\text{velocity}$$
   where $\eta_\text{velocity} = 0.2$–$0.4$ is the velocity ratio in the recirculation zone.

3. **Transport zone**: Straight channel segment at bulk velocity.

#### 6.4.2 Separation Criterion

A particle is classified as "coarse" if its terminal settling velocity exceeds the local upward air velocity in the separation zone:

$$v_t > v_\text{zone} \implies \text{coarse (falls to next stage)}$$
$$v_t < v_\text{zone} \implies \text{fines (entrained upward)}$$

The theoretical cut size for the zigzag is:

$$d_{50,\text{zigzag}} = \sqrt{\frac{18\,\mu\,v_\text{zone}}{g\,(\rho_p - \rho_f)}}$$

#### 6.4.3 Turbulent Dispersion

Gaussian velocity fluctuations model turbulence in the separation zones:

$$\mathbf{v}_\text{turbulent} = \mathbf{v}_\text{mean} + I \cdot |\mathbf{v}_\text{mean}| \cdot \boldsymbol{\xi}$$

where $I = 0.25$ is the turbulence intensity and $\boldsymbol{\xi} \sim N(0, 1)^3$.

### 6.5 Wheel Classifier Centrifugal Separation

The wheel (turbine) classifier is the critical second-stage separator, operating at 1000–8000 RPM.

#### 6.5.1 Force Balance

At the wheel rim, particles experience competing centrifugal and drag forces:

**Centrifugal (outward):**
$$F_c = m_p \omega^2 r = \frac{\pi}{6}d^3 \rho_p \omega^2 r \quad \propto d^3$$

**Drag (inward, from radial air flow):**
$$F_d = 3\pi\mu d |v_r| \quad \propto d \quad (\text{Stokes regime})$$

The force ratio determines the separation outcome:

$$\frac{F_c}{F_d} = \frac{d^2 \rho_p \omega^2 r}{18\mu |v_r|}$$

- $F_c/F_d > 1$: Particle rejected outward → **coarse** (starch).
- $F_c/F_d < 1$: Particle drawn inward through blades → **fines** (protein) → to cyclones.

#### 6.5.2 Theoretical Cut Size

Setting $F_c = F_d$ yields the wheel cut diameter:

$$d_{50,\text{wheel}} = \sqrt{\frac{18\,\mu\,v_r}{\Delta\rho \cdot \omega^2 \cdot R}}$$

where $v_r = Q / A_\text{blade\_passage}$ is the radial air velocity through the blade gaps.

For typical pilot-scale conditions (200 mm wheel, 8000 RPM, 3000 m³/h):
$$d_{50,\text{wheel}} \approx 25\ \mu\text{m}$$

#### 6.5.3 Blade Collision Model

Particles interacting with rotating blades experience impulsive collisions:

```python
check_wheel_blade_collision(pos, center, num_blades, thickness,
                           R_outer, R_hub, omega, time) → collision_normal
```

The blade positions rotate with time: $\theta_\text{blade}(t) = \theta_0 + \omega t + 2\pi k / n_\text{blades}$.

### 6.6 Cyclone Separation Model

#### 6.6.1 Rankine Combined Vortex

The velocity field inside each cyclone follows a Rankine combined vortex model with three components:

**Tangential velocity:**
$$v_\theta(r) = \begin{cases} \omega_\text{core} \cdot r & r \leq r_\text{core} \quad (\text{forced vortex}) \\ v_{\theta,\text{max}} \cdot (r_\text{core}/r)^n & r > r_\text{core} \quad (\text{free vortex}) \end{cases}$$

where $n \approx 0.7$ is the vortex exponent (between ideal free vortex $n=1$ and real losses) and $r_\text{core} \approx 0.4 R_\text{vf}$.

**Axial velocity:**
$$v_z(r) = \begin{cases} v_{z,\text{inner}} \cdot \left[1 - \left(\frac{r}{r_\text{vf}}\right)^2\right] & r \leq r_\text{vf} \quad (\text{upward, inner vortex}) \\ -v_{z,\text{outer}} \cdot \left(\frac{r - r_\text{vf}}{R - r_\text{vf}}\right) & r > r_\text{vf} \quad (\text{downward, outer vortex}) \end{cases}$$

**Radial velocity:**
$$v_r(r) = -\frac{Q}{2\pi r H_\text{eff}} \quad (\text{inward drift})$$

#### 6.6.2 Grade Efficiency Curve

Cyclone collection probability follows a modified Rosin-Rammler function:

$$\eta(d) = 1 - \exp\!\left[-0.693\left(\frac{d}{d_{50}}\right)^2\right]$$

The Lapple cut size for each cyclone stage:

$$d_{50} = \sqrt{\frac{9\,\mu\,W}{2\pi\,N_s\,v_\text{inlet}\,(\rho_p - \rho_f)}}$$

where $W$ is the inlet width and $N_s$ is the number of effective spiral turns.

#### 6.6.3 Collection Check

At each time step, particles in a cyclone zone are tested against the grade efficiency:

1. Compute local $d_{50}$ from cyclone geometry and inlet velocity.
2. Evaluate $\eta(d_p)$ for the particle's physical diameter.
3. Draw a uniform random number $u \in [0, 1]$.
4. If $u < \eta(d_p) \cdot \Delta t / \tau_\text{residence}$: particle collected (zone → dust zone).

### 6.7 Bag Filter Collection

Bag filter collection uses an inertial impaction model:

$$\text{Stk} = \frac{\rho_p d_p^2 v_\text{face}}{18\mu D_\text{fiber}}$$

Collection probability increases with Stokes number — larger, denser particles are more efficiently captured by impaction on filter fibers. The model achieves >99.9% efficiency for particles >1 μm.

### 6.8 Two Classification Topologies

**With preclassification** (`use_preclassification=True`):
- Particles enter at Zone 0 (venturi solids inlet).
- Zigzag removes coarse material before the wheel sees it.
- Reduces wheel loading and improves classification sharpness.

**Wheel-only** (`use_preclassification=False`):
- Particles enter at Zone 34 (wheel housing directly).
- No venturi or zigzag — air and solids merge at a three-point junction.
- Simpler equipment but higher wheel loading.

The topology is selected via `ClassificationSystemParams.use_preclassification` and propagates to both the geometry assembly and the physics simulator.

### 6.9 Bypass Flow

A configurable bypass ratio allows a fraction of the air to bypass the venturi and zigzag:

$$Q_\text{class} = Q_\text{total} \cdot (1 - \beta), \quad Q_\text{bypass} = Q_\text{total} \cdot \beta$$

The bypass merges before the cyclone train, so cyclones always see the full air flow. This is useful for tuning the zigzag cut without affecting downstream collection.

---

## 7. Fluid Dynamics Infrastructure

### 7.1 Navier-Stokes Solver

The `NavierStokesSolver` implements the incompressible Navier-Stokes equations on a staggered Cartesian grid using the projection method:

**Governing equations:**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

**Projection method (fractional step):**

1. **Advection**: $\mathbf{u}^* = \text{SemiLagrangian}(\mathbf{u}^n, \Delta t)$ — backtrace streamlines and interpolate.
2. **Diffusion**: $\mathbf{u}^{**} = \mathbf{u}^* + \nu\nabla^2\mathbf{u}^* \Delta t$ — explicit or Jacobi-implicit.
3. **Body forces**: $\mathbf{u}^{***} = \mathbf{u}^{**} + \mathbf{f}\Delta t$ — gravity, particle momentum source.
4. **Pressure Poisson**: $\nabla^2 p = \frac{\rho}{\Delta t}\nabla \cdot \mathbf{u}^{***}$ — solved via Jacobi or SOR iterations.
5. **Projection**: $\mathbf{u}^{n+1} = \mathbf{u}^{***} - \frac{\Delta t}{\rho}\nabla p$ — enforce divergence-free.

### 7.2 Advection Schemes

Three advection schemes are implemented as Warp kernels:

| Scheme | Order | Stability | Use Case |
|--------|-------|-----------|----------|
| Semi-Lagrangian | 1st | Unconditionally stable | Default, fast |
| Upwind | 1st | CFL-limited | Simple flows |
| MacCormack | 2nd | CFL-limited | Low dissipation |

### 7.3 Turbulence Models

**Smagorinsky LES** (subgrid-scale viscosity):

$$\nu_t = (C_s \Delta)^2 |\bar{S}|, \quad |\bar{S}| = \sqrt{2\bar{S}_{ij}\bar{S}_{ij}}$$

where $C_s = 0.1$ and $\Delta$ is the filter width (grid spacing).

**k-ε RANS** (two-equation model):

$$\nu_t = C_\mu \frac{k^2}{\varepsilon}, \quad C_\mu = 0.09$$

with transport equations for turbulent kinetic energy $k$ and dissipation rate $\varepsilon$:

$$\frac{\partial k}{\partial t} = P_k - \varepsilon + \nabla \cdot \left(\frac{\nu_t}{\sigma_k}\nabla k\right)$$

$$\frac{\partial \varepsilon}{\partial t} = \frac{\varepsilon}{k}(C_1 P_k - C_2 \varepsilon) + \nabla \cdot \left(\frac{\nu_t}{\sigma_\varepsilon}\nabla \varepsilon\right)$$

where $C_1 = 1.44$, $C_2 = 1.92$, $\sigma_k = 1.0$, $\sigma_\varepsilon = 1.3$.

### 7.4 CFD-DEM Two-Way Coupling

The `CFDDEMCoupler` manages bidirectional momentum exchange between the fluid (Eulerian grid) and particles (Lagrangian):

**Fluid → Particles** (interpolation):
$$\mathbf{u}_f(\mathbf{x}_p) = \text{TrilinearInterp}(\mathbf{U}_\text{grid}, \mathbf{x}_p)$$

**Particles → Fluid** (momentum source):
$$\mathbf{S}_f = -\sum_p \frac{\mathbf{F}_{\text{drag},p}}{V_\text{cell}} = -\sum_p \frac{m_p(\mathbf{u}_f - \mathbf{v}_p)/\tau_p}{V_\text{cell}}$$

where $\tau_p = m_p/(3\pi\mu d C_D \text{Re}/24)$ is the particle relaxation time.

The atomic scatter kernel ensures thread-safe accumulation of particle momentum sources onto the grid:

```python
@wp.kernel
def compute_particle_momentum_source(positions, velocities, masses,
                                     fluid_velocity_grid, momentum_source_grid,
                                     ...):
    tid = wp.tid()
    # Interpolate fluid velocity to particle position
    u_f = trilinear_interp(fluid_velocity_grid, pos)
    # Compute drag force
    F_drag = compute_drag(vel_p, u_f, diameter, ...)
    # Scatter to grid (atomic)
    wp.atomic_add(momentum_source_grid, ix, iy, iz, -F_drag / V_cell)
```

**Coupling modes:**
- **ONE_WAY**: Only fluid → particles. Appropriate for dilute suspensions ($\phi_v < 10^{-4}$).
- **TWO_WAY**: Full bidirectional coupling. Required for dense loading (classification system loading ratios up to $\mu = 2$).

---

## 8. Inter-System Coupling

### 8.1 Particle Transfer Protocol

When particles exit the feed system (Zone 4 = EXITED), they can be transferred to the classification system:

```python
# 1. Extract exited particles from feed system
transfer_data = feed_sim.get_particle_data_for_transfer()
# Returns: {
#   'positions': np.ndarray,     # World coordinates
#   'velocities': np.ndarray,    # Exit velocities
#   'diameters': np.ndarray,     # PHYSICAL (μm-scale) diameters
#   'densities': np.ndarray,     # Per-particle density
#   'types': np.ndarray,         # 0=protein, 1=starch, 2=fiber
#   'outlet_position': np.ndarray,  # Deagg outlet position
#   'outlet_direction': np.ndarray, # Flow direction
# }

# 2. Inject into classification system
n_injected = classification_sim.inject_particles_from_feed(
    transfer_data,
    offset=venturi_solids_inlet - deagg_outlet  # Position offset
)
```

The critical invariant is that **physical diameters** (not visual diameters) are transferred. The classification physics operates on the real particle size to compute drag, terminal velocity, and separation probabilities.

### 8.2 Air Flow Coupling

The air flow rate from the blower determines the classification air velocities:

```python
# From air system results
Q_m3_s = air_results['volume_flow_rate_m3_s']

# Build classification config from air and feed results
config = ClassificationFlowConfig.from_air_and_feed_results(
    air_result=air_results,
    feed_result=feed_results,
    classification_assembly=classification_assembly,
)
# config.air_flow_rate_m3s is set from air system operating point
```

The classification simulator then computes zone velocities from continuity:

$$v_\text{venturi\_inlet} = \frac{Q}{A_\text{venturi\_inlet}}$$
$$v_\text{zigzag} = \frac{Q}{A_\text{zigzag\_channel}}$$
$$v_\text{cyclone\_inlet} = \frac{Q}{A_\text{cyclone\_inlet}}$$

### 8.3 System-Level Orchestration

A **system-level coupling layer** provides functions that combine results from multiple subsystems so that classification and air systems see consistent flow and pressure:

**Feed + classification coupling**

- **Venturi physics from air and feed**: Given the air system operating point (flow rate, pressure) and feed system state (mass flow, outlet position), a single function computes the venturi state used by the classification physics: throat velocity (from continuity and throat area), static pressure drop (from Bernoulli and area ratio), and entrainment ratio. The classification simulator then uses these to set zone velocities and solids loading at the venturi solids inlet.

**Air + ductwork coupling**

- **Blower operating point**: The actual blower RPM, volumetric flow rate, and pressure rise are determined by the intersection of the blower characteristic curve (pressure vs. flow at that RPM) and the system resistance curve (pressure drop vs. flow through filter, dampers, ducts, and any downstream venturi/classification resistance). A coupling function computes this operating point so that the air system simulator reports the same flow rate that the classification simulator uses for continuity-based zone velocities.

---

## 9. Numerical Methods

### 9.1 Time Integration

All particle systems use semi-implicit (symplectic) Euler integration:

$$\mathbf{v}^{n+1} = \mathbf{v}^n + \mathbf{a}^n \Delta t$$
$$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1} \Delta t$$

This is first-order accurate but unconditionally stable for conservative forces, making it suitable for the stiff particle–wall interactions in the feed system.

The `WarpParticleSystem` additionally supports velocity Verlet integration:

$$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^n \Delta t + \frac{1}{2}\mathbf{a}^n \Delta t^2$$
$$\mathbf{v}^{n+1} = \mathbf{v}^n + \frac{1}{2}(\mathbf{a}^n + \mathbf{a}^{n+1})\Delta t$$

which provides second-order accuracy and better energy conservation, at the cost of an additional force evaluation per step.

### 9.2 Stability Controls

Several mechanisms prevent numerical instability:

1. **Acceleration clamping**: Maximum 500 m/s² for general forces, 100 m/s² for deagglomerator rotor coupling, 50 m/s² for feeder rotation.
2. **Velocity clamping**: Maximum 20 m/s for feed system particles.
3. **SPH velocity limit**: $v_\text{max} = \min(0.5 c_s, 0.1 h/\Delta t)$ for air particles.
4. **CFL condition** (SPH): $\Delta t < 0.4 h / (c_s + v_\text{max})$.
5. **Containment enforcement**: Hard radial and axial bounds per zone, applied post-integration.
6. **Escape detection**: Particles more than 2 m outside system bounds are deactivated.

### 9.3 Typical Time Steps

| Simulator | Default $\Delta t$ | Typical Duration | Total Steps |
|-----------|-------------------|-----------------|-------------|
| Air (basic) | 1 ms | 60 s | 60,000 |
| Air (SPH) | 1 ms | 10 s | 10,000 |
| Feed | 0.5 ms | 20 s | 40,000 |
| Classification | 1 ms | 5 s | 5,000 |
| CFD-DEM | 10 μs | 1 s | 100,000 |

### 9.4 Drag Model Selection

Two drag correlations are available, selected based on particle sphericity:

**Schiller-Naumann** ($\phi \geq 0.99$, i.e., spheres):

$$C_D = \frac{24}{\text{Re}_p}\left(1 + 0.15\,\text{Re}_p^{0.687}\right), \quad \text{Re}_p < 1000$$

$$C_D = 0.44, \quad \text{Re}_p \geq 1000$$

**Haider-Levenspiel** ($\phi < 0.99$, non-spherical):

$$C_D = \frac{24}{\text{Re}_p}\left(1 + A\,\text{Re}_p^B\right) + \frac{C}{1 + D/\text{Re}_p}$$

where $A, B, C, D$ are functions of sphericity $\phi$:

$$A = \exp(2.3288 - 6.4581\phi + 2.4486\phi^2)$$
$$B = 0.0964 + 0.5565\phi$$
$$C = \exp(4.905 - 13.8944\phi + 18.4222\phi^2 - 10.2599\phi^3)$$
$$D = \exp(1.4681 + 12.2584\phi - 20.7322\phi^2 + 15.8855\phi^3)$$

This is critical for food powders where particle shapes range from near-spherical starch granules ($\phi \approx 0.85$) to highly irregular protein bodies ($\phi \approx 0.70$).

---

## 10. Concluding Remarks and Directions for Future Manuscripts

### 10.1 Summary of Contributions

This multiphysics simulation framework provides:

1. **Four-tier fidelity hierarchy**: From analytical fan laws (millisecond evaluation) to full CFD-DEM (research-grade, hours per simulation), allowing selection of the appropriate physics level for each application.

2. **Zone-based Lagrangian tracking**: A unified particle physics kernel where zone-specific forces, wall interactions, and transition criteria are encoded in a single GPU kernel launch — eliminating the overhead of multiple kernel launches per component.

3. **Geometry-driven parameterization**: All physics parameters (angular velocities, flow areas, transition geometries, cut sizes) derived from the parametric geometry assemblies with zero hardcoded values.

4. **Physical particle transfer protocol**: Preserving micron-scale physical diameters across subsystem boundaries while maintaining separate visual-scale rendering diameters.

5. **Comprehensive food powder material database**: Experimentally-informed density, sphericity, and size distribution data for legume flour fractions (yellow pea, faba bean, oat) with appropriate drag models for non-spherical particles.

6. **SPH-based air flow visualization**: Weakly compressible SPH with Tait equation of state, coupled to a 1D hydraulic solver for the blower operating point, providing physically-grounded 3D flow visualization.

### 10.2 Potential Manuscript Topics

1. **Validation of GPU-accelerated Lagrangian particle tracking against experimental fractionation data**: Comparing simulated grade efficiency curves and protein recovery ratios against experimental air classification trials with yellow pea flour.

2. **Comparative study of zigzag + wheel vs. wheel-only classification topologies**: Using the two built-in topologies to quantify the benefit of preclassification on separation sharpness, energy consumption, and product purity.

3. **Sensitivity analysis of wheel classifier RPM and air flow rate on protein enrichment**: Parametric sweeps of $\omega$ and $Q$ using the GPU-accelerated simulator to map the $d_{50}$–purity–recovery trade-off surface.

4. **SPH simulation of air flow distribution in multi-stage cyclone systems**: Analyzing flow maldistribution between cyclone stages and its effect on overall separation efficiency.

5. **CFD-DEM simulation of zigzag classifier deflector plate optimization**: Using the two-way coupled solver to optimize plate angle, spacing, and number of stages for maximum separation sharpness at minimum pressure drop.

6. **Multi-pass recirculation strategies for enhanced protein recovery**: Leveraging the particle re-injection capability to simulate 2-pass and 3-pass classification with intermediate product recycling.

7. **Effect of particle shape (sphericity) on classification cut size**: Comparing Schiller-Naumann and Haider-Levenspiel drag predictions for protein bodies vs. starch granules and quantifying the shape-dependent separation bias.

---

## Appendix A: Warp Kernel Parameters for Feed System

| Parameter | Source | Typical Value |
|-----------|--------|---------------|
| `hopper_top_radius` | `FeedHopperParams.top_radius` | 0.25 m |
| `hopper_bottom_radius` | `FeedHopperParams.bottom_radius` | 0.075 m |
| `hopper_cylinder_height` | `FeedHopperParams.cylindrical_height` | 0.3 m |
| `hopper_cone_height` | `FeedHopperParams.conical_height` | 0.4 m |
| `airlock_omega` | $2\pi \times 20/60$ | 2.09 rad/s |
| `airlock_radius` | `RotaryAirlockParams.rotor_diameter / 2` | 0.1 m |
| `feeder_omega` | $2\pi \times 60/60$ | 6.28 rad/s |
| `feeder_axial_speed` | `pitch × RPM / 60` | 0.08 m/s |
| `deagg_omega` | $2\pi \times 1500/60$ | 157.1 rad/s |
| `deagg_rotor_radius` | `DeagglomeratorParams.rotor_diameter / 2` | 0.1 m |

## Appendix B: SPH Parameters for Air System

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Number of particles | $N$ | 1000 | — |
| Smoothing length | $h$ | 0.04 | m |
| Speed of sound | $c_s$ | 50 | m/s |
| Rest density | $\rho_0$ | 1.204 | kg/m³ |
| Tait exponent | $\gamma$ | 7 | — |
| SPH viscosity | $\mu_\text{SPH}$ | 0.01 | Pa·s |
| XSPH factor | $\varepsilon$ | 0.1 | — |
| Hash grid cell size | — | $2.5h$ | m |
| Max velocity | $v_\text{max}$ | $0.5c_s$ | m/s |

## Appendix C: Classification Zone Velocity Scaling

| Zone | Velocity Model | Scaling from $Q$ |
|------|---------------|------------------|
| Venturi inlet | Continuity | $v = Q / A_\text{inlet}$ |
| Venturi throat | Continuity | $v = Q / A_\text{throat}$ |
| Zigzag bulk | Continuity | $v = Q / A_\text{channel}$ |
| Zigzag separation | Velocity ratio | $v = v_\text{bulk} \times \eta_v$ |
| Wheel radial | Blade passage | $v_r = Q / A_\text{passage}$ |
| Cyclone inlet | Tangential | $v_t = Q / A_\text{inlet}$ |
| Cyclone vortex | Rankine model | $v_\theta(r) = f(r, r_\text{core}, n)$ |
| Bag filter face | Face velocity | $v = Q / A_\text{filter}$ |

## Appendix D: Collection Efficiency Models

| Collector | Model | Key Parameter | Typical Efficiency |
|-----------|-------|--------------|-------------------|
| Zigzag | Terminal velocity comparison | $v_t$ vs $v_\text{zone}$ | Varies by stage |
| Wheel | Centrifugal force balance | $F_c/F_d$ ratio | ~90% at $d_{50}$ |
| Cyclone | Rosin-Rammler grade curve | $d_{50}$ from Lapple | 50% at $d_{50}$ |
| Bag filter | Inertial impaction (Stokes) | Stk number | >99.9% for $d > 1$ μm |

---

*This technical note forms the Multiphysics Simulation section of a comprehensive documentation of the `airclassifier` computational framework. It builds on the Geometry Modeling technical note (companion document) and provides the physics foundation for future sections on experimental validation, process optimization, and control system design.*
