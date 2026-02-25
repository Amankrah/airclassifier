# Parametric Geometry Modeling and GPU-Accelerated Visualization of a Multi-Stage Air Classification System

**Technical Note — Geometry Modeling Section**

*Emmanuel Kwofie*

---

## Abstract

This technical note presents the parametric geometry modeling framework underpinning a computational air classification system designed for dry fractionation of legume flours. The framework implements a hierarchical, component-based architecture in Python that constructs complete plant-scale classification equipment from first-principles geometric primitives. Thirty-six distinct industrial components — spanning feed handling, air supply, multi-stage classification, particle collection, safety instrumentation, structural support, and exhaust treatment — are assembled via a port-based spatial connection system into a unified digital twin. Signed distance field (SDF) representations enable implicit surface queries for particle–wall interactions. Explicit triangulated meshes support **real-time 3D visualization** in Python via **PyVista** (VTK); **NVIDIA Warp** is used for GPU-accelerated SDF evaluation, particle physics, and simulation kernels—not for rendering. The system supports two classification topologies: a full preclassification path (venturi eductor, zigzag classifier, centrifugal wheel classifier, multi-stage cyclones, and bag filter) and a simplified wheel-only path. All geometric parameters derive from engineering design equations with no embedded magic numbers, ensuring dimensional consistency from pilot-scale (50 kg/h) to production-scale (2000+ kg/h) systems.

---

## 1. Introduction

### 1.1 Motivation

Dry fractionation of legume flours by air classification is a solvent-free, energy-efficient route to protein-enriched and starch-enriched fractions. The process exploits differences in particle size, density, and aerodynamic behavior between protein bodies (2–20 μm, ρ ≈ 1350 kg/m³) and starch granules (20–50 μm, ρ ≈ 1500 kg/m³) to achieve separation under turbulent airflow. Unlike wet extraction, air classification preserves native protein functionality and avoids wastewater generation, making it attractive for plant-based food ingredient manufacturing.

Designing such systems requires tight coupling between equipment geometry and the physics of particle transport, drag, centrifugal acceleration, and wall interaction. Traditional design workflows rely on empirical correlations (e.g., Lapple, Barth, Muschelknautz) applied to isolated components, with limited ability to predict system-level performance when components are interconnected. A parametric geometry model that can generate both visualization-quality meshes and physics-ready implicit surfaces for every component in the system addresses this gap.

### 1.2 Scope

This note covers:

1. **Geometric primitives** (Section 2) — the foundational shapes (cylinder, cone, tube, rectangular duct) and their dual mesh/SDF representations.
2. **Industrial components** (Section 3) — thirty-six parameterized equipment models organized by subsystem function.
3. **Connection port system** (Section 4) — the mechanism for spatially aligning components with dimensional compatibility checks.
4. **Subsystem assemblies** (Section 5) — six functionally grouped assemblies covering the full classification plant.
5. **Complete system integration** (Section 6) — two classification topologies and the coordinate system conventions that unify them.
6. **Signed distance fields** (Section 7) — implicit geometry representations for particle–wall physics on GPU.
7. **Mesh generation and export** (Section 8) — structured/unstructured grid generation, adaptive refinement, marching cubes surface extraction, and STL/VTK export.
8. **Real-time visualization** (Section 9) — the **PyVista** (VTK) rendering pipeline in Python, with animated components and live simulation overlay.
9. **NVIDIA Warp integration** (Section 10) — GPU kernel architecture for SDF evaluation, particle physics, and SPH fluid dynamics; used in simulation back ends, not for rasterization.

### 1.3 Software Architecture Overview

The geometry framework is organized into six logical layers:

1. **Primitives** — Cylinder, cone, tube, and rectangular duct; the four building blocks from which all components are composed.
2. **Components** — Thirty-six industrial equipment models (instrumentation, safety, support structures, etc.) each with parameters, mesh generation, and connection ports.
3. **Assembly** — Eight subsystem assemblies (classification, feed, air, ductwork, safety/instrumentation, support/exhaust) plus the complete integrated system.
4. **Connection ports** — The port-based spatial alignment system that positions components and checks dimensional compatibility.
5. **Signed distance fields** — Implicit geometry representation for cyclone (and extensible to other shapes) used for particle–wall queries on CPU or GPU.
6. **Mesh generation and export** — Structured and cylindrical grids, adaptive refinement, marching-cubes surface extraction from SDFs, and STL/VTK export.

All components share a common interface: a `Params` dataclass for configuration, a `generate_mesh()` method returning `(vertices, indices, normals)` as NumPy arrays, and (where applicable) a `to_warp_mesh(device)` method for GPU upload via NVIDIA Warp. The wheel classifier (`WheelClassifier`) is part of the classification assembly and is created internally; junction and connector components are used by assemblies for wheel-only or duct connections.

---

## 2. Geometric Primitives

### 2.1 Design Philosophy

Every industrial component in the classification system can be decomposed into combinations of four geometric primitives: cylinders, cones (frustums), hollow tubes, and rectangular ducts. Rather than modeling each component as a monolithic mesh, the framework constructs component geometry by composing primitives — a cyclone body is a cylinder atop a cone; a vortex finder is a tube with an annular flange; a blower scroll is a swept profile. This compositional approach ensures mesh consistency, simplifies parameterization, and allows the same primitive to serve both explicit (triangulated mesh) and implicit (SDF) representations.

### 2.2 Cylinder

The `CylinderParams` dataclass defines a right circular cylinder by its radius $r$, height $h$, base center $\mathbf{c}$, and alignment axis. The parametric surface for the lateral wall (Y-axis alignment) is:

$$\mathbf{P}(\theta, t) = \begin{pmatrix} r\cos\theta + c_x \\ th + c_y \\ r\sin\theta + c_z \end{pmatrix}, \quad \theta \in [0, 2\pi),\ t \in [0, 1]$$

The outward unit normal on the lateral surface is purely radial:

$$\hat{\mathbf{n}}(\theta) = (\cos\theta,\ 0,\ \sin\theta)$$

End caps are triangulated as fans from a central vertex to circumferential ring vertices, with normals aligned to the cylinder axis ($\pm\hat{\mathbf{y}}$).

**Mesh complexity.** For the default resolution of $n_r = 32$ radial and $n_a = 16$ axial divisions, the lateral surface generates $(n_a + 1) \times n_r = 544$ vertices and $2 n_a n_r = 1024$ triangles. Each end cap adds $n_r + 1$ vertices and $n_r$ triangles, yielding approximately 610 vertices and 1088 triangles total.

**Computed properties** available on the params dataclass include volume ($\pi r^2 h$), lateral surface area ($2\pi r h$), and total surface area ($2\pi r h + 2\pi r^2$).

**SDF formulation.** The signed distance to a solid cylinder is computed as:

$$d_r = \sqrt{(p_x - c_x)^2 + (p_z - c_z)^2} - r$$
$$d_a = \max\bigl(-(p_y - c_y),\ (p_y - c_y) - h\bigr)$$

$$\text{SDF}(\mathbf{p}) = \begin{cases} \sqrt{d_r^2 + d_a^2} & \text{if } d_r > 0 \text{ and } d_a > 0 \\ \max(d_r,\ d_a) & \text{otherwise} \end{cases}$$

The corner-case branch (both distances positive) handles the rounded-corner region at cylinder edges, ensuring $C^0$ continuity of the distance field.

A hollow variant `cylinder_sdf_hollow` replaces the radial distance with $d_r = \max(|\mathbf{p}|_r - r_o,\ r_i - |\mathbf{p}|_r)$, where $r_o$ and $r_i$ are the outer and inner radii.

Both SDF functions are decorated with `@wp.func` for compilation to GPU-native code via NVIDIA Warp's JIT compiler.

### 2.3 Cone (Frustum)

The `ConeParams` dataclass models a frustum (truncated cone) by its top radius $r_t$, bottom radius $r_b$, and height $h$. A full cone is obtained when $r_b \to 0$. The top face is positioned at the `center` coordinate and the cone extends downward.

The parametric surface with linearly interpolated radius is:

$$\mathbf{P}(\theta, t) = \begin{pmatrix} R(t)\cos\theta + c_x \\ -th + c_y \\ R(t)\sin\theta + c_z \end{pmatrix}, \quad R(t) = r_t(1 - t) + r_b\,t$$

The outward normal on the slant surface accounts for the wall inclination. Defining the slant height $s = \sqrt{h^2 + (r_t - r_b)^2}$, the radial and axial normal components are:

$$n_r = \frac{h}{s}, \quad n_a = \frac{r_t - r_b}{s}$$

$$\hat{\mathbf{n}}(\theta) = \bigl(n_r\cos\theta,\ n_a,\ n_r\sin\theta\bigr)$$

This ensures the normal is perpendicular to the slant surface rather than the horizontal.

**Computed properties** include the frustum volume $V = \frac{\pi h}{3}(r_t^2 + r_t r_b + r_b^2)$, slant height, half-angle $\alpha = \arctan\frac{r_t - r_b}{h}$, and lateral surface area $A_\text{lat} = \pi(r_t + r_b)s$.

**SDF formulation** for the frustum:

$$t = \text{clamp}\!\left(\frac{y_\text{local}}{h},\ 0,\ 1\right), \quad R_w = r_t(1-t) + r_b\,t$$

$$d_\text{slant} = \frac{(|\mathbf{p}|_r - R_w) \cdot h}{s}$$

The slant distance projection accounts for the cone angle, converting a radial displacement into the perpendicular distance to the inclined wall. Cap regions (above top, below bottom) are handled with Euclidean distance to the nearest cap edge.

### 2.4 Tube (Hollow Cylinder)

The `TubeParams` dataclass extends the cylinder to include inner and outer radii ($r_i < r_o$), modeling pipe walls, vortex finders, and duct sections. Validation in `__post_init__` enforces $r_i < r_o$.

The mesh consists of four surface groups:

| Surface | Normal Direction | Winding |
|---------|-----------------|---------|
| Outer cylinder | Radially outward | Standard |
| Inner cylinder | Radially inward | Reversed |
| Near-end annulus | Along axis (−) | Direction-dependent |
| Far-end annulus | Along axis (+) | Direction-dependent |

The reversed winding on the inner surface ensures correct backface culling when viewing from inside the tube (the flow passage). Each annular end face connects outer and inner ring vertices with $2n_r$ triangles.

**Computed properties** include wall thickness ($r_o - r_i$), cross-sectional flow area ($\pi r_i^2$), inner/outer surface areas, and wall material cross-section ($\pi(r_o^2 - r_i^2)$).

### 2.5 Rectangular Duct

The `RectangularDuctParams` dataclass defines a rectangular prism by width $w$, height $h_d$, length $l$, center position, and direction vector. An orthonormal basis is constructed from the direction vector:

$$\hat{\mathbf{r}} = \frac{\hat{\mathbf{d}} \times \hat{\mathbf{u}}}{|\hat{\mathbf{d}} \times \hat{\mathbf{u}}|}, \quad \hat{\mathbf{u}}' = \hat{\mathbf{r}} \times \hat{\mathbf{d}}$$

where $\hat{\mathbf{u}}$ is an initial up vector chosen to avoid degeneracy ($(0,1,0)$ unless the duct is nearly vertical, in which case $(1,0,0)$ is used).

Eight corner vertices define six quad faces (12 triangles). Each face normal is one of the six basis directions ($\pm\hat{\mathbf{d}}, \pm\hat{\mathbf{r}}, \pm\hat{\mathbf{u}}'$).

The hydraulic diameter, relevant for pressure drop calculations, is:

$$D_h = \frac{4wh_d}{2(w + h_d)}$$

### 2.6 Warp Mesh Conversion

All primitives provide a `to_warp_mesh(device)` method that uploads vertex and index arrays to GPU memory:

```python
def to_warp_mesh(self, device="cuda"):
    points = wp.array(self._vertices, dtype=wp.vec3, device=device)
    indices = wp.array(self._indices, dtype=wp.int32, device=device)
    return wp.Mesh(points=points, indices=indices)
```

The `wp.Mesh` object enables hardware-accelerated ray tracing and spatial queries on the GPU. Mesh generation is lazy — arrays are computed on first access and cached for subsequent calls.

---

## 3. Industrial Components

### 3.1 Component Architecture

Each of the thirty-six components follows a uniform pattern:

1. **Params dataclass** — Engineering parameters with SI units, computed properties (volumes, areas, velocities), and `__post_init__` validation.
2. **`generate_mesh()`** — Returns `(vertices: ndarray[N,3], indices: ndarray[M], normals: ndarray[N,3])` with `float32`/`int32` dtypes.
3. **Connection ports** — Named `ConnectionPort` instances defining inlet/outlet positions, directions, and diameters for inter-component alignment.
4. **Factory functions** — `create_standard_*()` convenience constructors with engineering-sensible defaults.

Components with moving parts additionally implement:
- **`get_static_mesh()`** — Non-moving geometry (housings, flanges, supports).
- **`get_rotor_mesh(angle)`** / **`get_blade_mesh(position)`** — Geometry at a given rotation angle or actuator position, enabling frame-by-frame animation.
- **`update_animation(dt, speed)`** — Advance internal angle state by a time step.

### 3.2 Component Inventory by Subsystem

The following table summarizes all thirty-six components across six functional subsystems:

| Phase | Subsystem | Components | Animated Parts |
|-------|-----------|-----------|----------------|
| — | Core Cyclone | CycloneBody, TangentialInlet, VortexFinder, DustOutlet, Overflow | — |
| 1 | Classification | ZigzagClassifier, VenturiEductor, WheelClassifier, MultiCycloneSystem, BagFilter | Wheel rotor |
| 2 | Feed | FeedHopper, RotaryAirlock, ScrewFeeder, Deagglomerator | Lid, rotor, screw, pin rotor |
| 3 | Air Supply | CentrifugalBlower, InletAirFilter, FlowDamper | Impeller, pulleys, blade |
| 4 | Ductwork | RoundDuct, RectangularDuct, Transition, Elbow, DiverterValve | Diverter blade |
| 5a | Safety | ExplosionVent, GroundingPoint | — |
| 5b | Instrumentation | PressurePort, TemperaturePort, SamplePort, SightGlass | — |
| 6a | Support | EquipmentLegs, StructuralFrame | — |
| 6b | Exhaust | Silencer, ExhaustStack | — |
| — | Connectors | Transition (round/rect reducers), junction components (e.g. ThreePointJunction for wheel-only mode) used internally by assemblies | — |

### 3.3 Core Cyclone Components

#### 3.3.1 CycloneBody

The cyclone body is the primary separation vessel, composed of an upper cylindrical section for vortex development and a lower conical section that concentrates the downward spiral flow toward the dust outlet. The `CycloneBodyParams` dataclass accepts:

- `cylinder_diameter` [m]: Defines the characteristic dimension $D$ from which many correlations scale.
- `cylinder_height` [m]: Typically $1.5D$ for Stairmand high-efficiency designs.
- `cone_height` [m]: Typically $2.5D$, governing residence time for fine particles.
- `cone_tip_diameter` [m]: Underflow opening, typically $0.375D$.

Mesh generation composes a `Cylinder` primitive (top) with a `Cone` primitive (bottom), merging vertex arrays with an index offset. The mesh resolution defaults to 48 radial segments, producing smooth curved surfaces suitable for visualization at engineering scale.

The Warp SDF function `cyclone_body_sdf()` provides a piecewise distance evaluation: cylindrical region ($d = r - R$), conical region (perpendicular distance to slant surface), and transition region at the cylinder–cone junction.

#### 3.3.2 TangentialInlet

Models the rectangular entry duct that introduces air and particles tangentially into the cyclone body. The inlet geometry includes a saddle-cut profile where the duct meets the cylindrical wall, computed by blending cross-sections along the duct length to match the body curvature. Key parameters include width, height, length, and the angular position around the cyclone circumference ($0$ to $2\pi$ radians), which enables different inlet orientations for series-connected cyclone arrangements.

The computed inlet velocity direction combines tangential and axial components:

$$\hat{\mathbf{v}}_\text{inlet} = \cos(\alpha_\text{entry})\,\hat{\mathbf{t}} + \sin(\alpha_\text{entry})\,(-\hat{\mathbf{y}})$$

where $\alpha_\text{entry}$ is the downward spiral entry angle.

#### 3.3.3 VortexFinder

The vortex finder is a tube that protrudes downward into the cyclone body, establishing the inner (ascending) vortex for fines extraction. It is modeled as a `Tube` primitive with optional flanges at the top. The insertion depth determines the separation between the inlet flow and the overflow exit — deeper insertion increases residence time but raises pressure drop.

The SDF function returns negative values inside the tube wall (the solid region between inner and outer radii) and positive values in both the flow passage and the external cyclone space, enabling the simulation to distinguish between particles inside the vortex finder (heading to overflow) and particles in the outer vortex (heading to underflow).

#### 3.3.4 DustOutlet and Overflow

The dust outlet at the cone tip collects the coarse (underflow) fraction. The `Overflow` component is a logical region rather than a physical shape — it tracks particles exiting upward through the vortex finder and records their size distribution for separation efficiency calculation.

### 3.4 Classification Components (Phase 1)

#### 3.4.1 Zigzag Classifier

The zigzag classifier is the primary air classifier stage, providing counter-current separation through alternating deflector plates. The `ZigzagClassifierParams` dataclass defines:

- `channel_width`, `channel_depth` [m]: Main separation channel dimensions.
- `num_stages` (3–7): Number of deflector plates — each stage provides an additional separation pass.
- `stage_height` [m]: Vertical spacing between plates.
- `plate_angle` [rad]: Deflector inclination from vertical (30°–60°, typical 45°).
- `plate_length_ratio` (0.4–0.6): Fraction of channel blocked by each plate.

The physics of separation rely on the competition between gravitational settling and aerodynamic drag at each deflector. Particles encountering a plate are deflected into the recirculation zone behind it, where local air velocity is reduced to 20–40% of the bulk velocity. Particles whose terminal settling velocity exceeds the local upward air velocity fall to the next stage (coarse); those entrained by the airflow rise to the next stage (fines).

Each deflector is represented by a `DeflectorPlate` dataclass recording stage number, side (left/right alternating), base and tip positions, angle, and surface normal. The `SeparationZone` dataclass characterizes the recirculation region behind each plate with its velocity ratio and turbulence intensity.

Mesh generation produces the channel walls, deflector plates with their specified thickness, and inlet/outlet ducts (air inlet at bottom, fines outlet at top, coarse outlet at bottom, and optional side feed inlet).

#### 3.4.2 Venturi Eductor

The venturi eductor entrains particulate material into the airstream using the pressure differential at a converging–diverging nozzle throat. The geometry is defined by inlet, throat, and outlet diameters, with convergent and divergent half-angles (10–15° and 3–7° respectively). The solids inlet port is positioned at the throat where static pressure is lowest, at a configurable angular position and tilt angle.

The mesh is generated as a surface of revolution with a piecewise-linear radius profile: convergent section, constant-diameter throat, and divergent recovery section. The solids inlet is a cylindrical branch intersecting the throat region.

Key computed properties include the area ratio $A_\text{inlet}/A_\text{throat}$ (governing the pressure differential via Bernoulli's equation), throat velocity, and the total length.

#### 3.4.3 Wheel Classifier

The centrifugal wheel (or turbine) classifier is the mandatory second-stage separator that achieves the fine cut sizes required for protein–starch separation. Operating at 1000–8000 RPM, it generates centrifugal accelerations of 1000–5000$g$ at the wheel rim, compared to the 1$g$ gravity-based separation of the zigzag.

The `WheelClassifierParams` dataclass specifies:

- **Wheel geometry**: diameter (200 mm typical), width, hub diameter, blade count (16–48), blade thickness, and shroud dimensions.
- **Housing**: volute type with controlled expansion ratio and clearance.
- **Inlets/outlets**: tangential feed inlet, axial fines outlet through the hub, and conical coarse hopper with gravity discharge.
- **Operating conditions**: RPM and target $d_{50}$.

The theoretical cut diameter follows from the balance of centrifugal and drag forces on a spherical particle at the wheel rim:

$$d_{50} = \sqrt{\frac{18\,\mu\,v_r}{\Delta\rho\,\omega^2\,r}}$$

where $\mu$ is air viscosity, $v_r$ is the radial inflow velocity, $\Delta\rho$ is the particle–air density difference, $\omega$ is the angular velocity, and $r$ is the wheel radius.

The wheel mesh includes a cage-style rotor with radial blades between front and rear shroud discs, and supports animation for real-time visualization of the spinning classifier.

#### 3.4.4 Multi-Cyclone System

The multi-cyclone system arranges 2–4 cyclone stages in series for progressively finer particle collection downstream of the wheel classifier. Each stage is a complete `CycloneAssembly` instance with its own body, inlet, vortex finder, and dust outlet. Typical configurations use:

| Stage | Diameter | Design $d_{50}$ | Target Fraction |
|-------|----------|-----------------|-----------------|
| Primary | 300 mm | 30–50 μm | Starch-rich coarse |
| Secondary | 200 mm | 15–25 μm | Mixed intermediate |
| Tertiary | 120 mm | 5–15 μm | Protein-rich fines |

Series connection ductwork is automatically generated between overflow exits and downstream inlets, with appropriate transitions for diameter changes.

#### 3.4.5 Bag Filter

The bag filter provides final particle collection (>99.9% efficiency for particles >1 μm) on the exhaust airstream leaving the cyclone train. The geometry includes:

- Rectangular housing with dirty-air and clean-air plenums separated by a tube sheet.
- Grid of cylindrical filter bags ($n_x \times n_z$ array) suspended from the tube sheet.
- Conical hopper below the dirty-air section for collected fines.
- Optional pulse-jet cleaning system: compressed-air header pipe, blow tubes with nozzles, and air receiver tank.

The `BagFilterParams` dataclass computes the total filter area $A_f = \pi D_b L_b \times N_\text{bags}$ and the air-to-cloth ratio $\dot{V}/A_f$ [m³/min/m²], which is the primary design parameter governing filter life and pressure drop.

### 3.5 Feed System Components (Phase 2)

#### 3.5.1 Feed Hopper

The feed hopper stores bulk powder and meters it to the downstream process. The geometry is a truncated cone (conical section) topped by a cylinder (straight section), with parametric control over:

- Top and bottom diameters (determining the half-angle for mass-flow discharge design).
- Cylindrical and conical section heights.
- Optional hinged lid with handle geometry.

The hopper capacity is computed from the combined cylindrical and conical volumes at a specified bulk density (500 kg/m³ for legume flour). The cone half-angle is a critical design parameter — values below approximately 25° from vertical are required for mass-flow discharge of cohesive food powders.

The mesh is split into body and lid components to support lid opening/closing animation during the filling simulation phase. The `get_lid_hinge_position()` method returns the hinge point for rotation calculations.

#### 3.5.2 Rotary Airlock

The rotary airlock provides a pressure seal between the atmospheric hopper and the pneumatic conveying system while metering powder at a controlled rate. The geometry includes:

- A cylindrical rotor with $n$ radial vanes (6–10 typical, 8 default) creating pockets.
- A close-fitting cylindrical housing with specified tip clearance (0.3 mm default for food-grade sealing).
- Circular flanged inlet and outlet ports.
- End plates with shaft bearing bosses.

The volumetric capacity is:

$$\dot{V} = V_\text{pocket} \times n_\text{vanes} \times \text{RPM} \times \text{fill factor}$$

where $V_\text{pocket}$ is the volume between adjacent vanes.

The rotor mesh is generated separately from the housing to enable rotation animation. The `update_rotation(dt, rpm)` method advances the internal angle state, and `get_rotor_mesh(angle)` returns vertices at any rotation position.

#### 3.5.3 Screw Feeder

The screw feeder provides precise volumetric dosing via a helical screw rotating inside an enclosed cylindrical tube. Key parameters include screw diameter, shaft diameter, pitch, flight thickness, and trough length with clearance. The screw supports optional variable pitch (decreasing toward the inlet to prevent flood feeding).

The mesh generates a helical surface by sweeping a cross-section along the screw axis with angular advance proportional to pitch:

$$\theta(x) = \frac{2\pi x}{p}$$

where $p$ is the screw pitch and $x$ is the axial position. The number of helical segments ($n_\text{helix} = 36$ default) controls the smoothness of the helix representation.

Flanged inlet and outlet necks sized to match the upstream airlock and downstream deagglomerator ensure dimensional compatibility in assembly.

#### 3.5.4 Deagglomerator

The deagglomerator breaks powder agglomerates using high-speed rotating pins that force material through a sizing screen. The geometry comprises:

- A cylindrical housing with inlet and outlet ports.
- A rotor shaft with multiple rows of radial pins ($n_\text{rows} \times n_\text{pins/row}$).
- A semi-cylindrical screen with specified aperture size and open area fraction.

Pin tip speed ($v_\text{tip} = \pi D_\text{rotor} \times \text{RPM}/60$) governs the impact energy available for breakage. Typical operating speeds of 1500 RPM with a 200 mm rotor yield tip speeds of approximately 16 m/s.

### 3.6 Air System Components (Phase 3)

#### 3.6.1 Centrifugal Blower

The centrifugal blower is the most geometrically complex single component, comprising:

- **Impeller**: Backward-curved blades (6–12) between hub and rim, with configurable inlet and outlet blade angles. Blade type (backward-curved, radial, forward-curved) determines efficiency characteristics.
- **Scroll/volute**: A spiral casing with controlled expansion that converts kinetic energy to pressure. The scroll profile follows an Archimedean spiral with the expansion ratio parameter.
- **Inlet bell**: A converging intake cone on the suction side.
- **Outlet duct**: Rectangular discharge from the scroll tangent.
- **Motor assembly**: Sized to IEC frame standards based on estimated shaft power ($P = \dot{V} \cdot \Delta p / \eta$). Includes cooling fins, terminal box, and mounting feet.
- **Belt drive**: Motor pulley, driven pulley, and belt connecting the motor shaft to the impeller shaft.

Three separate animated meshes are maintained:
1. Impeller (rotates at blower RPM)
2. Driven pulley (same speed as impeller)
3. Motor pulley (faster by the pulley ratio)

The `update_animation(dt, rpm)` method advances both impeller and motor angles:

$$\theta_\text{impeller}(t + \Delta t) = \theta_\text{impeller}(t) + \frac{2\pi \cdot \text{RPM}}{60} \cdot \Delta t$$

$$\theta_\text{motor}(t + \Delta t) = \theta_\text{motor}(t) + \frac{2\pi \cdot \text{RPM}}{60} \cdot \frac{D_\text{driven}}{D_\text{motor}} \cdot \Delta t$$

Key aerodynamic properties computed from the params include tip speed, specific speed, estimated efficiency (blade-type dependent), and shaft power.

#### 3.6.2 Inlet Air Filter

Models panel, bag, cartridge, or HEPA filter elements within a rectangular housing. The `filter_type` parameter selects internal geometry (e.g., cylindrical cartridge elements vs. flat panel media) and sets the design face velocity (0.05 m/s for HEPA through 2.5 m/s for coarse panels). The efficiency class follows ISO 16890 / EN 1822 designations (G1–G4, M5–M6, F7–F9, E10–E12, H13–H14).

#### 3.6.3 Flow Damper

Butterfly, louver, or iris dampers for airflow control and isolation. The butterfly damper models a single circular disc rotating about a diameter:

- At position 0 (closed): the blade is perpendicular to flow, blocking the duct.
- At position 1 (open): the blade is parallel to flow, offering minimal obstruction.

The flow area varies approximately as:

$$A(\phi) = A_\text{duct} \cdot \sin(\phi \cdot \pi/2)$$

where $\phi \in [0, 1]$ is the normalized position. The `get_blade_mesh(position)` method generates the blade at any intermediate angle.

### 3.7 Ductwork Components (Phase 4)

#### 3.7.1 Round and Rectangular Ducts

Round ducts are modeled as thin-walled tubes with optional flanges and insulation. Pressure drop is computed using the Darcy-Weisbach equation:

$$\Delta p = f \cdot \frac{L}{D_h} \cdot \frac{\rho v^2}{2}$$

with friction factor $f$ from the Swamee-Jain approximation to the Colebrook equation.

#### 3.7.2 Transitions

Transition pieces connect ducts of different sizes or shapes (round-to-round, round-to-rectangular, and vice versa). The `TransitionParams` dataclass validates expansion angles against maximum recommended values and computes the Borda-Carnot pressure loss coefficient for expansions:

$$K = \left(1 - \frac{A_1}{A_2}\right)^2$$

#### 3.7.3 Elbows

Elbow mesh generation sweeps a circular or rectangular cross-section along a circular arc defined by the bend radius, bend angle, and bend axis. The radius-to-diameter ratio ($R/D$) determines whether the bend is classified as tight radius ($R/D < 1.5$) or standard, affecting the pressure loss coefficient according to ASHRAE correlations.

Mitered elbows are modeled as a series of straight segments (gores) with angular cuts. Optional turning vanes inside the elbow are generated as thin curved plates following the arc centerline.

#### 3.7.4 Diverter Valve

Y-type two-way diverter for routing flow between alternate process paths. Three blade types are modeled: flap (pivoting plate), rotating (cylindrical gate), and plug (translating cone). The outlet angle parameter controls the bifurcation geometry.

### 3.8 Safety and Instrumentation (Phase 5)

Explosion vents are sized per EN 14491 / NFPA 68:

$$A_v = C \cdot V^{2/3} \cdot K_\text{St} / (100 \cdot \sqrt{\Delta p_\text{red}})$$

where $V$ is the vessel volume, $K_\text{St}$ is the deflagration index (150 bar·m/s for legume dust), and $\Delta p_\text{red}$ is the acceptable reduced explosion pressure.

Grounding points (weld studs or threaded bosses) are spaced to ensure equipotential bonding with maximum resistance of 1 Ω, critical for preventing electrostatic ignition in dust-laden atmospheres.

Instrumentation ports (pressure, temperature, sample, sight glass) are modeled with engineering-standard connections (NPT threaded, flanged, welded) and positioned according to measurement best practices — pressure ports at representative flow cross-sections, temperature probes at sufficient immersion depth, and sample ports with isokinetic nozzle options for representative particle sampling.

### 3.9 Support Structures and Exhaust (Phase 6)

Equipment legs (tubular, channel, or adjustable) are positioned at equal angular intervals on a mounting circle. The structural frame comprises vertical columns, horizontal beams, optional diagonal bracing, and platform gratings at specified elevation levels.

Silencer geometry includes an inner perforated duct, absorption splitter baffles (mineral wool), and an outer shell. The exhaust stack is a cylindrical pipe with base flange and rain cap (conical, Chinese hat, or H-cap options).

---

## 4. Connection Port System

### 4.1 Port Abstraction

The `ConnectionPort` dataclass is the fundamental mechanism for inter-component spatial alignment. Each port defines:

| Field | Type | Description |
|-------|------|-------------|
| `position` | `(x, y, z)` | Local position relative to component origin |
| `direction` | `(dx, dy, dz)` | Outward-facing unit normal |
| `diameter` | `float` | Port diameter for circular connections [m] |
| `width`, `height` | `float` | Dimensions for rectangular connections [m] |
| `port_type` | `PortType` enum | CIRCULAR, RECTANGULAR, FLANGED, SLIP, THREADED, GRAVITY |
| `name` | `str` | Human-readable identifier (e.g., "air_inlet", "fines_outlet") |
| `compatible_types` | `List[PortType]` | Acceptable mating port types |

Direction vectors are normalized in `__post_init__` to ensure unit-length invariants throughout the alignment pipeline.

### 4.2 Alignment Algorithm

The `calculate_alignment()` function positions a target component so that its inlet port mates flush with a source component's outlet port. Given source port $\mathbf{s}$ at world position $\mathbf{p}_s + \mathbf{s}_\text{local}$ with outward direction $\hat{\mathbf{d}}_s$, the target component position is:

$$\mathbf{p}_\text{target} = (\mathbf{p}_s + \mathbf{s}_\text{local}) - \mathbf{t}_\text{local} + g\,\hat{\mathbf{d}}_s$$

where $\mathbf{t}_\text{local}$ is the target port's local position and $g$ is the gap distance (typically 2 mm for gasket space).

Direction compatibility is verified by the dot product:

$$\hat{\mathbf{d}}_s \cdot \hat{\mathbf{d}}_t < -0.9$$

requiring the ports to face in approximately opposite directions (within ≈25°).

### 4.3 Dimensional Compatibility

The `is_compatible()` method on `ConnectionPort` checks:

1. **Type compatibility**: The source port type must appear in the target's `compatible_types` list.
2. **Size matching**: For circular ports, $|d_1 - d_2| \leq \tau \cdot \max(d_1, d_2)$ with default tolerance $\tau = 0.01$ (1%). Rectangular ports check both width and height independently.

When sizes differ beyond tolerance, a `Transition` component is automatically inserted to bridge the diameter change.

### 4.4 Series Connection

The `connect_in_series()` function chains an ordered sequence of components port-to-port:

```python
positions = connect_in_series(
    components=[filter, blower, damper_1, damper_2],
    port_pairs=[("outlet", "inlet"), ("outlet", "inlet"), ("outlet", "inlet")],
    start_position=(0, 0, 0),
    gaps=[0.002, 0.002, 0.002]  # 2mm gasket gaps
)
```

Each connection updates the cumulative position for the next component, yielding a dictionary of world positions indexed by component.

### 4.5 Assembly Validation

The `validate_assembly_connections()` function performs post-assembly checks on all port pairs, reporting:

- Gap distance between mating port centers (should be ≤ tolerance).
- Direction alignment (dot product should be < −0.9).
- Type and size compatibility.

The `print_connection_report()` function produces a human-readable summary with pass/fail indicators.

---

## 5. Subsystem Assemblies

### 5.1 Assembly Hierarchy

The system follows a four-level compositional hierarchy:

```
Level 0: Primitives (Cylinder, Cone, Tube, RectangularDuct)
Level 1: Components (CycloneBody, Blower, Hopper, ...)
Level 2: Subsystem Assemblies (FeedSystem, AirSystem, Classification, ...)
Level 3: Complete System (CompleteClassifierAssembly)
```

Each Level 2 assembly has a `Params` dataclass, a `build_mesh()` method that returns combined `(vertices, indices)`, and a `print_summary()` method for human-readable geometry reporting.

### 5.2 Feed System Assembly (Phase 2)

The `FeedSystemAssembly` arranges four components in a gravity-fed vertical stack:

```
Feed Hopper (top, elevated)
    │ gravity discharge
    ▼
Rotary Airlock (pressure seal + metering)
    │ pocket transfer
    ▼
Screw Feeder (horizontal, enclosed tube)
    │ helical advance
    ▼
Deagglomerator (high-speed lump breaking)
    │ through sizing screen
    ▼
[To classification system inlet]
```

The `FeedSystemParams` dataclass specifies hopper capacity, discharge diameter, airlock rotor diameter, screw feeder dimensions, deagglomerator rotor size, and screen aperture. Component spacing defaults to 2 mm (gasket space). The assembly automatically sizes transition connectors between components where port diameters differ.

Connection ports are matched by the alignment algorithm: hopper discharge outlet → airlock inlet, airlock outlet → feeder inlet, feeder outlet → deagglomerator inlet. The feeder axis is horizontal (X-axis), while the hopper–airlock–feeder stack is vertical (Y-axis), requiring a 90° direction change at the airlock outlet that is handled by the transition connector geometry.

### 5.3 Air System Assembly (Phase 3)

The `AirSystemAssembly` arranges the air supply chain:

```
Inlet Air Filter (atmospheric intake)
    │ filtered air
    ▼
[Ductwork: straight + 90° elbow]
    │
    ▼
Centrifugal Blower (scroll housing)
    │ pressurized air
    ▼
[Transition: rectangular outlet → round duct]
    │
    ▼
Flow Damper 1 (control)
    │
    ▼
Flow Damper 2 (isolation)
    │
    ▼
[To venturi air inlet]
```

The assembly computes duct diameters from the design flow rate and target velocity (15–20 m/s for pneumatic conveying). Elbow bend radii default to 1.5× duct diameter for acceptable pressure loss. The blower scroll geometry with its offset inlet bell requires careful spatial positioning to maintain the suction–discharge flow path.

### 5.4 Classification System Assembly (Phase 1)

This is the largest subsystem, integrating six component types with two possible topologies:

**Topology A — Full Preclassification:**

```
Venturi Eductor (particle entrainment)
    │ mixed air + solids
    ▼
Zigzag Classifier (gravity pre-separation)
    ├── coarse outlet → Rotary Airlock → [Collection]
    │
    ▼ fines
Wheel Classifier (centrifugal fine separation, MANDATORY)
    ├── coarse outlet (25–50 μm) → Rotary Airlock → [Collection]
    │
    ▼ fines (<25 μm)
Multi-Cyclone System (3-stage series)
    ├── Stage 1 underflow → [Collection]
    ├── Stage 2 underflow → [Collection]
    ├── Stage 3 underflow → [Collection]
    │
    ▼ overflow (clean air + ultrafines)
Bag Filter (final collection)
    │ clean exhaust air
    ▼
[To exhaust system]
```

**Topology B — Wheel-Only (No Preclassification):**

```
Three-Point Junction
    ├── Air inlet (from blower)
    ├── Solids inlet (from feed system)
    │
    ▼ merged stream
Wheel Classifier (direct classification)
    ├── coarse outlet → [Collection]
    │
    ▼ fines
Multi-Cyclone System → Bag Filter → [Exhaust]
```

The wheel classifier is mandatory in both topologies because it provides the centrifugal force ($1000$–$5000g$) necessary to achieve $d_{50} \approx 25$ μm for effective protein–starch separation. The zigzag pre-classifier ($1g$ separation) achieves $d_{50} \sim 100$+ μm and serves primarily to remove oversized material and reduce the loading on the wheel.

### 5.5 Ductwork System Assembly (Phase 4)

Configurable duct networks with automated pressure drop estimation using the equivalent-length method:

$$L_\text{eq} = L_\text{straight} + 30D \cdot n_{90°} + 16D \cdot n_{45°}$$

$$\Delta p = f \cdot \frac{L_\text{eq}}{D_h} \cdot \frac{\rho v^2}{2}$$

### 5.6 Safety and Instrumentation Assembly (Phase 5)

Positions safety and measurement devices around the classification vessel. Explosion vents are placed at the top (weakest structural point), grounding studs are distributed angularly around the lower vessel, and instrumentation ports are placed at representative measurement locations. The number of each device type scales with vessel volume for the `create_full_instrumentation()` factory function.

### 5.7 Support and Exhaust Assembly (Phase 6)

Combines structural elements (legs, frame, platforms) with exhaust treatment (silencer, stack). The frame height and platform levels are configured to provide operator access to instrumentation and maintenance points. Three factory variants cover compact (pilot-scale), standard, and industrial configurations.

---

## 6. Complete System Integration

### 6.1 Coordinate System Convention

All modules use a consistent Y-up coordinate system:

| Axis | Direction | Physical Meaning |
|------|-----------|-----------------|
| X | Horizontal right | Equipment width / lateral |
| Y | Vertical up | Height / gravity axis |
| Z | Horizontal depth | Equipment depth / into page |

Gravity acts in the $-Y$ direction. The classification system is positioned at the origin (venturi air inlet with preclassification, or junction when wheel-only); the feed system is elevated above and behind ($+Y$, $+Z$), and the air system below ($-Y$).

### 6.2 CompleteClassifierAssembly

The `CompleteSystemParams` dataclass controls the entire plant:

- **Throughput**: feed rate (500 kg/h default), cut size target (20 μm), air flow rate (3000 m³/h).
- **Subsystem enables**: Booleans for feed, air, ductwork, support, and exhaust inclusion.
- **Layout**: Compact or standard air duct routing.
- **Classification topology**: `use_preclassification=True/False`.

The build sequence is:

1. **Classification system** at origin — establishes the core flow path (venturi + zigzag + wheel + cyclones + bag filter when `use_preclassification=True`, or three-point junction + wheel + cyclones + bag filter when `use_preclassification=False`).
2. **Feed system** positioned at the venturi solids inlet (with preclassification) or at the three-point junction solids inlet (wheel-only), with ~15° downslope for gravity-assisted feed.
3. **Air system** connected to the venturi air inlet (with preclassification) or to the junction air inlet (wheel-only); compact layout uses a short duct with a single elbow; standard layout uses a multi-elbow routed path.
4. **Exhaust system** — silencer and stack positioned at the bag filter clean-air outlet.
5. **Ductwork** — connecting ducts, transitions, and elbows are built last so they can reference all subsystem port positions.

### 6.3 Inter-System Connections

The ductwork connecting subsystems is generated dynamically based on port positions:

- **Air-to-venturi**: Damper outlet → transition (if diameters differ) → straight duct → 90° elbow → venturi air inlet.
- **Feed-to-venturi**: Deagglomerator outlet → angled chute (15° from vertical) → venturi solids inlet.
- **Cyclone overflow-to-bag filter**: Round duct connecting cyclone train exhaust to bag filter dirty-air inlet.
- **Bag filter-to-exhaust**: Clean-air outlet → silencer → exhaust stack.

All connections use the port alignment system (Section 4) to ensure dimensional consistency and proper orientation.

### 6.4 Preclassification vs. Wheel-Only Topology

The classification topology is controlled by **`use_preclassification`** on `ClassificationSystemParams` (and thus when constructing the complete system via `classification_params=ClassificationSystemParams(use_preclassification=...)` or via the factory `create_core_connections_system(use_preclassification=...)`).

- **With preclassification** (`use_preclassification=True`, default): The classification assembly includes the venturi eductor, zigzag classifier, dropout hopper, wheel classifier, multi-cyclone system, and bag filter. Feed connects to the venturi solids inlet; air connects to the venturi air inlet. Coarse streams are collected at dropout, zigzag coarse outlet, and wheel coarse outlet.
- **Without preclassification** (`use_preclassification=False`, wheel-only): The assembly omits venturi, zigzag, and dropout. A **three-point junction** merges air (from the air system) and solids (from the feed system) and feeds the wheel classifier directly. The wheel classifier remains mandatory for the fine cut (~25 μm); multi-cyclone and bag filter are unchanged. This mode is used when feed is already pre-screened or when a simpler layout is desired.

In both topologies, the wheel classifier provides the centrifugal separation (1000–5000*g*) required for protein–starch cut sizes; the zigzag stage only appears when preclassification is enabled and provides a gravity-based pre-separation (~100 μm scale).

### 6.5 Factory Functions

| Function | Configuration |
|----------|--------------|
| `create_complete_classifier_system()` | 500 kg/h, 20 μm cut, full preclassification |
| `create_pilot_scale_system()` | Reduced throughput, compact footprint |
| `create_production_scale_system()` | 2000+ kg/h, industrial dimensions |
| `create_minimal_classifier_system()` | Classification core only; no feed, air, ductwork, support, or exhaust |
| `create_core_connections_system(use_preclassification=True/False)` | Classification + air + feed + ductwork + exhaust; no support structure. Use `use_preclassification=False` for wheel-only topology. |

---

## 7. Signed Distance Fields

### 7.1 Purpose and Formulation

The SDF provides an implicit surface representation where every point in 3D space has a signed distance to the nearest geometry surface:

$$\text{SDF}(\mathbf{p}) \begin{cases} < 0 & \text{inside the flow domain} \\ = 0 & \text{on the wall surface} \\ > 0 & \text{outside or inside solid walls} \end{cases}$$

This representation is essential for:

1. **Particle–wall collision detection**: A particle at position $\mathbf{p}$ with radius $r_p$ collides when $\text{SDF}(\mathbf{p}) + r_p \geq 0$.
2. **Wall-normal computation**: The outward normal at any point is $\hat{\mathbf{n}} = \nabla\text{SDF} / |\nabla\text{SDF}|$, computed via central differences with step size $\epsilon = 10^{-5}$ m.
3. **Region classification**: The sign and magnitude of the SDF at a particle's position determines which zone of the equipment it occupies.

### 7.2 CycloneSDF Implementation

The `CycloneSDF` class evaluates the complete cyclone distance field by partitioning the spatial domain into regions and applying the appropriate primitive SDF:

```
Region              | SDF Computation
─────────────────── | ────────────────────────────────────────
Above cylinder top  | Distance to nearest cap edge
Cylinder section    | d = r - R_cylinder
Cone section        | d_slant = (r - R_wall(y)) × h / s
Below cone tip      | Distance to dust outlet pipe boundary
Inside vortex finder| d = -(R_vf - r) (inside tube)
```

The `classify_region()` method returns a string label for the region containing a given point, enabling zone-based physics in the simulation.

### 7.3 SDFField: Discretized Volume

The `SDFField` class precomputes SDF values on a regular 3D grid for fast trilinear interpolation during simulation. Given grid bounds $[\mathbf{p}_\text{min}, \mathbf{p}_\text{max}]$ and resolution $(n_x, n_y, n_z)$:

1. Generate $n_x \times n_y \times n_z$ sample points.
2. Evaluate SDF at each point (batch on CPU or parallel kernel on GPU).
3. Store as a 3D array for $O(1)$ lookup with trilinear interpolation.

**Trilinear interpolation** at arbitrary point $\mathbf{p}$:

$$\mathbf{i} = \frac{\mathbf{p} - \mathbf{p}_\text{min}}{\Delta\mathbf{x}}, \quad \mathbf{t} = \mathbf{i} - \lfloor\mathbf{i}\rfloor$$

$$\text{SDF}(\mathbf{p}) \approx \sum_{c \in \{0,1\}^3} w_c \cdot \text{field}[\lfloor i_x\rfloor + c_x, \lfloor i_y\rfloor + c_y, \lfloor i_z\rfloor + c_z]$$

where $w_c = \prod_k |c_k - (1 - t_k)|$ are the trilinear weights.

### 7.4 Warp GPU Kernels for SDF

Six Warp constructs support GPU-accelerated SDF operations:

1. **Warp SDF parameter struct** (`@wp.struct`): A GPU-compatible parameter package holding the cyclone center (as a 3-vector) and all geometry dimensions (radii, heights, vortex finder and dust outlet dimensions).

2. **Cyclone SDF device function** (`@wp.func`): A device function that evaluates the piecewise signed distance (cylinder, cone, vortex finder, dust outlet) at a single point and is callable from within GPU kernels.

3. **SDF gradient device function** (`@wp.func`): Central-difference gradient of the SDF (six SDF evaluations per call) used to obtain outward normals.

4. **Compute SDF field kernel** (`@wp.kernel`): A parallel kernel where each thread evaluates the SDF at one grid point, filling a 3D field for fast lookup or visualization.

5. **Compute SDF gradient field kernel** (`@wp.kernel`): A parallel kernel that fills a 3D field of gradient vectors at each grid point using the same central-difference scheme.

6. **Classify points inside kernel** (`@wp.kernel`): A parallel kernel that writes a binary inside/outside flag for each point (e.g. for particle containment checks).

### 7.5 Visualization

A 2D slice visualization utility generates a contour plot at a fixed vertical (Y) height, with the zero-level contour highlighted to show the wall intersection. This is used to verify that the SDF correctly represents the geometry at different heights through the cyclone.

---

## 8. Mesh Generation and Export

### 8.1 Structured Grid Generation

The `MeshGenerator` class creates Cartesian and cylindrical grids suitable for CFD initialization and SDF sampling.

**Cartesian grid**: Uniform spacing in each direction using `np.linspace`. The `GridParams` dataclass computes cell size, total cell count, and cell volume. The `from_cyclone_bounds()` factory method adds a configurable padding margin and auto-calculates the number of divisions from a target cell size.

**Cylindrical grid**: Body-fitted coordinates for axisymmetric cyclone geometries:

$$(r_i, \theta_j, y_k) \to (r_i\cos\theta_j,\ y_k,\ r_i\sin\theta_j)$$

with hexahedral connectivity and periodic wrapping in the $\theta$ direction. This grid is well-suited for cyclone flow simulations where resolution should be concentrated near the cylinder wall.

### 8.2 Adaptive Refinement

The `generate_adaptive_points()` method creates non-uniform point distributions with increased density near geometry surfaces:

1. Evaluate SDF on a coarse base grid.
2. Identify cells where $|\text{SDF}| < d_\text{surface}$ (near-surface thickness parameter).
3. Subdivide near-surface cells to `near_surface_resolution` spacing.
4. Retain coarse cells in the far-field.

This reduces total point count while maintaining resolution where particle–wall interactions occur.

### 8.3 Marching Cubes Surface Extraction

Surface extraction applies the marching cubes algorithm to a 3D SDF field to obtain a triangulated iso-surface at $\text{SDF} = 0$. This yields a surface mesh derived directly from the implicit SDF rather than from parametric mesh generation, useful for visualization or export when only the SDF is available.

### 8.4 Mesh Quality Metrics

The `compute_mesh_quality()` method evaluates:

- **Triangle area statistics**: min, max, mean, total (via cross product $\frac{1}{2}|\mathbf{e}_1 \times \mathbf{e}_2|$).
- **Edge length statistics**: min, max across all edges.
- **Aspect ratio**: Maximum ratio of longest to shortest edge per triangle.

### 8.5 Export Formats

**STL (Stereolithography)**: Binary (compact) or ASCII format. Face normals are computed per-triangle as $\hat{\mathbf{n}} = (\mathbf{e}_1 \times \mathbf{e}_2) / |\mathbf{e}_1 \times \mathbf{e}_2|$. Binary STL uses an 80-byte header, 4-byte triangle count, and 50-byte records per triangle (12 bytes normal + 36 bytes vertices + 2 bytes attribute).

**VTK (Visualization Toolkit)**: Legacy ASCII or XML format, supporting optional point data arrays (e.g., SDF values, velocity magnitudes) and cell data arrays (e.g., pressure, zone labels) for post-processing in ParaView.

### 8.6 Sampling Point Generation

The `create_sampling_points()` function generates point clouds within a bounding box using three methods:

| Method | Distribution | Convergence | Use Case |
|--------|-------------|-------------|----------|
| Uniform | Pseudo-random | $O(N^{-1/2})$ | Quick visualization |
| Halton | Quasi-random (bases 2, 3, 5) | $O(N^{-1}\log^d N)$ | SDF sampling |
| Sobol | Quasi-random (direction numbers) | $O(N^{-1}\log^d N)$ | Integration |

The Halton sequence provides better space-filling properties than uniform random sampling, reducing the number of points needed to adequately resolve the SDF field.

---

## 9. Real-Time Visualization Pipeline

### 9.1 Architecture

The visualization pipeline uses **PyVista** (a VTK-based library in Python) for all 3D rendering. Geometry is produced in Python as vertex and index arrays from the parametric models, then passed to the renderer as polygonal data for display. NVIDIA Warp is not used for drawing; it is used only for GPU-accelerated simulation (particle dynamics, SDF evaluation) when a CUDA device is selected. The same pipeline can display the air system alone, the feed system alone, the classification system alone, or the complete integrated system (or all in sequence). The architecture separates static geometry (rendered once) from animated geometry (updated each frame):

```
┌─────────────────────────────────────────┐
│              PyVista Plotter             │
│  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │  Static   │  │ Animated │  │  Info │ │
│  │  Meshes   │  │  Meshes  │  │ Panel │ │
│  │ (housing, │  │(impeller,│  │(state,│ │
│  │  ducts,   │  │  rotor,  │  │  RPM, │ │
│  │  filter)  │  │  blade,  │  │ flow) │ │
│  └──────────┘  │  lid,    │  └───────┘ │
│                │particles) │            │
│                └──────────┘            │
│                                         │
│  Camera: Y-up, azimuth=-170°,           │
│          elevation=-20°                 │
└─────────────────────────────────────────┘
         ▲                    ▲
         │ generate_mesh()    │ step()
         │                    │ get_results()
    ┌────┴────┐          ┌────┴─────┐
    │ Geometry │          │ Simulation│
    │ Assembly │          │  (physics)│
    └─────────┘          └──────────┘
```

### 9.2 Static Mesh Rendering

For each assembly, component meshes are generated once and uploaded to the VTK renderer:

```python
v, i, _ = component.generate_mesh()
v = v + np.array(component_position)  # World-space offset
faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
mesh = pv.PolyData(v, faces)
plotter.add_mesh(mesh, color=COLORS[name], label=label, opacity=0.85)
```

The face array format prepends a vertex count (3 for triangles) before each face's vertex indices, as required by VTK's `PolyData` structure.

A consistent color palette assigns distinguishable colors to each component category (blue for air, green for feed, orange for dampers, red for cyclones, etc.) with legends for identification.

### 9.3 Animated Mesh Updates

Components with moving parts split their geometry into static and animated portions. The animation loop:

1. Queries the simulator for current state (RPM, damper position, lid angle).
2. Calls the component's animated mesh method (e.g., `get_impeller_mesh()`, `get_blade_mesh(position)`).
3. Updates the existing PyVista mesh's vertex array **in-place** using `mesh.points[:] = new_vertices` and marks the mesh as modified with `mesh.Modified()`.

In-place vertex updates avoid the overhead of removing and re-adding actors, enabling smooth animation at 30 FPS.

**Blower animation** example (three synchronized meshes):

```python
# 1. Advance animation state
blower.update_animation(wall_dt, current_rpm)

# 2. Update impeller vertices (blower RPM)
v_imp, _, _ = blower.get_impeller_mesh()
animated_actors['impeller']['mesh'].points[:] = v_imp + offset
animated_actors['impeller']['mesh'].Modified()

# 3. Update driven pulley (same speed as impeller)
v_dp, _, _ = blower.get_driven_pulley_mesh()
animated_actors['driven_pulley']['mesh'].points[:] = v_dp + offset

# 4. Update motor pulley (faster by pulley ratio)
v_mp, _, _ = blower.get_motor_pulley_mesh()
animated_actors['motor_pulley']['mesh'].points[:] = v_mp + offset
```

**Damper animation** uses a target-tracking approach with configurable transition time:

```python
damper.update_animation(wall_dt, target_position, transition_time=1.0)
current_pos = damper.get_blade_position()
v_blade, _, _ = damper.get_blade_mesh(current_pos)
```

**Feed hopper lid** rotation uses Rodrigues' formula to rotate all lid vertices about the hinge axis:

$$\mathbf{v}' = \mathbf{v}\cos\alpha + (\hat{\mathbf{k}} \times \mathbf{v})\sin\alpha + \hat{\mathbf{k}}(\hat{\mathbf{k}} \cdot \mathbf{v})(1 - \cos\alpha)$$

where $\hat{\mathbf{k}}$ is the hinge axis direction and $\alpha$ is the opening angle.

### 9.4 Particle Visualization

Feed system particles are rendered as a colored point cloud:

```python
positions = simulator.get_particle_positions()
velocities = simulator.get_particle_velocities()
speeds = np.linalg.norm(velocities, axis=1)

particle_mesh = pv.PolyData(positions)
particle_mesh['velocity'] = speeds

plotter.add_mesh(particle_mesh,
    scalars='velocity', cmap='YlOrBr',
    point_size=12, render_points_as_spheres=True,
    clim=[0, 1.5], show_scalar_bar=False)
```

Velocity-based coloring uses a brown/tan colormap (`YlOrBr`) to approximate the appearance of flour-like powder. Point size scales with the visual particle diameter parameter.

Particle actors are removed and re-created each frame (rather than updated in-place) because the number of active particles changes dynamically during pouring and discharge phases.

### 9.5 Information Overlay

A text panel in the upper-left corner displays real-time simulation state using PyVista's `add_text()` with a named actor for flicker-free updates:

```python
plotter.add_text(text, position='upper_left', font_size=12,
                color='black', name='sim_info')
```

The named actor (`name='sim_info'`) causes PyVista to replace the existing text actor rather than adding a new one, preventing accumulation of text layers.

### 9.6 Simulation Loop Timing

The visualization loop decouples simulation time from wall-clock time for smooth animation:

```python
target_fps = 30
frame_interval = 1.0 / target_fps
steps_per_frame = max(1, int(frame_interval / config.dt))

while step < total_steps:
    # Run multiple sim steps per visual frame
    for _ in range(steps_per_frame):
        simulator.step()
        step += 1

    # Update animated meshes using wall-clock dt
    wall_dt = min(time.time() - last_wall_time, 0.1)  # Clamped
    blower.update_animation(wall_dt, current_rpm)

    # Render frame
    plotter.update()
```

Wall-clock delta time is clamped to 0.1 s to prevent animation jumps during frame drops.

### 9.7 Shutdown Sequence

The air system visualization includes a physically-motivated shutdown animation:

1. Dampers close gradually (1.0 → 0.0 over 2 seconds) to prevent backflow.
2. Blower decelerates (RPM → 0) as dampers close.
3. Motor and driven pulleys slow proportionally.
4. Final state renders with "OFF" status and energy consumption summary.

---

## 10. NVIDIA Warp Integration

### 10.1 Role in the Framework

NVIDIA Warp is used for **simulation and geometry evaluation**, not for rendering. In the geometry and simulation pipeline it serves:

1. **SDF evaluation on GPU**: Parallel computation of signed distance values for thousands of particle–wall queries per time step, via the cyclone SDF kernels described in Section 7.4.
2. **Mesh representation**: `wp.Mesh` objects for hardware-accelerated spatial queries (ray tracing, closest-point) when simulation uses GPU.
3. **Physics kernels**: Particle dynamics, SPH fluid simulation, and CFD-DEM coupling kernels that operate on geometry-derived parameters.

Rendering remains in Python with PyVista; simulation can run on CPU or GPU via Warp's `device` parameter.

### 10.2 Kernel Architecture

The framework contains many Warp kernels across geometry and simulation. Representative roles are:

| Kernel role | Purpose |
|-------------|---------|
| SDF field computation | Parallel SDF evaluation on a 3D grid; each thread evaluates the cyclone SDF at one point. |
| SDF gradient field | Parallel computation of SDF gradients (surface normals) at grid points via central differences. |
| Point classification | Binary inside/outside classification of points against the cyclone (or other) geometry. |
| Feed / particle flow | Particle dynamics with zone-based geometry (hopper, airlock, feeder, deagglomerator). |
| Particle collisions | Hash-grid neighbor search and impulse-based collision response for granular flow. |
| SPH density and pressure | Smoothed-particle hydrodynamics density via Poly6 kernel. |
| SPH forces | Pressure gradient (Spiky kernel) and viscosity for fluid phase. |
| Boundary containment | Enforcement of duct and vessel geometry constraints on fluid/particle motion. |
| Blower acceleration | Centrifugal impeller force model for air flow generation. |

### 10.3 GPU Memory Management

All GPU arrays are pre-allocated at simulator initialization and reused across time steps:

```python
# Persistent arrays (allocated once)
self.positions = wp.zeros(n, dtype=wp.vec3, device=device)
self.velocities = wp.zeros(n, dtype=wp.vec3, device=device)
self.diameters = wp.zeros(n, dtype=float, device=device)
self.zones = wp.zeros(n, dtype=wp.int32, device=device)
```

This zero-allocation-per-step strategy avoids GPU memory fragmentation and eliminates the overhead of repeated `cudaMalloc`/`cudaFree` calls. A single `wp.synchronize()` call at the end of each time step ensures all kernel results are available before CPU-side post-processing.

### 10.4 Geometry-Physics Coupling

Geometry extraction functions convert assembly objects into flat parameter structures for GPU kernels:

```python
def extract_geometry(assembly: FeedSystemAssembly) -> Dict[str, ComponentGeometry]:
    """Extract geometric parameters from actual component dimensions.

    NO magic numbers — all values derived from component params and port positions.
    """
    hopper_geo = ComponentGeometry(
        center=assembly._hopper_position,
        radius=assembly.hopper.params.top_radius,
        length=assembly.hopper.params.total_height,
        inlet_position=assembly.hopper.ports['inlet'].position,
        outlet_position=assembly.hopper.ports['outlet'].position,
        ...
    )
```

This ensures that physics simulations always operate on the actual geometry dimensions, maintaining consistency between what is visualized and what is simulated.

### 10.5 Device Flexibility

All components support both CPU and GPU execution:

```python
device = "cuda" if wp.is_cuda_available() else "cpu"
mesh = component.to_warp_mesh(device=device)
```

CPU fallback enables development and testing on systems without NVIDIA GPUs, while CUDA acceleration provides the performance needed for real-time simulation of 5000+ particles with collision detection.

### 10.6 Dual Particle Scale Architecture

The feed system maintains two particle diameter scales:

- **Visual diameter** (e.g., 15 mm): Sized so that the configured number of visual particles (5000 default) fills the hopper volume visually. Used for PyVista rendering.
- **Physical diameter** (e.g., 10–100 μm): Actual particle sizes used for drag, settling, and classification physics.

This dual-scale approach allows the visualization to show a representative filling and flow behavior with a tractable number of large spheres, while the physics kernels operate on realistic particle sizes. When particles transfer from the feed system to the classification system, physical diameters are used:

```python
transfer_data = feed_simulator.get_particle_data_for_transfer()
# transfer_data contains PHYSICAL diameters (micron-scale)
classification_simulator.inject_particles_from_feed(transfer_data)
```

---

## 11. Concluding Remarks and Directions for Future Manuscripts

### 11.1 Summary of Contributions

This geometry modeling framework provides:

1. **A parametric, composable architecture** where every industrial component from a primitive cylinder to a complete multi-stage classification plant is defined by engineering-meaningful parameters with no embedded constants.

2. **Dual geometry representations** (explicit triangulated mesh + implicit SDF) from a single parameter set, enabling both visualization-quality rendering and physics-accurate particle–wall interaction.

3. **A port-based spatial assembly system** that automates component positioning, dimensional compatibility checking, and transition piece insertion.

4. **GPU-accelerated geometry operations** through NVIDIA Warp's JIT-compiled kernels, supporting real-time SDF evaluation, gradient computation, and parallel particle physics.

5. **Two classification topologies** (preclassification with venturi + zigzag + wheel, or direct wheel-only) configurable from a single parameter set.

### 11.2 Potential Manuscript Topics

The following research directions emerge naturally from this framework:

1. **Parametric sensitivity of cyclone geometry on protein separation efficiency**: Using the parameterized cyclone and classification assemblies to systematically vary diameter ratios, cone angles, and vortex finder depths, and quantifying the effect on $d_{50}$ and separation sharpness via GPU-accelerated Lagrangian particle tracking.

2. **Comparative evaluation of zigzag + wheel vs. wheel-only classification topologies**: Leveraging the two built-in topologies to compare separation performance, pressure drop, and energy consumption under matched throughput conditions.

3. **GPU-accelerated CFD-DEM simulation of food powder air classification**: Documenting the Warp kernel architecture, SPH fluid solver, and Lagrangian particle physics with validation against experimental fractionation data.

4. **Real-time digital twin for air classifier process control**: Describing the PyVista visualization pipeline, animated component rendering, and live simulation overlay as a basis for operator training and process monitoring.

5. **Equipment-scale geometry generation for additive manufacturing of pilot-scale classifiers**: Using the STL/VTK export pipeline with adaptive mesh refinement to produce print-ready geometries for metal 3D printing of miniature cyclone and zigzag classifier prototypes.

6. **Port-based assembly automation for modular process equipment design**: Generalizing the connection port system beyond air classification to arbitrary process equipment configurations.

---

## Appendix A: Mesh Statistics for Default Configurations

| Assembly | Components | Total Vertices | Total Triangles |
|----------|-----------|---------------|-----------------|
| Single Cyclone | 5 | ~3,200 | ~5,800 |
| Feed System | 4 + transitions | ~8,500 | ~14,000 |
| Air System | 3 + ductwork | ~12,000 | ~20,000 |
| Classification System | 6 + ductwork | ~25,000 | ~42,000 |
| Complete System | All subsystems | ~65,000 | ~110,000 |

## Appendix B: Key Engineering Parameters (Pilot Scale)

| Parameter | Value | Unit |
|-----------|-------|------|
| Throughput | 500 | kg/h |
| Air flow rate | 3000 | m³/h |
| System pressure drop | 5000 | Pa |
| Blower power | ~5 | kW |
| Zigzag channel width | 150 | mm |
| Zigzag stages | 5 | — |
| Wheel classifier diameter | 200 | mm |
| Wheel RPM | 8000 | RPM |
| Wheel $d_{50}$ | 25 | μm |
| Primary cyclone diameter | 300 | mm |
| Secondary cyclone diameter | 200 | mm |
| Tertiary cyclone diameter | 120 | mm |
| Hopper capacity | 500 | kg |
| Airlock rotor diameter | 200 | mm |
| Screw feeder diameter | 100 | mm |
| Deagglomerator RPM | 1500 | RPM |

## Appendix C: Warp Kernel and Function Count by Domain

| Domain | Kernels | Device functions (`@wp.func`) | Structs |
|--------|---------|-------------------------------|---------|
| SDF (signed distance fields) | 3 | 2 | 1 |
| Geometric primitives | 0 | 4 | 0 |
| Feed / particle flow physics | 2 | 0 | 0 |
| Air / fluid flow physics | 5 | 0 | 0 |
| CFD–DEM coupling | 2 | 0 | 0 |
| Particle system | 5+ | 2+ | 3 |
| **Total** | **17+** | **8+** | **4** |

---

*This technical note forms the Geometry Modeling section of a comprehensive documentation of the computational air classification framework. Subsequent sections will address particle physics, fluid dynamics, material modeling, and experimental validation.*
