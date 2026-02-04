# Complete System Flow Analysis

## What's Missing in Current Simulation

Looking at your complete system visualization, there are **three critical flow paths** that need proper physics modeling:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE SYSTEM FLOW PATHS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FEED SYSTEM (Green)                    CLASSIFICATION (Blue)           │
│  ┌──────────────┐                      ┌────────────────────┐          │
│  │   Hopper     │                      │    Bag Filter      │          │
│  │     ↓        │                      │        ↑           │          │
│  │  Airlock     │                      │    Cyclones ×3     │          │
│  │     ↓        │                      │        ↑           │          │
│  │Screw Feeder  │                      │     Zigzag         │          │
│  │     ↓        │                      │        ↑           │          │
│  │Deagglomerator│                      │     Venturi        │          │
│  └──────┬───────┘                      └────────┬───────────┘          │
│         │                                        │                      │
│         │  FEED CHUTE (15°)                      │                      │
│         │  ──────────────────────►  SOLIDS_INLET │                      │
│         │  (gravity + air assist)                │                      │
│                                                  │                      │
│                                           AIR_INLET                     │
│                                                  ↑                      │
│                                                  │                      │
│  AIR SYSTEM (Orange)                             │                      │
│  ┌──────────────┐                               │                      │
│  │   Filter     │                               │                      │
│  │     ↓        │                               │                      │
│  │   Blower     │──► Damper0 ──► Damper1 ──────┘                       │
│  │  (1768 m³/h) │    (100%)      (100%)   AIR DUCT PATH                │
│  └──────────────┘                          (with elbows)                │
│                                                                         │
│  EXHAUST (Red/Purple)                                                   │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Bag Filter Clean Air ──► Silencer ──► Stack ──► Atm    │          │
│  └──────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Critical Missing Physics

### 1. Air Ductwork Pressure Drops

The air path from dampers to venturi includes multiple segments that aren't accounted for:

| Segment | Description | Missing Calculation |
|---------|-------------|---------------------|
| Duct 1 | Short straight after damper | Friction loss |
| Elbow 1 | +X to -Z turn | Minor loss (K~0.3) |
| Duct 2 | Vertical drop in -Z | Friction loss |
| Elbow 2 | -Z to +Y turn | Minor loss (K~0.3) |
| Duct 3 | Toward classifier +Y | Friction loss |
| Elbow 3 | +Y to -X turn | Minor loss (K~0.3) |
| Duct 4 | Horizontal -X | Friction loss |
| Elbow 4 | -X to +Y final approach | Minor loss (K~0.3) |
| Duct 5 | Final approach | Friction loss |
| Transition | Round-to-round to venturi | Minor loss (K~0.1) |

**Total path length: ~3-4 meters with 4 elbows**

### 2. Feed Chute Dynamics

The 15° gravity chute has important physics:

| Parameter | Effect |
|-----------|--------|
| Chute angle (15°) | Controls particle velocity |
| Chute diameter | Affects air entrainment |
| Particle loading | Changes effective density |
| Air leakage | Can affect venturi operation |

### 3. Solids Loading at Venturi

When particles enter the venturi, they affect the airflow:

| Effect | Impact |
|--------|--------|
| Momentum exchange | Particles accelerated by air reduce air velocity |
| Effective density | Two-phase flow has higher effective density |
| Pressure drop | Additional dP from particle acceleration |

---

## Complete System Pressure Balance

### Current Understanding (Incomplete)

Your current simulation calculates:
- Blower output: 1768 m³/h at 2500 RPM
- Venturi velocities: inlet 97.7 m/s, throat 390.8 m/s
- Zigzag velocity: 20.46 m/s

**BUT** it doesn't account for the pressure drops in the connecting ductwork!

### Required System Pressure Balance

```
P_blower = ΔP_dampers + ΔP_air_ductwork + ΔP_venturi + ΔP_zigzag 
         + ΔP_cyclones + ΔP_bag_filter + ΔP_exhaust_duct
```

Let me calculate each term:

---

## Detailed Pressure Drop Calculations

### Air Ductwork Path: Dampers → Venturi

From your complete_system.py, the air duct parameters are:
- Duct diameter: 0.2 m (200 mm)
- Elbow bend radius: R/D = 1.0
- Number of elbows: 4
- Total duct length: ~3.5 m (estimated from geometry)

```python
def calculate_air_ductwork_pressure_drop(Q_m3_h, duct_d=0.2, total_length=3.5, 
                                          num_elbows=4, air_density=1.204,
                                          air_viscosity=1.82e-5):
    """
    Calculate pressure drop through air ductwork from dampers to venturi.
    
    Args:
        Q_m3_h: Volumetric flow rate [m³/h]
        duct_d: Duct diameter [m]
        total_length: Total duct length [m]
        num_elbows: Number of 90° elbows
        air_density: Air density [kg/m³]
        air_viscosity: Dynamic viscosity [Pa·s]
    
    Returns:
        dict with pressure drops [Pa]
    """
    import numpy as np
    
    Q = Q_m3_h / 3600  # m³/s
    A = np.pi * (duct_d / 2) ** 2
    v = Q / A
    
    # Reynolds number
    Re = air_density * v * duct_d / air_viscosity
    
    # Friction factor (Colebrook-White, smooth pipe approximation)
    if Re < 2300:
        f = 64 / Re
    else:
        # Blasius equation for smooth pipes
        f = 0.316 / (Re ** 0.25)
    
    # Dynamic pressure
    q = 0.5 * air_density * v ** 2
    
    # Straight duct friction loss
    dP_friction = f * (total_length / duct_d) * q
    
    # Elbow losses (K = 0.3 for R/D = 1.0, 90° bend)
    K_elbow = 0.3
    dP_elbows = num_elbows * K_elbow * q
    
    # Transition loss (contraction/expansion)
    K_transition = 0.1
    dP_transition = K_transition * q
    
    # Total
    dP_total = dP_friction + dP_elbows + dP_transition
    
    return {
        'velocity_m_s': v,
        'reynolds': Re,
        'friction_factor': f,
        'dynamic_pressure_Pa': q,
        'friction_loss_Pa': dP_friction,
        'elbow_loss_Pa': dP_elbows,
        'transition_loss_Pa': dP_transition,
        'total_loss_Pa': dP_total,
    }
```

### For your system (1768 m³/h, D=200mm):

```
Q = 1768 m³/h = 0.491 m³/s
A = π × 0.1² = 0.0314 m²
v = 0.491 / 0.0314 = 15.6 m/s
Re = 1.204 × 15.6 × 0.2 / 1.82e-5 = 206,000 (turbulent)
f = 0.316 / 206000^0.25 = 0.0148

q = 0.5 × 1.204 × 15.6² = 146.5 Pa

dP_friction = 0.0148 × (3.5/0.2) × 146.5 = 38 Pa
dP_elbows = 4 × 0.3 × 146.5 = 176 Pa
dP_transition = 0.1 × 146.5 = 15 Pa

TOTAL AIR DUCTWORK LOSS = 229 Pa
```

### Venturi Pressure Drop

From your simulation diagnostics:
- Throat suction: 85.9 kPa (this is the Bernoulli pressure drop)
- But this is recovered - the NET loss is much lower

For a well-designed venturi:
```
dP_net = K × q_inlet
K ≈ 0.1-0.2 for a good venturi
q_inlet = 0.5 × 1.204 × 97.7² = 5,746 Pa

dP_venturi_net ≈ 0.15 × 5746 = 862 Pa
```

### Classification System Pressure Drops

| Component | Velocity | K or Method | ΔP |
|-----------|----------|-------------|-----|
| Zigzag (5 stages) | 20.5 m/s | K~2.0 per stage | 10 × 253 = 2,530 Pa |
| Duct to cyclones | 65.8 m/s | Friction + elbow | ~150 Pa |
| Primary cyclone | 173.7 m/s | K~6-8 | ~13,000 Pa |
| Secondary cyclone | 139 m/s | K~6-8 | ~8,400 Pa |
| Tertiary cyclone | 104 m/s | K~6-8 | ~4,700 Pa |
| Bag filter | 13 cm/s | 1000-2000 Pa typical | ~1,500 Pa |

**This reveals a major issue:** The cyclone pressure drops alone are ~26,000 Pa, but your blower only delivers 1,736 Pa!

---

## The Real Problem: Complete System Analysis

### Blower Operating Point

Your blower at 2500 RPM delivers:
- Flow: 1768 m³/h
- Pressure: 1,736 Pa

### System Resistance Curve

The total system resistance is:
```
ΔP_total = ΔP_ductwork + ΔP_venturi + ΔP_classification

At Q = 1768 m³/h:
- Ductwork: ~230 Pa
- Venturi net: ~860 Pa  
- Zigzag: ~2,500 Pa (but effectively bypassed at this flow)
- Cyclones: ~26,000 Pa (if all in series)
- Bag filter: ~1,500 Pa

TOTAL: ~31,000 Pa >> 1,736 Pa available!
```

### What This Means

**The current simulation is physically impossible.** The blower cannot deliver 1768 m³/h through this system because:

1. System resistance (~31,000 Pa) >> Blower pressure (1,736 Pa)
2. The actual flow would be MUCH lower (where blower curve intersects system curve)

### Why the Simulation "Works"

The simulation treats each component separately:
- Air system delivers "1768 m³/h" (its own calculation)
- Classification system receives "1768 m³/h" (passed as parameter)
- No check that these are compatible!

---

## Solids Loading Effects on Venturi

### Integrated Model (feedclass_flow_physics)

The feed system delivers particles to the venturi's solids_inlet. The following are computed from geometry and physics (no magic numbers):

1. **Momentum Transfer**: Particles entering at chute exit velocity (from `feedclass_flow_physics` kinetics) must be accelerated to air velocity at the venturi throat; momentum balance gives additional pressure drop.
2. **Effective Density**: Two-phase mixture density from volume fraction (solids + air) using venturi throat area and flow rates from assembly geometry.
3. **Loading Ratio**: μ = ṁ_solids / ṁ_air, with ṁ_solids from feed system throughput (e.g. `CompleteSystemParams.throughput_kg_h` or `FeedSystemParams.feeder_target_rate_kg_h`) and ṁ_air from air flow.

Run feed-to-venturi ductwork and kinetics: `python examples/run_physics_flow.py --no-sim --feed-to-venturi`. Segment geometry (lengths, directions, venturi solids inlet diameter) comes from `CompleteClassifierAssembly.get_feed_to_venturi_ductwork()` and classification subsystem venturi ports.

### Solids Loading Calculations

```python
def calculate_solids_loading_effect(
    air_flow_m3_h: float = 1768,
    solids_flow_kg_h: float = 755,  # From feed system
    air_density: float = 1.204,
    particle_density: float = 1420,
    venturi_throat_velocity: float = 390.8,
    particle_entry_velocity: float = 2.0,  # Gravity chute exit ~2 m/s
):
    """
    Calculate effect of solids loading on venturi operation.
    """
    import numpy as np
    
    # Mass flow rates
    m_dot_air = air_flow_m3_h / 3600 * air_density  # kg/s
    m_dot_solids = solids_flow_kg_h / 3600  # kg/s
    
    # Loading ratio
    mu = m_dot_solids / m_dot_air
    
    # Momentum balance at throat
    # Air momentum: m_dot_air × v_air
    # Particle momentum in: m_dot_solids × v_particle_entry
    # Particle momentum out: m_dot_solids × v_particle_exit
    
    # For particles to reach air velocity, air must supply momentum
    momentum_transfer = m_dot_solids * (venturi_throat_velocity - particle_entry_velocity)
    
    # This appears as additional pressure drop
    # Force = momentum rate, Pressure = Force / Area
    throat_area = np.pi * (0.04 / 2) ** 2  # 40mm throat
    dP_solids_acceleration = momentum_transfer / throat_area / venturi_throat_velocity
    
    # Effective mixture density at throat
    # Volume fraction of solids is small, but mass loading is significant
    vol_flow_air = air_flow_m3_h / 3600  # m³/s
    vol_flow_solids = m_dot_solids / particle_density  # m³/s
    
    solid_volume_fraction = vol_flow_solids / (vol_flow_air + vol_flow_solids)
    
    # Mixture density (volume weighted)
    rho_mixture = (1 - solid_volume_fraction) * air_density + solid_volume_fraction * particle_density
    
    return {
        'air_mass_flow_kg_s': m_dot_air,
        'solids_mass_flow_kg_s': m_dot_solids,
        'loading_ratio': mu,
        'momentum_transfer_N': momentum_transfer,
        'pressure_drop_solids_Pa': dP_solids_acceleration,
        'solid_volume_fraction': solid_volume_fraction,
        'mixture_density_kg_m3': rho_mixture,
    }
```

### For your system:

```
Air: 1768 m³/h × 1.204 kg/m³ / 3600 = 0.591 kg/s
Solids: 755 kg/h / 3600 = 0.210 kg/s

Loading ratio μ = 0.210 / 0.591 = 0.355 (35.5% by mass)

Momentum transfer = 0.210 × (390.8 - 2.0) = 81.6 N

Additional pressure drop from solids = 81.6 / (0.00126 m² × 390.8 m/s) = 166 Pa
```

This is actually a relatively small effect compared to the ductwork and cyclone losses.

---

## Feed Chute Physics

### Gravity Chute at 15° (from geometry)

The feed-to-venturi chute angle and segment geometry are defined in `CompleteClassifierAssembly._build_feed_to_solids_inlet()`: shaft angle 15° from horizontal, segment directions and lengths from ductwork components. The chute angle is exposed as `get_feed_to_venturi_chute_angle_deg()`.

Particle velocity and residence time along each segment are computed in `feedclass_flow_physics.compute_particle_kinetics_feed()` from physics only:

- Angle from horizontal θ derived from segment direction (unit vector).
- Effective acceleration: a_eff = g × (sin(θ) − μ × cos(θ)) with μ = 0.4 (powder on steel, `FRICTION_POWDER_STEEL`).
- Particle velocity along chute: v = √(2 × a_eff × L) when a_eff > 0, capped by terminal velocity; residence time = L / v.

No magic numbers: segment lengths and directions come from `get_feed_to_venturi_ductwork()`; particle diameter and density from config/material.

The particles flow down the 15° chute under gravity:

```python
def calculate_chute_particle_velocity(
    chute_angle_deg: float = 15.0,
    chute_length: float = 1.5,  # meters
    particle_density: float = 1420,
    friction_coefficient: float = 0.4,  # Powder on steel
):
    """
    Calculate particle velocity at chute exit.
    
    For a particle sliding down an inclined chute:
    a = g × (sin(θ) - μ × cos(θ))
    
    If a > 0, particle accelerates
    v_exit = √(2 × a × L) (starting from rest)
    """
    import numpy as np
    
    g = 9.81
    theta = np.radians(chute_angle_deg)
    
    # Acceleration
    a = g * (np.sin(theta) - friction_coefficient * np.cos(theta))
    
    if a <= 0:
        # Chute angle too shallow - particles won't flow!
        return {
            'flows': False,
            'min_angle_deg': np.degrees(np.arctan(friction_coefficient)),
            'acceleration': a,
        }
    
    # Exit velocity (from rest)
    v_exit = np.sqrt(2 * a * chute_length)
    
    # Time to traverse
    t_traverse = np.sqrt(2 * chute_length / a)
    
    return {
        'flows': True,
        'acceleration_m_s2': a,
        'exit_velocity_m_s': v_exit,
        'traverse_time_s': t_traverse,
        'chute_angle_deg': chute_angle_deg,
        'chute_length_m': chute_length,
    }
```

### For 15° chute:

```
sin(15°) = 0.259
cos(15°) = 0.966
μ = 0.4 (powder on steel)

a = 9.81 × (0.259 - 0.4 × 0.966) = 9.81 × (0.259 - 0.386) = -1.25 m/s²

PROBLEM: At 15° with μ=0.4, particles WON'T flow by gravity alone!
Minimum angle = arctan(0.4) = 21.8°
```

This suggests the chute needs:
- Steeper angle (>22°), OR
- Air assist (pneumatic conveying), OR
- Vibration, OR
- Lower friction (polished surface, μ~0.25 → min angle 14°)

---

## Recommended Complete System Physics Model

### Integrated Flow Calculator

```python
class CompleteSystemFlowModel:
    """
    Integrated flow model for complete air classification system.
    
    Accounts for:
    - Blower curve (flow vs pressure)
    - All ductwork pressure drops
    - Venturi operation with solids loading
    - Classification system resistance
    - Solids feed dynamics
    """
    
    def __init__(self, params=None):
        self.params = params or {}
        
        # Blower parameters (from your air_system simulation)
        self.blower_rpm = 2500
        self.blower_design_flow_m3_h = 3000
        self.blower_design_pressure_Pa = 5000
        self.blower_design_rpm = 3000
        
        # Ductwork parameters
        self.air_duct_diameter = 0.2  # m
        self.air_duct_length = 3.5  # m
        self.air_duct_num_elbows = 4
        
        # Feed parameters
        self.solids_feed_rate_kg_h = 755
        self.chute_angle_deg = 15
        self.chute_length = 1.5
        
        # Classification parameters
        self.zigzag_area = 0.024  # m² (120×200mm)
        self.cyclone_d50_design = [40e-6, 20e-6, 10e-6]
        
    def blower_curve(self, Q_m3_h):
        """
        Calculate blower pressure at given flow.
        Uses affinity laws from design point.
        
        Simplified: P = P_design × (1 - (Q/Q_design)²)
        """
        Q_ratio = Q_m3_h / (self.blower_design_flow_m3_h * 
                           (self.blower_rpm / self.blower_design_rpm))
        
        P_max = self.blower_design_pressure_Pa * (self.blower_rpm / self.blower_design_rpm) ** 2
        
        # Simplified fan curve
        P = P_max * (1 - Q_ratio ** 2)
        return max(P, 0)
    
    def system_resistance(self, Q_m3_h):
        """
        Calculate total system pressure drop at given flow.
        """
        if Q_m3_h <= 0:
            return 0
        
        Q = Q_m3_h / 3600
        rho = 1.204
        
        # Air ductwork
        A_duct = np.pi * (self.air_duct_diameter / 2) ** 2
        v_duct = Q / A_duct
        q_duct = 0.5 * rho * v_duct ** 2
        
        Re = rho * v_duct * self.air_duct_diameter / 1.82e-5
        f = 0.316 / (Re ** 0.25) if Re > 2300 else 64 / Re
        
        dP_duct_friction = f * (self.air_duct_length / self.air_duct_diameter) * q_duct
        dP_duct_elbows = self.air_duct_num_elbows * 0.3 * q_duct
        dP_ductwork = dP_duct_friction + dP_duct_elbows
        
        # Venturi (simplified)
        v_venturi_inlet = Q / (np.pi * 0.04 ** 2)
        q_venturi = 0.5 * rho * v_venturi_inlet ** 2
        dP_venturi = 0.15 * q_venturi
        
        # Zigzag
        v_zigzag = Q / self.zigzag_area
        q_zigzag = 0.5 * rho * v_zigzag ** 2
        dP_zigzag = 10 * q_zigzag  # K~10 for 5 stages
        
        # Cyclones (series)
        # Inlet velocities scale with flow
        dP_cyclones = 0  # Would need detailed calculation
        # Simplified: use empirical correlation
        v_cy1_inlet = Q / (0.075 * 0.15)  # Primary inlet area
        q_cy1 = 0.5 * rho * v_cy1_inlet ** 2
        dP_cyclones = 7 * q_cy1  # K~7 for all cyclones combined
        
        # Bag filter
        face_velocity = Q / 3.68  # ~3.68 m² filter area
        dP_bagfilter = 1500 * (face_velocity / 0.02) ** 1.8  # Empirical
        
        # Total
        return dP_ductwork + dP_venturi + dP_zigzag + dP_cyclones + dP_bagfilter
    
    def find_operating_point(self):
        """
        Find intersection of blower curve and system resistance.
        
        This is the ACTUAL operating flow rate.
        """
        from scipy.optimize import brentq
        
        def residual(Q):
            return self.blower_curve(Q) - self.system_resistance(Q)
        
        # Find where curves intersect
        try:
            Q_operating = brentq(residual, 1, self.blower_design_flow_m3_h)
        except:
            Q_operating = 0  # No intersection - blower can't overcome resistance
        
        return {
            'operating_flow_m3_h': Q_operating,
            'operating_pressure_Pa': self.blower_curve(Q_operating),
            'system_resistance_Pa': self.system_resistance(Q_operating),
        }
    
    def calculate_complete_system_state(self, Q_m3_h=None):
        """
        Calculate complete system state at given flow rate.
        If Q not specified, finds operating point.
        """
        if Q_m3_h is None:
            op = self.find_operating_point()
            Q_m3_h = op['operating_flow_m3_h']
        
        Q = Q_m3_h / 3600
        rho = 1.204
        
        # Ductwork
        A_duct = np.pi * (self.air_duct_diameter / 2) ** 2
        v_duct = Q / A_duct
        
        # Zigzag
        v_zigzag = Q / self.zigzag_area
        
        # Cut size
        mu = 1.82e-5
        g = 9.81
        rho_p = 1420
        d50 = np.sqrt(18 * mu * v_zigzag / (g * (rho_p - rho)))
        
        # Solids loading
        m_air = Q * rho
        m_solids = self.solids_feed_rate_kg_h / 3600
        loading = m_solids / m_air
        
        return {
            'flow_m3_h': Q_m3_h,
            'duct_velocity_m_s': v_duct,
            'zigzag_velocity_m_s': v_zigzag,
            'd50_um': d50 * 1e6,
            'solids_loading_ratio': loading,
            'blower_pressure_Pa': self.blower_curve(Q_m3_h),
            'system_resistance_Pa': self.system_resistance(Q_m3_h),
        }
```

---

## Summary: What Needs to Be Added

### 1. System Pressure Balance Check

Before running simulation, verify:
```python
def validate_system_pressure_balance(blower_pressure, system_resistance):
    if system_resistance > blower_pressure:
        raise ValueError(
            f"System resistance ({system_resistance:.0f} Pa) exceeds "
            f"blower capacity ({blower_pressure:.0f} Pa). "
            f"Flow will be lower than expected!"
        )
```

### 2. Ductwork Pressure Drop Calculator

Add to `complete_system.py`:
```python
def calculate_ductwork_pressure_drops(self, Q_m3_h):
    """Calculate pressure drops through all connecting ductwork."""
    # Iterate through self._duct_connections
    # Sum friction and minor losses
    pass
```

### 3. Solids Loading Correction

Add to venturi/classification physics:
```python
def apply_solids_loading_correction(self, Q_air, m_solids):
    """Correct air velocity for momentum exchange with particles."""
    pass
```

### 4. Feed Chute Validation

Check if gravity flow is possible:
```python
def validate_chute_flow(self, angle_deg, friction_coeff=0.4):
    min_angle = np.degrees(np.arctan(friction_coeff))
    if angle_deg < min_angle:
        warnings.warn(f"Chute angle {angle_deg}° < minimum {min_angle:.1f}°")
```

### 5. Operating Point Finder

Find actual system flow rate:
```python
def find_system_operating_point(self):
    """Find where blower curve intersects system resistance."""
    pass
```

---

## Key Findings

| Issue | Current State | Required Fix |
|-------|---------------|--------------|
| Air ductwork dP | Not calculated | Add ~230 Pa to system |
| Solids loading | Not modeled | Add momentum exchange |
| Feed chute | 15° assumed to work | Verify angle > 22° or add assist |
| System pressure balance | Not checked | Find actual operating point |
| Cyclone dP | Not in system curve | May dominate total resistance |

The **most critical finding** is that your cyclones alone may require ~26,000 Pa, but your blower only provides ~1,700 Pa. This needs investigation - either the cyclone pressure drops are over-estimated, or the system cannot physically operate as simulated.
