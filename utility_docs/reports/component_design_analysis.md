# Component Design Analysis: Cyclone, Multi-Cyclone, and Zigzag Classifier

## Executive Summary

**The component geometries are correctly designed.** The problem is not in the component code — it's an **operating conditions mismatch** between your air system flow rate and what the classification system needs.

However, I recommend adding **validation and helper methods** to these components to prevent this type of mismatch in future use.

---

## 1. Component-by-Component Assessment

### Zigzag Classifier ✅ Geometry OK, Needs Operating Validation

**Current Design:**
| Parameter | Value | Assessment |
|-----------|-------|------------|
| Channel width | 120 mm | Standard pilot scale |
| Channel depth | 200 mm | Good aspect ratio |
| Stages | 5 | Appropriate for flour |
| Zigzag angle | 120° | Standard |
| Cross-section | 240 cm² | Correctly calculated |

**The Problem:**
The zigzag has no method to validate if the flow rate will achieve the desired cut size. The user can accidentally operate it at 50× the appropriate flow rate with no warning.

**Recommended Addition to `zigzag_classifier.py`:**

```python
def calculate_cut_size_d50(self, volumetric_flow: float, 
                           particle_density: float = 1420.0,
                           air_density: float = 1.204,
                           air_viscosity: float = 1.82e-5) -> float:
    """
    Calculate the cut size (d50) for given flow rate.
    
    Particles smaller than d50 go to fines, larger go to coarse.
    
    Args:
        volumetric_flow: Flow rate [m³/s]
        particle_density: Particle density [kg/m³]
        air_density: Air density [kg/m³]
        air_viscosity: Dynamic viscosity [Pa·s]
    
    Returns:
        d50 cut size [m]
    """
    v_air = self.get_air_velocity(volumetric_flow)
    g = 9.81
    
    # Stokes law rearranged for terminal velocity = air velocity
    # v_t = d² × (ρ_p - ρ_f) × g / (18 × μ)
    # d50 = √(18 × μ × v_air / (g × (ρ_p - ρ_f)))
    
    d50 = np.sqrt(18 * air_viscosity * v_air / (g * (particle_density - air_density)))
    return d50

def calculate_required_flow_for_d50(self, target_d50: float,
                                     particle_density: float = 1420.0,
                                     air_density: float = 1.204,
                                     air_viscosity: float = 1.82e-5) -> float:
    """
    Calculate required flow rate to achieve target cut size.
    
    Args:
        target_d50: Desired cut size [m]
        particle_density: Particle density [kg/m³]
        air_density: Air density [kg/m³]
        air_viscosity: Dynamic viscosity [Pa·s]
    
    Returns:
        Required volumetric flow rate [m³/s]
    """
    g = 9.81
    
    # Rearrange: v_air = d50² × (ρ_p - ρ_f) × g / (18 × μ)
    v_air = (target_d50**2) * (particle_density - air_density) * g / (18 * air_viscosity)
    
    Q = v_air * self.params.channel_cross_section_area
    return Q

def validate_operating_conditions(self, volumetric_flow: float,
                                   min_particle_size: float = 5e-6,
                                   max_particle_size: float = 100e-6,
                                   particle_density: float = 1420.0) -> dict:
    """
    Validate if flow rate is appropriate for particle separation.
    
    Args:
        volumetric_flow: Flow rate [m³/s]
        min_particle_size: Smallest particle to separate [m]
        max_particle_size: Largest particle to separate [m]
        particle_density: Particle density [kg/m³]
    
    Returns:
        Dictionary with validation results and recommendations
    """
    d50 = self.calculate_cut_size_d50(volumetric_flow, particle_density)
    v_air = self.get_air_velocity(volumetric_flow)
    
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'd50_um': d50 * 1e6,
        'air_velocity_m_s': v_air,
    }
    
    # Check if d50 is within particle range
    if d50 > max_particle_size:
        result['valid'] = False
        result['errors'].append(
            f"Cut size ({d50*1e6:.1f} µm) > max particle ({max_particle_size*1e6:.1f} µm). "
            f"ALL particles will go to fines. Reduce flow rate."
        )
    
    if d50 < min_particle_size:
        result['valid'] = False
        result['errors'].append(
            f"Cut size ({d50*1e6:.1f} µm) < min particle ({min_particle_size*1e6:.1f} µm). "
            f"ALL particles will go to coarse. Increase flow rate."
        )
    
    # Velocity warnings
    if v_air > 5.0:
        result['warnings'].append(
            f"Air velocity ({v_air:.1f} m/s) is high. May cause excessive turbulence."
        )
    
    if v_air > 20.0:
        result['warnings'].append(
            f"Air velocity ({v_air:.1f} m/s) is very high. Zigzag will act as transport duct, not separator."
        )
    
    # Calculate recommended flow range
    Q_for_max = self.calculate_required_flow_for_d50(max_particle_size * 0.8, particle_density)
    Q_for_min = self.calculate_required_flow_for_d50(min_particle_size * 1.2, particle_density)
    
    result['recommended_flow_range_m3_s'] = (Q_for_min, Q_for_max)
    result['recommended_flow_range_m3_h'] = (Q_for_min * 3600, Q_for_max * 3600)
    
    return result
```

---

### Single Cyclone ✅ Geometry OK, Needs Cut Size Calculation

**Current Design:**
The cyclone uses standard proportions (Stairmand high-efficiency):
| Ratio | Value | Standard Range |
|-------|-------|----------------|
| Cylinder height/D | 1.5 | 1.5-2.0 ✅ |
| Cone height/D | 2.5 | 2.5-3.0 ✅ |
| Inlet width/D | 0.25 | 0.2-0.25 ✅ |
| Inlet height/D | 0.5 | 0.5 ✅ |
| Vortex finder/D | 0.5 | 0.4-0.5 ✅ |
| Dust outlet/D | 0.375 | 0.25-0.375 ✅ |

**Recommended Addition to `cyclone.py`:**

```python
def calculate_cut_size_d50(self, volumetric_flow: float,
                           particle_density: float = 1420.0,
                           air_density: float = 1.204,
                           air_viscosity: float = 1.82e-5,
                           num_turns: float = 5.0) -> float:
    """
    Calculate cyclone cut size using Lapple equation.
    
    Args:
        volumetric_flow: Flow rate [m³/s]
        particle_density: Particle density [kg/m³]
        air_density: Air density [kg/m³]
        air_viscosity: Dynamic viscosity [Pa·s]
        num_turns: Effective number of turns in cyclone
    
    Returns:
        d50 cut size [m]
    """
    p = self.params
    
    # Inlet velocity
    inlet_area = p.inlet_width * p.inlet_height
    v_inlet = volumetric_flow / inlet_area
    
    # Lapple equation for cyclone d50
    # d50 = √(9 × μ × W / (π × N × v_i × (ρ_p - ρ_g)))
    
    d50 = np.sqrt(
        9 * air_viscosity * p.inlet_width / 
        (np.pi * num_turns * v_inlet * (particle_density - air_density))
    )
    
    return d50

def get_collection_efficiency(self, particle_diameter: float,
                              volumetric_flow: float,
                              particle_density: float = 1420.0) -> float:
    """
    Estimate collection efficiency for a given particle size.
    
    Uses Lapple efficiency curve approximation.
    
    Args:
        particle_diameter: Particle diameter [m]
        volumetric_flow: Flow rate [m³/s]
        particle_density: Particle density [kg/m³]
    
    Returns:
        Collection efficiency (0-1)
    """
    d50 = self.calculate_cut_size_d50(volumetric_flow, particle_density)
    
    # Lapple efficiency curve: η = 1 / (1 + (d50/d)²)
    ratio = d50 / particle_diameter
    efficiency = 1.0 / (1.0 + ratio**2)
    
    return efficiency

def calculate_pressure_drop(self, volumetric_flow: float,
                           air_density: float = 1.204) -> float:
    """
    Estimate cyclone pressure drop.
    
    Args:
        volumetric_flow: Flow rate [m³/s]
        air_density: Air density [kg/m³]
    
    Returns:
        Pressure drop [Pa]
    """
    p = self.params
    
    # Inlet velocity
    inlet_area = p.inlet_width * p.inlet_height
    v_inlet = volumetric_flow / inlet_area
    
    # Empirical correlation: ΔP = K × (ρ × v²/2)
    # K ≈ 6-8 for standard cyclones
    K = 7.0
    
    dP = K * 0.5 * air_density * v_inlet**2
    return dP

def validate_operating_conditions(self, volumetric_flow: float,
                                   particle_density: float = 1420.0) -> dict:
    """
    Validate cyclone operating conditions.
    """
    p = self.params
    inlet_area = p.inlet_width * p.inlet_height
    v_inlet = volumetric_flow / inlet_area
    
    d50 = self.calculate_cut_size_d50(volumetric_flow, particle_density)
    dP = self.calculate_pressure_drop(volumetric_flow)
    
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'd50_um': d50 * 1e6,
        'inlet_velocity_m_s': v_inlet,
        'pressure_drop_Pa': dP,
    }
    
    # Velocity checks
    if v_inlet < 10:
        result['warnings'].append(
            f"Inlet velocity ({v_inlet:.1f} m/s) is low. May have poor separation."
        )
    
    if v_inlet > 30:
        result['warnings'].append(
            f"Inlet velocity ({v_inlet:.1f} m/s) is high. Increased wear and pressure drop."
        )
    
    if v_inlet > 50:
        result['errors'].append(
            f"Inlet velocity ({v_inlet:.1f} m/s) is excessive. Severe erosion likely."
        )
        result['valid'] = False
    
    # Pressure drop warning
    if dP > 2500:
        result['warnings'].append(
            f"Pressure drop ({dP:.0f} Pa) is high. Consider larger cyclone or lower flow."
        )
    
    # Cut size check
    if d50 < 1e-6:
        result['warnings'].append(
            f"Cut size ({d50*1e6:.2f} µm) is submicron. Cyclone may collect excessively."
        )
    
    return result
```

---

### Multi-Cyclone System ✅ Design OK, Needs Staging Validation

**Current Design:**
The series arrangement is correct for protein separation. The design d50 values are reasonable:
| Stage | Diameter | Design d50 |
|-------|----------|------------|
| Primary | 300 mm | 40 µm |
| Secondary | 200 mm | 20 µm |
| Tertiary | 120 mm | 10 µm |

**The Problem:**
The `design_d50` is just a label — it's not used to calculate or validate actual operating d50. At 1,768 m³/h, the actual d50 values are:
- Primary: ~0.8 µm (not 40 µm!)
- Material never reaches secondary/tertiary

**Recommended Addition to `multi_cyclone.py`:**

```python
def calculate_stage_performance(self, volumetric_flow: float,
                                 particle_density: float = 1420.0) -> List[dict]:
    """
    Calculate actual cut sizes and efficiencies for each stage.
    
    Args:
        volumetric_flow: System flow rate [m³/s]
        particle_density: Particle density [kg/m³]
    
    Returns:
        List of performance dicts for each stage
    """
    results = []
    
    for stage in self.params.stages:
        cyclone = self._cyclones[stage.name]
        d50 = cyclone.calculate_cut_size_d50(volumetric_flow, particle_density)
        dP = cyclone.calculate_pressure_drop(volumetric_flow)
        
        # Inlet velocity
        inlet_area = cyclone.params.inlet_width * cyclone.params.inlet_height
        v_inlet = volumetric_flow / inlet_area
        
        results.append({
            'name': stage.name,
            'design_d50_um': stage.design_d50 * 1e6,
            'actual_d50_um': d50 * 1e6,
            'd50_ratio': d50 / stage.design_d50,  # Should be ~1.0
            'inlet_velocity_m_s': v_inlet,
            'pressure_drop_Pa': dP,
        })
    
    return results

def calculate_required_flow_for_design_d50(self, 
                                            particle_density: float = 1420.0) -> float:
    """
    Calculate flow rate needed to achieve design d50 values.
    
    Uses the primary cyclone's design d50 as reference.
    
    Returns:
        Required volumetric flow rate [m³/s]
    """
    primary = self.params.stages[0]
    cyclone = self._cyclones[primary.name]
    p = cyclone.params
    
    target_d50 = primary.design_d50
    air_viscosity = 1.82e-5
    air_density = 1.204
    num_turns = 5.0
    
    # Rearrange Lapple equation to solve for v_inlet
    # d50² = 9 × μ × W / (π × N × v_i × (ρ_p - ρ_g))
    # v_i = 9 × μ × W / (π × N × d50² × (ρ_p - ρ_g))
    
    v_inlet = (9 * air_viscosity * p.inlet_width / 
               (np.pi * num_turns * target_d50**2 * (particle_density - air_density)))
    
    inlet_area = p.inlet_width * p.inlet_height
    Q = v_inlet * inlet_area
    
    return Q

def validate_staging(self, volumetric_flow: float,
                     particle_density: float = 1420.0) -> dict:
    """
    Validate that staging will work at given flow rate.
    """
    stage_perf = self.calculate_stage_performance(volumetric_flow, particle_density)
    
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stages': stage_perf,
    }
    
    # Check if d50 values are in correct order (decreasing)
    d50_values = [s['actual_d50_um'] for s in stage_perf]
    if d50_values != sorted(d50_values, reverse=True):
        result['warnings'].append(
            "Cut sizes are not in expected order. Check cyclone sizing."
        )
    
    # Check if primary d50 is reasonable
    primary_d50 = stage_perf[0]['actual_d50_um']
    if primary_d50 < 5:
        result['errors'].append(
            f"Primary cyclone d50 ({primary_d50:.1f} µm) is too small. "
            f"Will collect ALL material. Reduce flow rate."
        )
        result['valid'] = False
    
    # Check for excessive d50 ratio (design vs actual)
    for stage in stage_perf:
        ratio = stage['d50_ratio']
        if ratio < 0.1:  # Actual is <10% of design
            result['errors'].append(
                f"{stage['name']}: actual d50 ({stage['actual_d50_um']:.1f} µm) is "
                f"{ratio*100:.0f}% of design ({stage['design_d50_um']:.0f} µm). "
                f"Flow rate is much too high."
            )
            result['valid'] = False
    
    # Calculate recommended flow
    Q_design = self.calculate_required_flow_for_design_d50(particle_density)
    result['recommended_flow_m3_s'] = Q_design
    result['recommended_flow_m3_h'] = Q_design * 3600
    result['current_flow_m3_h'] = volumetric_flow * 3600
    result['flow_ratio'] = volumetric_flow / Q_design
    
    return result
```

---

## 2. System-Level Recommendation

Add a validation step to `ClassificationSystemAssembly` that runs at startup:

```python
def validate_system_configuration(self, air_flow_m3_h: float,
                                   particle_density: float = 1420.0,
                                   min_particle_um: float = 5.0,
                                   max_particle_um: float = 100.0) -> dict:
    """
    Validate entire classification system configuration.
    
    Should be called before running simulation.
    """
    Q = air_flow_m3_h / 3600  # Convert to m³/s
    
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'components': {}
    }
    
    # Validate zigzag
    zigzag_val = self.zigzag.validate_operating_conditions(
        Q, min_particle_um*1e-6, max_particle_um*1e-6, particle_density
    )
    result['components']['zigzag'] = zigzag_val
    if not zigzag_val['valid']:
        result['valid'] = False
        result['errors'].extend(zigzag_val['errors'])
    
    # Validate cyclones
    cyclone_val = self.multi_cyclone.validate_staging(Q, particle_density)
    result['components']['cyclones'] = cyclone_val
    if not cyclone_val['valid']:
        result['valid'] = False
        result['errors'].extend(cyclone_val['errors'])
    
    # Summary
    if not result['valid']:
        result['recommendation'] = (
            f"Current flow ({air_flow_m3_h:.0f} m³/h) is incompatible with classification. "
            f"Recommended: {cyclone_val['recommended_flow_m3_h']:.0f} m³/h for design cut sizes."
        )
    
    return result
```

---

## 3. Your Specific Case: What to Do

### Current State
| Parameter | Current | Needed | Ratio |
|-----------|---------|--------|-------|
| Air flow | 1,768 m³/h | ~35 m³/h | 50× too high |
| Zigzag d50 | 695 µm | ~35 µm | 20× too high |
| Cyclone 1 d50 | 0.8 µm | ~40 µm | 50× too low |

### Options (in order of practicality)

**Option 1: Bypass Most Air** ✅ Recommended
```
Blower (1768 m³/h)
    │
    ├─── 1733 m³/h → Bypass/Recirculate
    │
    └─── 35 m³/h → Classification System
```
Add a bypass damper and flow control valve.

**Option 2: Separate Blowers**
- Use main blower for transport (venturi)
- Add small blower (50-100 m³/h) for zigzag classification air

**Option 3: Larger Zigzag** (Not recommended)
Would need 100× channel area — impractical.

**Option 4: Different Technology**
At high flow rates, consider:
- Turbo air classifier (works at high velocity)
- Multi-rotor classifier
- Gravitational-inertial separator

---

## 4. Quick Fix for Testing

To test that your code works correctly, run the simulation with a much lower flow rate:

```python
# In your simulation setup:
air_flow_m3_h = 35.0  # Instead of 1768

# Or calculate from desired d50:
target_d50_um = 35.0  # microns
Q_required = zigzag.calculate_required_flow_for_d50(target_d50_um * 1e-6)
air_flow_m3_h = Q_required * 3600
```

This will let you verify the separation physics works before addressing the flow rate mismatch.

---

## Summary

| Component | Geometry | Code Additions Needed |
|-----------|----------|----------------------|
| Zigzag | ✅ Correct | Add `calculate_cut_size_d50()`, `validate_operating_conditions()` |
| Cyclone | ✅ Correct | Add `calculate_cut_size_d50()`, `calculate_pressure_drop()`, `validate_operating_conditions()` |
| Multi-cyclone | ✅ Correct | Add `calculate_stage_performance()`, `validate_staging()` |
| Classification Assembly | ✅ Correct | Add `validate_system_configuration()` |

**The components don't need geometry changes — they need operating validation methods to prevent misuse.**
