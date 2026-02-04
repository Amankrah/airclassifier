# Air System Regulation Guide

## The Problem

Your air system delivers **1768 m³/h** but classification needs **~14 m³/h** — a **126× mismatch**.

| Parameter | Current | Required | Ratio |
|-----------|---------|----------|-------|
| Flow rate | 1768 m³/h | 14 m³/h | 126× too high |
| Zigzag velocity | 20.5 m/s | 0.16 m/s | 126× too high |
| Zigzag d50 | 685 µm | 35 µm | 20× too high |

---

## Available Control Options

### Option 1: Damper Throttling ❌ NOT RECOMMENDED

Your existing butterfly dampers *could* theoretically throttle the flow, but:

```python
# At 99% closed (position = 0.01), effective area is:
area = full_area × sin(0.01 × π/2) = full_area × 0.016

# This creates:
# - Massive pressure drop (~86 kPa!)
# - Blower operating in surge (unstable)
# - Extremely poor efficiency (<10%)
# - Potential blower damage
```

**Verdict:** Don't try to throttle 126× with dampers. The blower will surge and potentially be damaged.

---

### Option 2: VFD Speed Control ⚠️ IMPRACTICAL

Centrifugal fan affinity laws:
- Flow ∝ RPM
- Pressure ∝ RPM²
- Power ∝ RPM³

To reduce flow from 1768 to 14 m³/h:
```
Required RPM = 2500 × (14/1768) = 20 RPM
```

**Problem:** Centrifugal blowers don't work below ~30% of design speed (~900 RPM minimum). At 20 RPM, there's no meaningful pressure rise.

---

### Option 3: Bypass System ✅ RECOMMENDED

Add a bypass branch that diverts most air away from the classification system.

```
                         ┌──────────────────┐
                         │   BYPASS LINE    │
                         │  (1750 m³/h)     │
                         └────────┬─────────┘
                                  │
Blower ────┬──────────────────────┴──────────────── Exhaust/Return
(1768 m³/h)│
           │
           └─► CLASSIFICATION ─────────────────────► Classifier
               (14-50 m³/h)                          
```

**Implementation:**

```python
# Add to air_system.py or create bypass_system.py

@dataclass
class BypassSystemParams:
    """Parameters for bypass flow control."""
    main_flow_m3_h: float = 1768.0      # Total blower output
    classifier_flow_m3_h: float = 50.0   # Flow to classifier
    bypass_damper_diameter: float = 0.25  # Bypass line diameter [m]
    classifier_damper_diameter: float = 0.08  # Classifier line diameter [m]


class BypassFlowController:
    """
    Bypass system for flow regulation.
    
    Splits blower output between:
    - Bypass line (high flow, low restriction)
    - Classification line (low flow, metered)
    """
    
    def __init__(self, params: BypassSystemParams):
        self.params = params
        self.bypass_damper_position = 1.0   # Fully open
        self.classifier_damper_position = 0.05  # Nearly closed
        
    def calculate_flow_split(self) -> dict:
        """Calculate flow distribution between branches."""
        p = self.params
        
        # Classifier branch: small damper, mostly closed
        # Area ratio determines flow split (simplified)
        A_bypass = np.pi * (p.bypass_damper_diameter/2)**2 * self.bypass_damper_position
        A_class = np.pi * (p.classifier_damper_diameter/2)**2 * self.classifier_damper_position
        
        # Flow splits proportional to area (simplified parallel resistance)
        total_A = A_bypass + A_class
        
        bypass_fraction = A_bypass / total_A if total_A > 0 else 1.0
        class_fraction = A_class / total_A if total_A > 0 else 0.0
        
        return {
            'bypass_flow_m3_h': p.main_flow_m3_h * bypass_fraction,
            'classifier_flow_m3_h': p.main_flow_m3_h * class_fraction,
            'bypass_fraction': bypass_fraction,
            'classifier_fraction': class_fraction,
        }
    
    def set_classifier_flow(self, target_flow_m3_h: float):
        """
        Adjust damper positions to achieve target classifier flow.
        
        Args:
            target_flow_m3_h: Desired flow to classifier [m³/h]
        """
        p = self.params
        
        # Target classifier fraction
        target_fraction = target_flow_m3_h / p.main_flow_m3_h
        
        # Adjust classifier damper to achieve target
        # This is simplified - real system would need feedback control
        
        # Keep bypass fully open
        self.bypass_damper_position = 1.0
        
        # Calculate required classifier damper position
        # A_class / (A_bypass + A_class) = target_fraction
        # A_class = target_fraction * A_bypass / (1 - target_fraction)
        
        A_bypass = np.pi * (p.bypass_damper_diameter/2)**2
        A_class_needed = target_fraction * A_bypass / (1 - target_fraction)
        A_class_full = np.pi * (p.classifier_damper_diameter/2)**2
        
        self.classifier_damper_position = min(1.0, A_class_needed / A_class_full)
        
        return self.calculate_flow_split()
```

---

### Option 4: Separate Classification Blower ✅ MOST ROBUST

Use the main blower only for transport (venturi), and add a small dedicated blower for classification air.

```
MAIN BLOWER (1768 m³/h)
    │
    └──► VENTURI ──► TRANSPORT ──► (particles entrained)
                         │
                         ▼
                    ┌─────────────────┐
                    │   CLASSIFIER    │◄── CLASSIFICATION BLOWER (50 m³/h)
                    │                 │
                    └─────────────────┘
```

**Implementation:**

```python
# Add to your system configuration

@dataclass  
class DualBlowerSystemParams:
    """Parameters for dual-blower system."""
    
    # Main blower (transport)
    transport_flow_m3_h: float = 1768.0
    transport_pressure_Pa: float = 5000.0
    
    # Classification blower (small, dedicated)
    classification_flow_m3_h: float = 50.0  # Adjustable 10-100 m³/h
    classification_pressure_Pa: float = 500.0  # Lower pressure needed


def create_classification_blower(target_d50_um: float = 35.0,
                                  zigzag_area_m2: float = 0.024,
                                  particle_density: float = 1420.0) -> dict:
    """
    Size a dedicated classification blower for target cut size.
    
    Args:
        target_d50_um: Target cut size [µm]
        zigzag_area_m2: Zigzag channel cross-section [m²]
        particle_density: Particle density [kg/m³]
    
    Returns:
        Blower specification dict
    """
    air_viscosity = 1.82e-5
    air_density = 1.204
    g = 9.81
    
    # Calculate required air velocity for target d50
    target_d50 = target_d50_um * 1e-6
    v_air = (target_d50**2) * (particle_density - air_density) * g / (18 * air_viscosity)
    
    # Calculate required flow rate
    Q_m3_s = v_air * zigzag_area_m2
    Q_m3_h = Q_m3_s * 3600
    
    # Size blower with 50% margin
    design_flow = Q_m3_h * 1.5
    
    # Estimate required pressure (zigzag + cyclones + filter)
    # Rough estimate: 500-1000 Pa for classification system
    design_pressure = 800  # Pa
    
    return {
        'target_d50_um': target_d50_um,
        'required_velocity_m_s': v_air,
        'required_flow_m3_h': Q_m3_h,
        'design_flow_m3_h': design_flow,
        'design_pressure_Pa': design_pressure,
        'recommended_blower': f"Small centrifugal, {design_flow:.0f} m³/h @ {design_pressure} Pa",
        'estimated_power_W': design_flow/3600 * design_pressure / 0.6,  # 60% efficiency
    }


# Example usage:
spec = create_classification_blower(target_d50_um=35.0)
print(f"For d50=35µm: Need {spec['design_flow_m3_h']:.1f} m³/h blower")
# Output: For d50=35µm: Need 6.7 m³/h blower
```

---

### Option 5: Recirculation System ⚠️ MODERATE COMPLEXITY

Recirculate most air back to the blower inlet, with only a small fraction going through the classifier.

```
           ┌─────────────────────────────────────┐
           │         RECIRCULATION LINE          │
           │            (1720 m³/h)              │
           └────────────────┬────────────────────┘
                            │
Inlet ──► BLOWER ───────────┴────► CLASSIFIER ──► Exhaust
          (1768)                    (48 m³/h)
```

**Considerations:**
- Air temperature rises with recirculation (compression heat)
- May need cooling
- Blower operates at full speed but against higher resistance

---

## Recommended Solution: Bypass + Control Damper

Here's a complete implementation you can add to your system:

```python
# bypass_controller.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ClassificationFlowController:
    """
    Flow controller for classification system.
    
    Uses a bypass arrangement to regulate flow through the classifier
    while allowing the main blower to operate at its design point.
    """
    
    # System parameters
    blower_flow_m3_h: float = 1768.0
    
    # Bypass branch
    bypass_diameter_m: float = 0.25
    bypass_damper_Cv: float = 200.0  # Flow coefficient when open
    
    # Classification branch  
    classifier_diameter_m: float = 0.10
    classifier_damper_Cv: float = 50.0
    
    # Operating state
    bypass_position: float = 1.0  # 0-1, fully open
    classifier_position: float = 0.1  # 0-1, mostly closed
    
    def get_classifier_flow(self) -> float:
        """Calculate current flow through classifier [m³/h]."""
        # Simplified parallel resistance model
        # Cv_effective determines flow split
        
        Cv_bypass = self.bypass_damper_Cv * self.bypass_position
        Cv_class = self.classifier_damper_Cv * self.classifier_position
        
        if Cv_bypass + Cv_class < 1e-6:
            return 0.0
        
        # Flow splits proportional to Cv
        class_fraction = Cv_class / (Cv_bypass + Cv_class)
        
        return self.blower_flow_m3_h * class_fraction
    
    def set_classifier_flow_target(self, target_m3_h: float) -> dict:
        """
        Adjust dampers to achieve target classifier flow.
        
        Args:
            target_m3_h: Target flow through classifier [m³/h]
            
        Returns:
            Dict with new positions and actual flow
        """
        # Keep bypass fully open for stability
        self.bypass_position = 1.0
        
        # Calculate required Cv ratio
        # class_fraction = Cv_class / (Cv_bypass + Cv_class)
        # target = total_flow * class_fraction
        # target/total = Cv_class / (Cv_bypass + Cv_class)
        
        target_fraction = target_m3_h / self.blower_flow_m3_h
        
        if target_fraction >= 1.0:
            # Close bypass, open classifier
            self.bypass_position = 0.0
            self.classifier_position = 1.0
        else:
            # Solve for classifier position
            # target_fraction = (Cv_class_max * pos) / (Cv_bypass + Cv_class_max * pos)
            # target_fraction * (Cv_bypass + Cv_class_max * pos) = Cv_class_max * pos
            # target_fraction * Cv_bypass = Cv_class_max * pos * (1 - target_fraction)
            # pos = target_fraction * Cv_bypass / (Cv_class_max * (1 - target_fraction))
            
            Cv_bypass = self.bypass_damper_Cv * self.bypass_position
            pos = target_fraction * Cv_bypass / (self.classifier_damper_Cv * (1 - target_fraction))
            self.classifier_position = min(1.0, max(0.01, pos))
        
        actual_flow = self.get_classifier_flow()
        
        return {
            'bypass_position': self.bypass_position,
            'classifier_position': self.classifier_position,
            'target_flow_m3_h': target_m3_h,
            'actual_flow_m3_h': actual_flow,
            'bypass_flow_m3_h': self.blower_flow_m3_h - actual_flow,
        }
    
    def calculate_zigzag_d50(self, zigzag_area_m2: float = 0.024,
                             particle_density: float = 1420.0) -> float:
        """Calculate zigzag cut size at current flow."""
        Q = self.get_classifier_flow() / 3600  # m³/s
        v_air = Q / zigzag_area_m2
        
        air_viscosity = 1.82e-5
        air_density = 1.204
        g = 9.81
        
        d50 = np.sqrt(18 * air_viscosity * v_air / (g * (particle_density - air_density)))
        return d50
    
    def auto_tune_for_d50(self, target_d50_um: float,
                          zigzag_area_m2: float = 0.024,
                          particle_density: float = 1420.0) -> dict:
        """
        Automatically adjust dampers to achieve target cut size.
        
        Args:
            target_d50_um: Target cut size [µm]
            zigzag_area_m2: Zigzag channel area [m²]
            particle_density: Particle density [kg/m³]
            
        Returns:
            Configuration dict
        """
        air_viscosity = 1.82e-5
        air_density = 1.204
        g = 9.81
        
        # Calculate required air velocity
        target_d50 = target_d50_um * 1e-6
        v_air_required = (target_d50**2) * (particle_density - air_density) * g / (18 * air_viscosity)
        
        # Calculate required flow rate
        Q_required_m3_s = v_air_required * zigzag_area_m2
        Q_required_m3_h = Q_required_m3_s * 3600
        
        # Set classifier flow
        result = self.set_classifier_flow_target(Q_required_m3_h)
        
        # Calculate actual d50
        actual_d50 = self.calculate_zigzag_d50(zigzag_area_m2, particle_density)
        
        result['target_d50_um'] = target_d50_um
        result['actual_d50_um'] = actual_d50 * 1e6
        result['required_velocity_m_s'] = v_air_required
        result['actual_velocity_m_s'] = self.get_classifier_flow() / 3600 / zigzag_area_m2
        
        return result


# Example usage
if __name__ == "__main__":
    controller = ClassificationFlowController(blower_flow_m3_h=1768.0)
    
    # Auto-tune for protein/starch separation (d50 = 35 µm)
    config = controller.auto_tune_for_d50(target_d50_um=35.0)
    
    print("=" * 60)
    print("BYPASS CONTROLLER CONFIGURATION")
    print("=" * 60)
    print(f"\nTarget d50:         {config['target_d50_um']:.1f} µm")
    print(f"Actual d50:         {config['actual_d50_um']:.1f} µm")
    print(f"\nBypass damper:      {config['bypass_position']*100:.0f}% open")
    print(f"Classifier damper:  {config['classifier_position']*100:.1f}% open")
    print(f"\nBypass flow:        {config['bypass_flow_m3_h']:.0f} m³/h")
    print(f"Classifier flow:    {config['actual_flow_m3_h']:.1f} m³/h")
    print(f"Air velocity:       {config['actual_velocity_m_s']*100:.2f} cm/s")
    print("=" * 60)
```

---

## Quick Implementation Steps

### Step 1: Add Bypass Branch to Air System

Modify `air_system.py` to include a bypass branch:

```python
# In AirSystemParams, add:
bypass_enabled: bool = True
bypass_diameter: float = 0.25  # [m]
classification_flow_m3_h: float = 50.0  # Target flow to classifier

# In AirSystemAssembly, add bypass damper:
if self.params.bypass_enabled:
    self.bypass_damper = create_standard_damper(
        diameter=self.params.bypass_diameter,
        damper_type="butterfly",
        position=1.0  # Fully open
    )
    self.classification_damper = create_standard_damper(
        diameter=self._duct_diameter * 0.3,  # Smaller for classification
        damper_type="butterfly", 
        position=0.1  # Mostly closed
    )
```

### Step 2: Add Flow Controller

Add the `ClassificationFlowController` class from above to your system.

### Step 3: Modify Simulation Setup

```python
# In your simulation setup:
controller = ClassificationFlowController(blower_flow_m3_h=1768.0)

# Set for protein separation
config = controller.auto_tune_for_d50(target_d50_um=35.0)

# Use the actual classifier flow in simulation
air_flow_m3_h = config['actual_flow_m3_h']  # ~4.5 m³/h instead of 1768!
```

---

## Summary: Recommended Approach

| Option | Complexity | Effectiveness | Recommendation |
|--------|------------|---------------|----------------|
| Damper throttling | Low | ❌ Dangerous | Don't use |
| VFD speed control | Medium | ❌ Impractical | RPM too low |
| **Bypass system** | Medium | ✅ **Best** | **Recommended** |
| Dual blowers | High | ✅ Excellent | Best for production |
| Recirculation | Medium | ⚠️ OK | Heat management needed |

**For your pilot system, implement Option 3 (Bypass):**
1. Add a bypass branch with large-diameter damper
2. Add a small-diameter metering damper on the classification line
3. Use the `ClassificationFlowController` to regulate flow
4. Target ~14-50 m³/h through the classifier for d50 = 35-100 µm

This lets your blower operate efficiently while delivering the correct flow rate to the classifier.
