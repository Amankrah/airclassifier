"""
Bypass Flow Controller for Air Classification System
=====================================================

This module provides flow regulation for air classification systems where
the main blower capacity far exceeds the classifier requirements.

The bypass system splits the blower output into:
1. Bypass line (high flow) - diverts excess air
2. Classification line (low flow) - metered flow to classifier

Usage:
    from bypass_controller import ClassificationFlowController
    
    # Create controller with your blower capacity
    controller = ClassificationFlowController(blower_flow_m3_h=1768.0)
    
    # Auto-tune for target cut size
    config = controller.auto_tune_for_d50(target_d50_um=35.0)
    
    # Use the regulated flow in your simulation
    classifier_flow = config['actual_flow_m3_h']
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class FluidProperties:
    """Standard air properties at 20°C."""
    density: float = 1.204  # kg/m³
    viscosity: float = 1.82e-5  # Pa·s
    temperature_C: float = 20.0


@dataclass
class BypassSystemGeometry:
    """Geometry parameters for bypass system."""
    # Main duct from blower
    main_duct_diameter: float = 0.266  # [m] From your air system
    
    # Bypass branch (large, low restriction)
    bypass_diameter: float = 0.25  # [m]
    bypass_length: float = 1.0  # [m]
    
    # Classification branch (small, metered)
    classifier_diameter: float = 0.10  # [m]
    classifier_length: float = 0.5  # [m]
    
    # Zigzag classifier
    zigzag_width: float = 0.120  # [m]
    zigzag_depth: float = 0.200  # [m]
    
    @property
    def zigzag_area(self) -> float:
        """Zigzag channel cross-sectional area [m²]."""
        return self.zigzag_width * self.zigzag_depth
    
    @property
    def bypass_area(self) -> float:
        """Bypass duct area [m²]."""
        return np.pi * (self.bypass_diameter / 2) ** 2
    
    @property
    def classifier_area(self) -> float:
        """Classification duct area [m²]."""
        return np.pi * (self.classifier_diameter / 2) ** 2


class ClassificationFlowController:
    """
    Flow controller for air classification systems.
    
    Uses a bypass arrangement to regulate flow through the classifier
    while allowing the main blower to operate at its design point.
    
    Physical layout:
    
        Blower ──┬── Bypass Damper ────────── Exhaust/Return
        (1768)   │   (mostly open)
                 │
                 └── Classifier Damper ────── Classifier
                     (mostly closed)          (14-50 m³/h)
    
    The controller adjusts damper positions to achieve:
    - Target d50 cut size in zigzag classifier
    - Or target flow rate through classifier
    
    Attributes:
        blower_flow_m3_h: Total blower output [m³/h]
        geometry: BypassSystemGeometry
        fluid: FluidProperties
        bypass_position: Bypass damper position (0-1)
        classifier_position: Classifier damper position (0-1)
    """
    
    def __init__(self, 
                 blower_flow_m3_h: float = 1768.0,
                 geometry: BypassSystemGeometry = None,
                 fluid: FluidProperties = None):
        """
        Initialize flow controller.
        
        Args:
            blower_flow_m3_h: Total blower output [m³/h]
            geometry: Bypass system geometry (uses defaults if None)
            fluid: Fluid properties (uses air at 20°C if None)
        """
        self.blower_flow_m3_h = blower_flow_m3_h
        self.geometry = geometry or BypassSystemGeometry()
        self.fluid = fluid or FluidProperties()
        
        # Damper positions (0 = closed, 1 = fully open)
        self.bypass_position = 1.0  # Start fully open
        self.classifier_position = 0.1  # Start mostly closed
        
        # Damper flow coefficients (Cv) at fully open
        # Cv = Q * sqrt(SG / dP) where Q in m³/h, dP in bar
        # Larger Cv = less restriction
        self.bypass_Cv_max = 200.0  # Large, low restriction
        self.classifier_Cv_max = 50.0  # Smaller, more restriction
    
    @property
    def bypass_Cv(self) -> float:
        """Effective bypass Cv at current position."""
        # Butterfly valve: Cv ~ sin(θ) where θ = position * 90°
        return self.bypass_Cv_max * np.sin(self.bypass_position * np.pi / 2)
    
    @property
    def classifier_Cv(self) -> float:
        """Effective classifier Cv at current position."""
        return self.classifier_Cv_max * np.sin(self.classifier_position * np.pi / 2)
    
    def get_flow_split(self) -> Dict[str, float]:
        """
        Calculate flow distribution between bypass and classifier.
        
        Uses parallel resistance model where flow splits proportionally
        to the conductance (Cv) of each branch.
        
        Returns:
            Dict with flow rates and fractions for each branch
        """
        Cv_total = self.bypass_Cv + self.classifier_Cv
        
        if Cv_total < 1e-6:
            return {
                'bypass_flow_m3_h': 0.0,
                'classifier_flow_m3_h': 0.0,
                'bypass_fraction': 0.0,
                'classifier_fraction': 0.0,
            }
        
        bypass_fraction = self.bypass_Cv / Cv_total
        classifier_fraction = self.classifier_Cv / Cv_total
        
        return {
            'bypass_flow_m3_h': self.blower_flow_m3_h * bypass_fraction,
            'classifier_flow_m3_h': self.blower_flow_m3_h * classifier_fraction,
            'bypass_fraction': bypass_fraction,
            'classifier_fraction': classifier_fraction,
        }
    
    def get_classifier_flow(self) -> float:
        """Get current flow through classifier [m³/h]."""
        return self.get_flow_split()['classifier_flow_m3_h']
    
    def set_classifier_flow_target(self, target_m3_h: float) -> Dict[str, float]:
        """
        Adjust dampers to achieve target classifier flow.
        
        Keeps bypass fully open and adjusts classifier damper.
        
        Args:
            target_m3_h: Target flow through classifier [m³/h]
            
        Returns:
            Dict with achieved configuration
        """
        # Keep bypass fully open for blower stability
        self.bypass_position = 1.0
        
        # Calculate required classifier fraction
        target_fraction = target_m3_h / self.blower_flow_m3_h
        target_fraction = min(0.99, max(0.001, target_fraction))
        
        # Solve for classifier Cv to achieve target fraction
        # fraction = Cv_class / (Cv_bypass + Cv_class)
        # Cv_class = fraction * Cv_bypass / (1 - fraction)
        
        Cv_class_needed = target_fraction * self.bypass_Cv / (1 - target_fraction)
        
        # Convert to damper position
        # Cv = Cv_max * sin(position * π/2)
        # position = (2/π) * arcsin(Cv / Cv_max)
        
        Cv_ratio = Cv_class_needed / self.classifier_Cv_max
        if Cv_ratio >= 1.0:
            self.classifier_position = 1.0
        elif Cv_ratio <= 0.0:
            self.classifier_position = 0.01  # Never fully close
        else:
            self.classifier_position = (2 / np.pi) * np.arcsin(Cv_ratio)
        
        # Get actual achieved values
        split = self.get_flow_split()
        
        return {
            'target_flow_m3_h': target_m3_h,
            'actual_flow_m3_h': split['classifier_flow_m3_h'],
            'bypass_flow_m3_h': split['bypass_flow_m3_h'],
            'bypass_position': self.bypass_position,
            'classifier_position': self.classifier_position,
            'flow_error_percent': 100 * (split['classifier_flow_m3_h'] - target_m3_h) / target_m3_h if target_m3_h > 0 else 0,
        }
    
    def calculate_zigzag_d50(self, particle_density: float = 1420.0) -> float:
        """
        Calculate zigzag cut size at current flow.
        
        Args:
            particle_density: Particle density [kg/m³]
            
        Returns:
            Cut size d50 [m]
        """
        Q_m3_s = self.get_classifier_flow() / 3600
        v_air = Q_m3_s / self.geometry.zigzag_area
        
        g = 9.81
        mu = self.fluid.viscosity
        rho_p = particle_density
        rho_f = self.fluid.density
        
        # Stokes settling: v_t = d² × (ρ_p - ρ_f) × g / (18 × μ)
        # At d50: v_t = v_air
        # d50 = √(18 × μ × v_air / (g × (ρ_p - ρ_f)))
        
        if v_air <= 0:
            return 0.0
        
        d50 = np.sqrt(18 * mu * v_air / (g * (rho_p - rho_f)))
        return d50
    
    def calculate_required_flow_for_d50(self, target_d50: float,
                                         particle_density: float = 1420.0) -> float:
        """
        Calculate required flow rate for target cut size.
        
        Args:
            target_d50: Target cut size [m]
            particle_density: Particle density [kg/m³]
            
        Returns:
            Required flow rate [m³/h]
        """
        g = 9.81
        mu = self.fluid.viscosity
        rho_p = particle_density
        rho_f = self.fluid.density
        
        # v_air = d50² × (ρ_p - ρ_f) × g / (18 × μ)
        v_air = (target_d50**2) * (rho_p - rho_f) * g / (18 * mu)
        
        Q_m3_s = v_air * self.geometry.zigzag_area
        Q_m3_h = Q_m3_s * 3600
        
        return Q_m3_h
    
    def auto_tune_for_d50(self, target_d50_um: float,
                          particle_density: float = 1420.0) -> Dict[str, float]:
        """
        Automatically adjust dampers to achieve target cut size.
        
        This is the main method for setting up the classification system.
        
        Args:
            target_d50_um: Target cut size [µm]
            particle_density: Particle density [kg/m³]
            
        Returns:
            Configuration dict with all operating parameters
        """
        target_d50_m = target_d50_um * 1e-6
        
        # Calculate required flow
        required_flow = self.calculate_required_flow_for_d50(target_d50_m, particle_density)
        
        # Check if achievable
        if required_flow > self.blower_flow_m3_h:
            print(f"WARNING: Required flow ({required_flow:.1f} m³/h) exceeds blower capacity!")
            required_flow = self.blower_flow_m3_h * 0.95
        
        # Set damper positions
        result = self.set_classifier_flow_target(required_flow)
        
        # Calculate achieved d50
        actual_d50 = self.calculate_zigzag_d50(particle_density)
        
        # Add additional info
        result.update({
            'target_d50_um': target_d50_um,
            'actual_d50_um': actual_d50 * 1e6,
            'd50_error_percent': 100 * (actual_d50 * 1e6 - target_d50_um) / target_d50_um,
            'zigzag_velocity_m_s': result['actual_flow_m3_h'] / 3600 / self.geometry.zigzag_area,
            'particle_density_kg_m3': particle_density,
        })
        
        return result
    
    def print_configuration(self, config: Dict[str, float] = None):
        """Print current or specified configuration."""
        if config is None:
            config = self.get_flow_split()
            config['actual_d50_um'] = self.calculate_zigzag_d50() * 1e6
            config['zigzag_velocity_m_s'] = config['classifier_flow_m3_h'] / 3600 / self.geometry.zigzag_area
            config['bypass_position'] = self.bypass_position
            config['classifier_position'] = self.classifier_position
        
        print("=" * 60)
        print("CLASSIFICATION FLOW CONTROLLER - CONFIGURATION")
        print("=" * 60)
        
        print(f"\nBLOWER:")
        print(f"  Total output:       {self.blower_flow_m3_h:.0f} m³/h")
        
        print(f"\nDAMPER POSITIONS:")
        print(f"  Bypass damper:      {config['bypass_position']*100:.0f}% open")
        print(f"  Classifier damper:  {config['classifier_position']*100:.1f}% open")
        
        print(f"\nFLOW DISTRIBUTION:")
        print(f"  Bypass flow:        {config.get('bypass_flow_m3_h', 0):.0f} m³/h")
        print(f"  Classifier flow:    {config.get('classifier_flow_m3_h', config.get('actual_flow_m3_h', 0)):.1f} m³/h")
        
        print(f"\nZIGZAG CLASSIFIER:")
        print(f"  Channel area:       {self.geometry.zigzag_area*1e4:.0f} cm²")
        print(f"  Air velocity:       {config.get('zigzag_velocity_m_s', 0)*100:.2f} cm/s")
        print(f"  Cut size (d50):     {config.get('actual_d50_um', 0):.1f} µm")
        
        if 'target_d50_um' in config:
            print(f"\nTARGET:")
            print(f"  Target d50:         {config['target_d50_um']:.1f} µm")
            print(f"  Error:              {config.get('d50_error_percent', 0):.1f}%")
        
        print("=" * 60)
    
    def validate_for_classification(self, 
                                    min_particle_um: float = 5.0,
                                    max_particle_um: float = 100.0,
                                    particle_density: float = 1420.0) -> Dict[str, any]:
        """
        Validate current configuration for particle classification.
        
        Args:
            min_particle_um: Smallest particle size [µm]
            max_particle_um: Largest particle size [µm]
            particle_density: Particle density [kg/m³]
            
        Returns:
            Validation result dict
        """
        d50 = self.calculate_zigzag_d50(particle_density)
        d50_um = d50 * 1e6
        
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'd50_um': d50_um,
            'flow_m3_h': self.get_classifier_flow(),
        }
        
        # Check if d50 is within useful range
        if d50_um > max_particle_um:
            result['valid'] = False
            result['errors'].append(
                f"d50 ({d50_um:.1f} µm) > max particle ({max_particle_um:.0f} µm). "
                f"All particles will go to fines. Reduce flow."
            )
        
        if d50_um < min_particle_um:
            result['valid'] = False
            result['errors'].append(
                f"d50 ({d50_um:.1f} µm) < min particle ({min_particle_um:.0f} µm). "
                f"All particles will go to coarse. Increase flow."
            )
        
        # Warnings
        v_air = self.get_classifier_flow() / 3600 / self.geometry.zigzag_area
        if v_air > 2.0:
            result['warnings'].append(
                f"Air velocity ({v_air:.2f} m/s) is high. May cause excessive turbulence."
            )
        
        if self.classifier_position > 0.8:
            result['warnings'].append(
                "Classifier damper nearly fully open. Limited control range remaining."
            )
        
        # Recommended range
        Q_for_35um = self.calculate_required_flow_for_d50(35e-6, particle_density)
        Q_for_100um = self.calculate_required_flow_for_d50(100e-6, particle_density)
        
        result['recommended_flow_range_m3_h'] = (Q_for_35um, Q_for_100um)
        
        return result


def create_standard_bypass_controller(blower_flow_m3_h: float = 1768.0,
                                       zigzag_width_m: float = 0.120,
                                       zigzag_depth_m: float = 0.200) -> ClassificationFlowController:
    """
    Create a standard bypass controller for typical air classification system.
    
    Args:
        blower_flow_m3_h: Main blower output [m³/h]
        zigzag_width_m: Zigzag channel width [m]
        zigzag_depth_m: Zigzag channel depth [m]
        
    Returns:
        Configured ClassificationFlowController
    """
    geometry = BypassSystemGeometry(
        zigzag_width=zigzag_width_m,
        zigzag_depth=zigzag_depth_m,
    )
    
    return ClassificationFlowController(
        blower_flow_m3_h=blower_flow_m3_h,
        geometry=geometry,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BYPASS FLOW CONTROLLER - DEMONSTRATION")
    print("=" * 70)
    
    # Create controller for your system
    controller = ClassificationFlowController(blower_flow_m3_h=1768.0)
    
    # Show current state (before tuning)
    print("\n--- BEFORE TUNING (Default Configuration) ---")
    controller.print_configuration()
    
    # Auto-tune for protein/starch separation
    print("\n--- AUTO-TUNING FOR d50 = 35 µm ---")
    config = controller.auto_tune_for_d50(target_d50_um=35.0, particle_density=1420.0)
    controller.print_configuration(config)
    
    # Validate
    print("\n--- VALIDATION ---")
    validation = controller.validate_for_classification(
        min_particle_um=5.0,
        max_particle_um=100.0,
        particle_density=1420.0
    )
    
    if validation['valid']:
        print("✅ Configuration is VALID for classification")
    else:
        print("❌ Configuration has ERRORS:")
        for error in validation['errors']:
            print(f"   - {error}")
    
    if validation['warnings']:
        print("⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    print(f"\nRecommended flow range: {validation['recommended_flow_range_m3_h'][0]:.1f} - "
          f"{validation['recommended_flow_range_m3_h'][1]:.1f} m³/h")
    
    # Test different target d50 values
    print("\n" + "=" * 70)
    print("D50 vs FLOW RATE TABLE")
    print("=" * 70)
    print(f"{'d50 (µm)':<12} {'Flow (m³/h)':<15} {'Velocity (cm/s)':<18} {'Damper Pos':<12}")
    print("-" * 60)
    
    for d50 in [20, 35, 50, 75, 100, 150]:
        config = controller.auto_tune_for_d50(target_d50_um=d50)
        print(f"{d50:<12} {config['actual_flow_m3_h']:<15.1f} "
              f"{config['zigzag_velocity_m_s']*100:<18.3f} "
              f"{config['classifier_position']*100:<12.2f}%")
    
    print("=" * 70)
