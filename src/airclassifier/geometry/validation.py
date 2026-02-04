"""
Validation Methods for Classification Components
================================================

Add these methods to your existing component classes to validate
operating conditions and prevent flow rate mismatches.

Usage:
    Copy the relevant methods into your component files, or import
    this module and use the standalone functions.
"""

import numpy as np
from typing import Dict, List


# =============================================================================
# ZIGZAG CLASSIFIER VALIDATION METHODS
# =============================================================================
# Add these to zigzag_classifier.py -> ZigzagClassifier class

class ZigzagValidationMixin:
    """Mixin class with validation methods for ZigzagClassifier."""

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

        # Stokes law: d50 = √(18 × μ × v_air / (g × (ρ_p - ρ_f)))
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
            'volumetric_flow_m3_h': volumetric_flow * 3600,
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
            result['errors'].append(
                f"Air velocity ({v_air:.1f} m/s) is very high. "
                f"Zigzag will act as transport duct, not separator."
            )
            result['valid'] = False

        # Calculate recommended flow range
        Q_for_max = self.calculate_required_flow_for_d50(max_particle_size * 0.8, particle_density)
        Q_for_min = self.calculate_required_flow_for_d50(min_particle_size * 1.2, particle_density)

        result['recommended_flow_range_m3_s'] = (Q_for_min, Q_for_max)
        result['recommended_flow_range_m3_h'] = (Q_for_min * 3600, Q_for_max * 3600)

        return result


# =============================================================================
# CYCLONE VALIDATION METHODS
# =============================================================================
# Add these to cyclone.py -> CycloneAssembly class

class CycloneValidationMixin:
    """Mixin class with validation methods for CycloneAssembly."""

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

        # Lapple equation
        # d50 = √(9 × μ × W / (π × N × v_i × (ρ_p - ρ_g)))
        d50 = np.sqrt(
            9 * air_viscosity * p.inlet_width /
            (np.pi * num_turns * v_inlet * (particle_density - air_density))
        )

        return d50

    def calculate_required_flow_for_d50(self, target_d50: float,
                                         particle_density: float = 1420.0,
                                         air_density: float = 1.204,
                                         air_viscosity: float = 1.82e-5,
                                         num_turns: float = 5.0) -> float:
        """
        Calculate required flow rate to achieve target cut size.

        Args:
            target_d50: Desired cut size [m]
            particle_density: Particle density [kg/m³]

        Returns:
            Required volumetric flow rate [m³/s]
        """
        p = self.params

        # Rearrange Lapple: v_i = 9 × μ × W / (π × N × d50² × (ρ_p - ρ_g))
        v_inlet = (9 * air_viscosity * p.inlet_width /
                   (np.pi * num_turns * target_d50**2 * (particle_density - air_density)))

        inlet_area = p.inlet_width * p.inlet_height
        Q = v_inlet * inlet_area

        return Q

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

        # Empirical: ΔP = K × (ρ × v²/2), K ≈ 6-8
        K = 7.0
        dP = K * 0.5 * air_density * v_inlet**2

        return dP

    def validate_operating_conditions(self, volumetric_flow: float,
                                       particle_density: float = 1420.0) -> dict:
        """
        Validate cyclone operating conditions.

        Args:
            volumetric_flow: Flow rate [m³/s]
            particle_density: Particle density [kg/m³]

        Returns:
            Validation results dictionary
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
            'volumetric_flow_m3_h': volumetric_flow * 3600,
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


# =============================================================================
# MULTI-CYCLONE VALIDATION METHODS
# =============================================================================
# Add these to multi_cyclone.py -> MultiCycloneSystem class

class MultiCycloneValidationMixin:
    """Mixin class with validation methods for MultiCycloneSystem."""

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

            # Calculate actual d50 at this flow rate
            d50 = cyclone.calculate_cut_size_d50(volumetric_flow, particle_density)
            dP = cyclone.calculate_pressure_drop(volumetric_flow)

            # Inlet velocity
            inlet_area = cyclone.params.inlet_width * cyclone.params.inlet_height
            v_inlet = volumetric_flow / inlet_area

            results.append({
                'name': stage.name,
                'diameter_mm': stage.diameter * 1000,
                'design_d50_um': stage.design_d50 * 1e6,
                'actual_d50_um': d50 * 1e6,
                'd50_ratio': d50 / stage.design_d50 if stage.design_d50 > 0 else float('inf'),
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

        Q = cyclone.calculate_required_flow_for_d50(primary.design_d50, particle_density)
        return Q

    def validate_staging(self, volumetric_flow: float,
                         particle_density: float = 1420.0) -> dict:
        """
        Validate that staging will work at given flow rate.

        Args:
            volumetric_flow: Flow rate [m³/s]
            particle_density: Particle density [kg/m³]

        Returns:
            Validation results dictionary
        """
        stage_perf = self.calculate_stage_performance(volumetric_flow, particle_density)

        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stages': stage_perf,
            'current_flow_m3_h': volumetric_flow * 3600,
        }

        # Check if d50 values are in correct order (should decrease with each stage)
        d50_values = [s['actual_d50_um'] for s in stage_perf]
        if d50_values != sorted(d50_values, reverse=True):
            result['warnings'].append(
                "Cut sizes are not in expected decreasing order. Check cyclone sizing."
            )

        # Check if primary d50 is reasonable for flour particles
        primary_d50 = stage_perf[0]['actual_d50_um']
        if primary_d50 < 5:
            result['errors'].append(
                f"Primary cyclone d50 ({primary_d50:.1f} µm) is too small. "
                f"Will collect ALL material in first stage. Reduce flow rate."
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
            elif ratio < 0.5:
                result['warnings'].append(
                    f"{stage['name']}: actual d50 ({stage['actual_d50_um']:.1f} µm) is "
                    f"only {ratio*100:.0f}% of design ({stage['design_d50_um']:.0f} µm)."
                )

        # Calculate recommended flow
        Q_design = self.calculate_required_flow_for_design_d50(particle_density)
        result['recommended_flow_m3_s'] = Q_design
        result['recommended_flow_m3_h'] = Q_design * 3600
        result['flow_ratio'] = volumetric_flow / Q_design if Q_design > 0 else float('inf')

        return result


# =============================================================================
# STANDALONE UTILITY FUNCTIONS
# =============================================================================

def calculate_zigzag_d50(channel_area_m2: float, volumetric_flow_m3_s: float,
                         particle_density: float = 1420.0,
                         air_density: float = 1.204,
                         air_viscosity: float = 1.82e-5) -> float:
    """
    Calculate zigzag classifier cut size.

    Args:
        channel_area_m2: Cross-sectional area [m²]
        volumetric_flow_m3_s: Volumetric flow rate [m³/s]
        particle_density: Particle density [kg/m³]
        air_density: Air density [kg/m³]
        air_viscosity: Dynamic viscosity [Pa·s]

    Returns:
        Cut size d50 [m]
    """
    v_air = volumetric_flow_m3_s / channel_area_m2
    g = 9.81
    d50 = np.sqrt(18 * air_viscosity * v_air / (g * (particle_density - air_density)))
    return d50


def calculate_cyclone_d50(inlet_width_m: float, inlet_height_m: float,
                          volumetric_flow_m3_s: float,
                          particle_density: float = 1420.0,
                          air_density: float = 1.204,
                          air_viscosity: float = 1.82e-5,
                          num_turns: float = 5.0) -> float:
    """
    Calculate cyclone cut size using Lapple equation.

    Args:
        inlet_width_m: Inlet width [m]
        inlet_height_m: Inlet height [m]
        volumetric_flow_m3_s: Volumetric flow rate [m³/s]
        particle_density: Particle density [kg/m³]
        air_density: Air density [kg/m³]
        air_viscosity: Dynamic viscosity [Pa·s]
        num_turns: Effective number of turns

    Returns:
        Cut size d50 [m]
    """
    inlet_area = inlet_width_m * inlet_height_m
    v_inlet = volumetric_flow_m3_s / inlet_area

    d50 = np.sqrt(
        9 * air_viscosity * inlet_width_m /
        (np.pi * num_turns * v_inlet * (particle_density - air_density))
    )
    return d50


def print_flow_recommendations(zigzag_area_m2: float,
                               cyclone_inlet_width_m: float,
                               cyclone_inlet_height_m: float,
                               target_zigzag_d50_um: float = 35.0,
                               target_cyclone_d50_um: float = 40.0,
                               particle_density: float = 1420.0) -> None:
    """
    Print recommended flow rates for target cut sizes.
    """
    print("=" * 60)
    print("FLOW RATE RECOMMENDATIONS")
    print("=" * 60)

    air_viscosity = 1.82e-5
    air_density = 1.204
    g = 9.81

    # Zigzag required flow
    target_d50_zz = target_zigzag_d50_um * 1e-6
    v_air_zz = (target_d50_zz**2) * (particle_density - air_density) * g / (18 * air_viscosity)
    Q_zz = v_air_zz * zigzag_area_m2

    print(f"\nZIGZAG CLASSIFIER:")
    print(f"  Target d50:       {target_zigzag_d50_um:.0f} µm")
    print(f"  Channel area:     {zigzag_area_m2 * 1e4:.1f} cm²")
    print(f"  Required flow:    {Q_zz * 3600:.1f} m³/h ({Q_zz * 1000:.2f} L/s)")
    print(f"  Air velocity:     {v_air_zz:.3f} m/s")

    # Cyclone required flow
    target_d50_cy = target_cyclone_d50_um * 1e-6
    inlet_area = cyclone_inlet_width_m * cyclone_inlet_height_m
    num_turns = 5.0

    v_inlet = (9 * air_viscosity * cyclone_inlet_width_m /
               (np.pi * num_turns * target_d50_cy**2 * (particle_density - air_density)))
    Q_cy = v_inlet * inlet_area

    print(f"\nPRIMARY CYCLONE:")
    print(f"  Target d50:       {target_cyclone_d50_um:.0f} µm")
    print(f"  Inlet area:       {inlet_area * 1e4:.1f} cm²")
    print(f"  Required flow:    {Q_cy * 3600:.1f} m³/h ({Q_cy * 1000:.2f} L/s)")
    print(f"  Inlet velocity:   {v_inlet:.1f} m/s")

    # System recommendation
    Q_system = min(Q_zz, Q_cy)
    print(f"\nSYSTEM RECOMMENDATION:")
    print(f"  Operating flow:   {Q_system * 3600:.1f} m³/h")
    print(f"  (Limited by:      {'zigzag' if Q_zz < Q_cy else 'cyclone'})")
    print("=" * 60)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Calculate recommendations for your system

    # Your current zigzag geometry
    zigzag_width = 0.120  # 120 mm
    zigzag_depth = 0.200  # 200 mm
    zigzag_area = zigzag_width * zigzag_depth  # 0.024 m²

    # Your current cyclone geometry (primary)
    cyclone_diameter = 0.300  # 300 mm
    cyclone_inlet_width = cyclone_diameter * 0.25  # 75 mm
    cyclone_inlet_height = cyclone_diameter * 0.5  # 150 mm

    # Print recommendations
    print_flow_recommendations(
        zigzag_area_m2=zigzag_area,
        cyclone_inlet_width_m=cyclone_inlet_width,
        cyclone_inlet_height_m=cyclone_inlet_height,
        target_zigzag_d50_um=35.0,  # For protein/starch separation
        target_cyclone_d50_um=40.0,
        particle_density=1420.0  # Yellow pea
    )

    # Compare with your current flow
    current_flow = 1768 / 3600  # m³/s (your air system output)

    print("\n" + "=" * 60)
    print("CURRENT OPERATING CONDITIONS")
    print("=" * 60)

    d50_zz = calculate_zigzag_d50(zigzag_area, current_flow, 1420.0)
    d50_cy = calculate_cyclone_d50(cyclone_inlet_width, cyclone_inlet_height, current_flow, 1420.0)

    print(f"\nAt current flow ({current_flow * 3600:.0f} m³/h):")
    print(f"  Zigzag d50:       {d50_zz * 1e6:.1f} µm (want ~35 µm)")
    print(f"  Cyclone d50:      {d50_cy * 1e6:.2f} µm (want ~40 µm)")
    print(f"\n  Status: {'OK' if d50_zz < 100e-6 and d50_cy > 5e-6 else 'FLOW TOO HIGH'}")
