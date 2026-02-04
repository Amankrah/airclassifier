"""
Complete System Physics Model
=============================

Integrates all subsystem physics for the complete air classification system:
- Blower performance curve
- Ductwork pressure drops (air supply and feed chute)
- Solids loading effects on venturi
- Classification system resistance
- Operating point determination

This module addresses the gap between subsystem simulations and the complete
integrated system behavior.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings


@dataclass
class DuctSegment:
    """A single duct segment with geometry and loss coefficients."""
    name: str
    segment_type: str  # 'straight', 'elbow', 'transition', 'contraction', 'expansion'
    diameter: float = 0.0  # [m] For round ducts
    length: float = 0.0  # [m] For straight segments
    bend_radius: float = 0.0  # [m] For elbows (R/D ratio)
    bend_angle: float = 90.0  # [degrees]
    K_minor: float = 0.0  # Minor loss coefficient (if specified)
    inlet_diameter: float = 0.0  # For transitions
    outlet_diameter: float = 0.0  # For transitions


@dataclass
class BlowerPerformance:
    """Blower/fan performance parameters."""
    design_flow_m3_h: float = 3000.0
    design_pressure_Pa: float = 5000.0
    design_rpm: float = 3000.0
    operating_rpm: float = 2500.0
    blade_type: str = "backward_curved"
    
    def get_pressure_at_flow(self, Q_m3_h: float) -> float:
        """
        Calculate blower pressure at given flow rate.
        Uses affinity laws and typical fan curve shape.
        """
        # Scale by RPM ratio
        rpm_ratio = self.operating_rpm / self.design_rpm
        
        # Scaled design point
        Q_scaled = self.design_flow_m3_h * rpm_ratio
        P_scaled = self.design_pressure_Pa * rpm_ratio ** 2
        
        # Normalized flow
        if Q_scaled <= 0:
            return P_scaled
        
        Q_norm = Q_m3_h / Q_scaled
        
        # Fan curve: P/P_max = 1 - (Q/Q_max)^2 (simplified parabolic)
        # More accurate: use polynomial fit to actual fan data
        P = P_scaled * (1.0 - 0.8 * Q_norm ** 2)
        
        return max(P, 0)
    
    def get_flow_at_pressure(self, P_Pa: float) -> float:
        """Inverse: find flow for given pressure."""
        rpm_ratio = self.operating_rpm / self.design_rpm
        Q_scaled = self.design_flow_m3_h * rpm_ratio
        P_scaled = self.design_pressure_Pa * rpm_ratio ** 2
        
        if P_Pa >= P_scaled:
            return 0.0
        
        # P = P_scaled * (1 - 0.8 * Q_norm^2)
        # Q_norm = sqrt((1 - P/P_scaled) / 0.8)
        Q_norm = np.sqrt((1.0 - P_Pa / P_scaled) / 0.8)
        
        return Q_norm * Q_scaled


@dataclass 
class CompleteSystemParams:
    """Parameters for complete system physics model."""
    
    # Blower
    blower: BlowerPerformance = field(default_factory=BlowerPerformance)
    
    # Air ductwork (dampers to venturi)
    air_duct_diameter: float = 0.200  # [m]
    air_duct_total_length: float = 3.5  # [m] Total straight length
    air_duct_num_elbows: int = 4
    air_duct_elbow_RD: float = 1.0  # R/D ratio for elbows
    
    # Venturi
    venturi_inlet_diameter: float = 0.080  # [m]
    venturi_throat_diameter: float = 0.040  # [m]
    venturi_outlet_diameter: float = 0.072  # [m]
    venturi_K_loss: float = 0.15  # Net pressure loss coefficient
    
    # Feed chute (deagglomerator to venturi solids inlet)
    feed_chute_diameter: float = 0.035  # [m]
    feed_chute_length: float = 1.5  # [m]
    feed_chute_angle_deg: float = 15.0  # [degrees from horizontal]
    feed_friction_coefficient: float = 0.35  # Powder on smooth steel
    
    # Solids feed
    solids_feed_rate_kg_h: float = 755.0  # [kg/h]
    solids_density: float = 1420.0  # [kg/m³]
    
    # Zigzag classifier
    zigzag_width: float = 0.120  # [m]
    zigzag_depth: float = 0.200  # [m]
    zigzag_num_stages: int = 5
    zigzag_K_per_stage: float = 2.0  # Loss coefficient per stage
    
    # Cyclones (in series)
    cyclone_diameters: List[float] = field(default_factory=lambda: [0.300, 0.200, 0.120])
    cyclone_inlet_width_ratio: float = 0.25  # Inlet width / D
    cyclone_inlet_height_ratio: float = 0.50  # Inlet height / D
    cyclone_K_loss: float = 7.0  # Typical cyclone loss coefficient
    
    # Bag filter
    bag_filter_area: float = 3.68  # [m²]
    bag_filter_base_dP: float = 1200.0  # [Pa] at design face velocity
    bag_filter_design_velocity: float = 0.02  # [m/s]
    
    # Exhaust duct
    exhaust_duct_diameter: float = 0.180  # [m]
    exhaust_duct_length: float = 2.0  # [m]
    exhaust_duct_num_elbows: int = 1
    
    # Air properties
    air_density: float = 1.204  # [kg/m³]
    air_viscosity: float = 1.82e-5  # [Pa·s]
    
    @property
    def zigzag_area(self) -> float:
        return self.zigzag_width * self.zigzag_depth


class CompleteSystemPhysics:
    """
    Complete air classification system physics model.
    
    Calculates:
    - Total system resistance curve
    - Blower operating point
    - Flow distribution
    - Pressure drops in all components
    - Solids loading effects
    - Classification performance (d50)
    """
    
    def __init__(self, params: CompleteSystemParams = None):
        self.params = params or CompleteSystemParams()
        
    def calculate_friction_factor(self, Re: float) -> float:
        """Calculate Darcy friction factor for pipe flow."""
        if Re <= 0:
            return 0.0
        elif Re < 2300:
            return 64 / Re  # Laminar
        else:
            # Blasius equation (turbulent, smooth pipe)
            return 0.316 / (Re ** 0.25)
    
    def calculate_reynolds(self, velocity: float, diameter: float) -> float:
        """Calculate Reynolds number."""
        p = self.params
        return p.air_density * velocity * diameter / p.air_viscosity
    
    def calculate_dynamic_pressure(self, velocity: float) -> float:
        """Calculate dynamic pressure q = 0.5 * rho * v²."""
        return 0.5 * self.params.air_density * velocity ** 2
    
    def calculate_straight_duct_loss(self, Q_m3_s: float, diameter: float, 
                                      length: float) -> Dict:
        """Calculate pressure loss in straight duct."""
        if Q_m3_s <= 0 or diameter <= 0:
            return {'dP': 0, 'velocity': 0, 'Re': 0}
        
        area = np.pi * (diameter / 2) ** 2
        velocity = Q_m3_s / area
        
        Re = self.calculate_reynolds(velocity, diameter)
        f = self.calculate_friction_factor(Re)
        q = self.calculate_dynamic_pressure(velocity)
        
        dP = f * (length / diameter) * q
        
        return {
            'dP': dP,
            'velocity': velocity,
            'Re': Re,
            'friction_factor': f,
            'dynamic_pressure': q,
        }
    
    def calculate_elbow_loss(self, Q_m3_s: float, diameter: float,
                             R_D: float = 1.0) -> Dict:
        """Calculate pressure loss in 90° elbow."""
        if Q_m3_s <= 0 or diameter <= 0:
            return {'dP': 0, 'K': 0}
        
        area = np.pi * (diameter / 2) ** 2
        velocity = Q_m3_s / area
        q = self.calculate_dynamic_pressure(velocity)
        
        # K depends on R/D ratio
        # K = 0.16 for R/D=2, K=0.3 for R/D=1, K=0.9 for R/D=0.5
        if R_D >= 2.0:
            K = 0.16
        elif R_D >= 1.0:
            K = 0.16 + (2.0 - R_D) * (0.3 - 0.16)
        else:
            K = 0.3 + (1.0 - R_D) * (0.9 - 0.3) / 0.5
        
        dP = K * q
        
        return {'dP': dP, 'K': K, 'velocity': velocity}
    
    def calculate_air_ductwork_loss(self, Q_m3_h: float) -> Dict:
        """
        Calculate total pressure drop through air supply ductwork.
        Path: Damper outlet → multiple elbows → Venturi air inlet
        """
        p = self.params
        Q = Q_m3_h / 3600
        
        # Straight duct friction
        duct_result = self.calculate_straight_duct_loss(
            Q, p.air_duct_diameter, p.air_duct_total_length
        )
        
        # Elbow losses
        elbow_result = self.calculate_elbow_loss(
            Q, p.air_duct_diameter, p.air_duct_elbow_RD
        )
        
        total_elbow_dP = elbow_result['dP'] * p.air_duct_num_elbows
        
        # Transition to venturi
        area_duct = np.pi * (p.air_duct_diameter / 2) ** 2
        area_venturi = np.pi * (p.venturi_inlet_diameter / 2) ** 2
        
        v_duct = Q / area_duct if area_duct > 0 else 0
        q_duct = self.calculate_dynamic_pressure(v_duct)
        
        # Contraction/expansion loss
        if area_venturi < area_duct:
            # Contraction
            K_trans = 0.5 * (1 - area_venturi / area_duct) ** 2
        else:
            # Expansion
            K_trans = (1 - area_duct / area_venturi) ** 2
        
        dP_transition = K_trans * q_duct
        
        return {
            'friction_dP': duct_result['dP'],
            'elbow_dP': total_elbow_dP,
            'transition_dP': dP_transition,
            'total_dP': duct_result['dP'] + total_elbow_dP + dP_transition,
            'duct_velocity': duct_result['velocity'],
            'Reynolds': duct_result.get('Re', 0),
        }
    
    def calculate_venturi_loss(self, Q_m3_h: float, include_solids: bool = True) -> Dict:
        """
        Calculate venturi pressure drop including solids loading effect.
        """
        p = self.params
        Q = Q_m3_h / 3600
        
        # Air-only calculation
        area_inlet = np.pi * (p.venturi_inlet_diameter / 2) ** 2
        area_throat = np.pi * (p.venturi_throat_diameter / 2) ** 2
        
        v_inlet = Q / area_inlet if area_inlet > 0 else 0
        v_throat = Q / area_throat if area_throat > 0 else 0
        
        q_inlet = self.calculate_dynamic_pressure(v_inlet)
        q_throat = self.calculate_dynamic_pressure(v_throat)
        
        # Bernoulli pressure drop at throat (for suction calculation)
        dP_bernoulli = q_throat - q_inlet
        
        # Net loss (unrecovered pressure)
        dP_net = p.venturi_K_loss * q_inlet
        
        # Solids loading effect
        dP_solids = 0
        loading_ratio = 0
        
        if include_solids and p.solids_feed_rate_kg_h > 0:
            m_air = Q * p.air_density
            m_solids = p.solids_feed_rate_kg_h / 3600
            
            loading_ratio = m_solids / m_air if m_air > 0 else 0
            
            # Momentum transfer: particles accelerated from ~2 m/s to throat velocity
            v_particle_entry = 2.0  # Approximate chute exit velocity
            momentum_transfer = m_solids * (v_throat - v_particle_entry)
            
            # Additional pressure drop
            dP_solids = momentum_transfer / area_throat / v_throat if v_throat > 0 else 0
        
        return {
            'inlet_velocity': v_inlet,
            'throat_velocity': v_throat,
            'bernoulli_dP': dP_bernoulli,
            'net_dP': dP_net,
            'solids_dP': dP_solids,
            'total_dP': dP_net + dP_solids,
            'loading_ratio': loading_ratio,
        }
    
    def calculate_zigzag_loss(self, Q_m3_h: float) -> Dict:
        """Calculate zigzag classifier pressure drop and cut size."""
        p = self.params
        Q = Q_m3_h / 3600
        
        velocity = Q / p.zigzag_area if p.zigzag_area > 0 else 0
        q = self.calculate_dynamic_pressure(velocity)
        
        # Total loss coefficient
        K_total = p.zigzag_num_stages * p.zigzag_K_per_stage
        dP = K_total * q
        
        # Cut size calculation
        g = 9.81
        if velocity > 0:
            d50 = np.sqrt(
                18 * p.air_viscosity * velocity / 
                (g * (p.solids_density - p.air_density))
            )
        else:
            d50 = 0
        
        return {
            'velocity': velocity,
            'dP': dP,
            'd50_m': d50,
            'd50_um': d50 * 1e6,
            'K_total': K_total,
        }
    
    def calculate_cyclone_loss(self, Q_m3_h: float, cyclone_index: int = 0) -> Dict:
        """Calculate pressure drop for a single cyclone."""
        p = self.params
        Q = Q_m3_h / 3600
        
        if cyclone_index >= len(p.cyclone_diameters):
            return {'dP': 0, 'inlet_velocity': 0}
        
        D = p.cyclone_diameters[cyclone_index]
        inlet_width = D * p.cyclone_inlet_width_ratio
        inlet_height = D * p.cyclone_inlet_height_ratio
        inlet_area = inlet_width * inlet_height
        
        v_inlet = Q / inlet_area if inlet_area > 0 else 0
        q_inlet = self.calculate_dynamic_pressure(v_inlet)
        
        dP = p.cyclone_K_loss * q_inlet
        
        # Cut size (Lapple equation)
        N = 5.0  # Number of turns
        if v_inlet > 0:
            d50 = np.sqrt(
                9 * p.air_viscosity * inlet_width /
                (np.pi * N * v_inlet * (p.solids_density - p.air_density))
            )
        else:
            d50 = 0
        
        return {
            'inlet_velocity': v_inlet,
            'dP': dP,
            'd50_m': d50,
            'd50_um': d50 * 1e6,
            'diameter': D,
        }
    
    def calculate_all_cyclones_loss(self, Q_m3_h: float) -> Dict:
        """Calculate combined pressure drop for all cyclones in series."""
        p = self.params
        
        total_dP = 0
        cyclone_results = []
        
        for i in range(len(p.cyclone_diameters)):
            result = self.calculate_cyclone_loss(Q_m3_h, i)
            cyclone_results.append(result)
            total_dP += result['dP']
        
        return {
            'total_dP': total_dP,
            'cyclones': cyclone_results,
        }
    
    def calculate_bagfilter_loss(self, Q_m3_h: float) -> Dict:
        """Calculate bag filter pressure drop."""
        p = self.params
        Q = Q_m3_h / 3600
        
        face_velocity = Q / p.bag_filter_area if p.bag_filter_area > 0 else 0
        
        # Pressure drop scales with velocity^1.8 approximately
        velocity_ratio = face_velocity / p.bag_filter_design_velocity if p.bag_filter_design_velocity > 0 else 1
        dP = p.bag_filter_base_dP * (velocity_ratio ** 1.8)
        
        return {
            'face_velocity': face_velocity,
            'dP': dP,
        }
    
    def calculate_exhaust_duct_loss(self, Q_m3_h: float) -> Dict:
        """Calculate exhaust ductwork pressure drop."""
        p = self.params
        Q = Q_m3_h / 3600
        
        duct_result = self.calculate_straight_duct_loss(
            Q, p.exhaust_duct_diameter, p.exhaust_duct_length
        )
        
        elbow_result = self.calculate_elbow_loss(Q, p.exhaust_duct_diameter, 1.0)
        total_elbow_dP = elbow_result['dP'] * p.exhaust_duct_num_elbows
        
        return {
            'friction_dP': duct_result['dP'],
            'elbow_dP': total_elbow_dP,
            'total_dP': duct_result['dP'] + total_elbow_dP,
        }
    
    def calculate_system_resistance(self, Q_m3_h: float) -> Dict:
        """
        Calculate total system resistance at given flow rate.
        
        This is the sum of all component pressure drops.
        """
        air_duct = self.calculate_air_ductwork_loss(Q_m3_h)
        venturi = self.calculate_venturi_loss(Q_m3_h)
        zigzag = self.calculate_zigzag_loss(Q_m3_h)
        cyclones = self.calculate_all_cyclones_loss(Q_m3_h)
        bagfilter = self.calculate_bagfilter_loss(Q_m3_h)
        exhaust = self.calculate_exhaust_duct_loss(Q_m3_h)
        
        total_dP = (
            air_duct['total_dP'] +
            venturi['total_dP'] +
            zigzag['dP'] +
            cyclones['total_dP'] +
            bagfilter['dP'] +
            exhaust['total_dP']
        )
        
        return {
            'total_dP': total_dP,
            'air_ductwork': air_duct,
            'venturi': venturi,
            'zigzag': zigzag,
            'cyclones': cyclones,
            'bagfilter': bagfilter,
            'exhaust': exhaust,
        }
    
    def find_operating_point(self, tolerance: float = 1.0) -> Dict:
        """
        Find system operating point where blower curve intersects system resistance.
        
        Uses bisection method to find Q where P_blower(Q) = dP_system(Q).
        """
        p = self.params
        
        Q_min = 1.0  # m³/h
        Q_max = p.blower.design_flow_m3_h * 1.5
        
        def residual(Q):
            P_blower = p.blower.get_pressure_at_flow(Q)
            dP_system = self.calculate_system_resistance(Q)['total_dP']
            return P_blower - dP_system
        
        # Check bounds
        if residual(Q_min) < 0:
            # System resistance too high even at minimum flow
            return {
                'converged': False,
                'message': "System resistance exceeds blower capacity at all flow rates",
                'Q_m3_h': 0,
                'P_Pa': 0,
            }
        
        if residual(Q_max) > 0:
            # Blower has excess capacity
            Q_max *= 2
        
        # Bisection
        for _ in range(50):
            Q_mid = (Q_min + Q_max) / 2
            res = residual(Q_mid)
            
            if abs(res) < tolerance:
                break
            
            if res > 0:
                Q_min = Q_mid
            else:
                Q_max = Q_mid
        
        Q_operating = Q_mid
        P_operating = p.blower.get_pressure_at_flow(Q_operating)
        
        return {
            'converged': True,
            'Q_m3_h': Q_operating,
            'P_Pa': P_operating,
            'system_resistance_Pa': self.calculate_system_resistance(Q_operating)['total_dP'],
        }
    
    def validate_feed_chute_flow(self) -> Dict:
        """
        Validate that particles will flow in the gravity chute.
        
        For gravity flow: tan(angle) > friction_coefficient
        """
        p = self.params
        
        min_angle_rad = np.arctan(p.feed_friction_coefficient)
        min_angle_deg = np.degrees(min_angle_rad)
        
        flows = p.feed_chute_angle_deg > min_angle_deg
        
        # Calculate exit velocity if it flows
        if flows:
            g = 9.81
            theta = np.radians(p.feed_chute_angle_deg)
            a = g * (np.sin(theta) - p.feed_friction_coefficient * np.cos(theta))
            v_exit = np.sqrt(2 * a * p.feed_chute_length)
            t_traverse = np.sqrt(2 * p.feed_chute_length / a)
        else:
            a = 0
            v_exit = 0
            t_traverse = float('inf')
        
        return {
            'flows_by_gravity': flows,
            'chute_angle_deg': p.feed_chute_angle_deg,
            'minimum_angle_deg': min_angle_deg,
            'friction_coefficient': p.feed_friction_coefficient,
            'acceleration_m_s2': a,
            'exit_velocity_m_s': v_exit,
            'traverse_time_s': t_traverse,
        }
    
    def get_complete_system_state(self, Q_m3_h: float = None) -> Dict:
        """
        Get complete system state at given flow rate.
        If Q not specified, finds operating point first.
        """
        if Q_m3_h is None:
            op_point = self.find_operating_point()
            if not op_point['converged']:
                return {'error': op_point['message']}
            Q_m3_h = op_point['Q_m3_h']
        
        p = self.params
        
        # Get all component states
        resistance = self.calculate_system_resistance(Q_m3_h)
        chute = self.validate_feed_chute_flow()
        
        return {
            'flow_m3_h': Q_m3_h,
            'blower_pressure_Pa': p.blower.get_pressure_at_flow(Q_m3_h),
            'system_resistance_Pa': resistance['total_dP'],
            'pressure_balance_Pa': p.blower.get_pressure_at_flow(Q_m3_h) - resistance['total_dP'],
            
            'air_ductwork': resistance['air_ductwork'],
            'venturi': resistance['venturi'],
            'zigzag': resistance['zigzag'],
            'cyclones': resistance['cyclones'],
            'bagfilter': resistance['bagfilter'],
            'exhaust': resistance['exhaust'],
            
            'feed_chute': chute,
            
            'd50_zigzag_um': resistance['zigzag']['d50_um'],
            'd50_primary_cyclone_um': resistance['cyclones']['cyclones'][0]['d50_um'] if resistance['cyclones']['cyclones'] else 0,
        }
    
    def print_system_report(self, Q_m3_h: float = None):
        """Print comprehensive system report."""
        state = self.get_complete_system_state(Q_m3_h)
        
        if 'error' in state:
            print(f"ERROR: {state['error']}")
            return
        
        print("=" * 70)
        print("COMPLETE SYSTEM PHYSICS REPORT")
        print("=" * 70)
        
        print(f"\nOPERATING POINT:")
        print(f"  Flow rate:          {state['flow_m3_h']:.1f} m³/h")
        print(f"  Blower pressure:    {state['blower_pressure_Pa']:.0f} Pa")
        print(f"  System resistance:  {state['system_resistance_Pa']:.0f} Pa")
        print(f"  Pressure balance:   {state['pressure_balance_Pa']:.1f} Pa")
        
        print(f"\nPRESSURE DROP BREAKDOWN:")
        print(f"  {'Component':<25} {'dP (Pa)':<12} {'% of Total':<10}")
        print("-" * 50)
        
        total = state['system_resistance_Pa']
        components = [
            ('Air ductwork', state['air_ductwork']['total_dP']),
            ('Venturi', state['venturi']['total_dP']),
            ('Zigzag classifier', state['zigzag']['dP']),
            ('Cyclones (all)', state['cyclones']['total_dP']),
            ('Bag filter', state['bagfilter']['dP']),
            ('Exhaust duct', state['exhaust']['total_dP']),
        ]
        
        for name, dP in components:
            pct = 100 * dP / total if total > 0 else 0
            print(f"  {name:<25} {dP:<12.0f} {pct:<10.1f}%")
        
        print("-" * 50)
        print(f"  {'TOTAL':<25} {total:<12.0f}")
        
        print(f"\nVELOCITIES:")
        print(f"  Air duct:           {state['air_ductwork']['duct_velocity']:.1f} m/s")
        print(f"  Venturi inlet:      {state['venturi']['inlet_velocity']:.1f} m/s")
        print(f"  Venturi throat:     {state['venturi']['throat_velocity']:.1f} m/s")
        print(f"  Zigzag:             {state['zigzag']['velocity']:.2f} m/s")
        print(f"  Bag filter face:    {state['bagfilter']['face_velocity']*100:.1f} cm/s")
        
        print(f"\nCLASSIFICATION:")
        print(f"  Zigzag d50:         {state['d50_zigzag_um']:.1f} µm")
        print(f"  Primary cyclone d50: {state['d50_primary_cyclone_um']:.1f} µm")
        print(f"  Solids loading:     {state['venturi']['loading_ratio']*100:.1f}%")
        
        print(f"\nFEED CHUTE:")
        chute = state['feed_chute']
        if chute['flows_by_gravity']:
            print(f"  Status:             ✓ FLOWS (angle {chute['chute_angle_deg']:.0f}° > min {chute['minimum_angle_deg']:.1f}°)")
            print(f"  Exit velocity:      {chute['exit_velocity_m_s']:.2f} m/s")
        else:
            print(f"  Status:             ✗ BLOCKED (angle {chute['chute_angle_deg']:.0f}° < min {chute['minimum_angle_deg']:.1f}°)")
            print(f"  Recommendation:     Increase angle or add air assist")
        
        print("=" * 70)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create system with your parameters
    blower = BlowerPerformance(
        design_flow_m3_h=3000,
        design_pressure_Pa=5000,
        design_rpm=3000,
        operating_rpm=2500,
    )
    
    params = CompleteSystemParams(
        blower=blower,
        air_duct_diameter=0.200,
        air_duct_total_length=3.5,
        air_duct_num_elbows=4,
        venturi_inlet_diameter=0.080,
        venturi_throat_diameter=0.040,
        solids_feed_rate_kg_h=755,
        zigzag_width=0.120,
        zigzag_depth=0.200,
        feed_chute_angle_deg=15,
        feed_friction_coefficient=0.35,
    )
    
    system = CompleteSystemPhysics(params)
    
    # Find operating point and print report
    print("\n" + "=" * 70)
    print("FINDING SYSTEM OPERATING POINT")
    print("=" * 70)
    
    op_point = system.find_operating_point()
    print(f"\nOperating point: {op_point['Q_m3_h']:.1f} m³/h at {op_point['P_Pa']:.0f} Pa")
    
    # Print full report
    system.print_system_report()
    
    # Also check at the "assumed" flow rate
    print("\n" + "=" * 70)
    print("COMPARISON: AT ASSUMED 1768 m³/h (from isolated air system simulation)")
    print("=" * 70)
    
    state_assumed = system.get_complete_system_state(1768)
    print(f"\nAt Q = 1768 m³/h:")
    print(f"  Blower can provide:  {state_assumed['blower_pressure_Pa']:.0f} Pa")
    print(f"  System requires:     {state_assumed['system_resistance_Pa']:.0f} Pa")
    print(f"  Deficit:             {state_assumed['pressure_balance_Pa']:.0f} Pa")
    print(f"\n  → BLOWER CANNOT DELIVER THIS FLOW THROUGH THE COMPLETE SYSTEM!")
