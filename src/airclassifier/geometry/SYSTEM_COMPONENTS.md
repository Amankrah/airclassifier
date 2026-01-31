# Air Classification System Components

## Complete System Architecture for Legume Protein Separation

This document outlines all geometry components required to build a complete air classification system for protein separation from milled legumes (yellow peas, faba beans, oat).

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE AIR CLASSIFICATION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐ │
│  │ FEED SYSTEM  │────►│  AIR SUPPLY  │────►│     CLASSIFICATION ZONE      │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────────┘ │
│                                                        │                     │
│                                            ┌───────────┴───────────┐        │
│                                            ▼                       ▼        │
│                                    ┌──────────────┐       ┌──────────────┐  │
│                                    │   PROTEIN    │       │   STARCH     │  │
│                                    │  COLLECTION  │       │  COLLECTION  │  │
│                                    └──────────────┘       └──────────────┘  │
│                                            │                       │        │
│                                            └───────────┬───────────┘        │
│                                                        ▼                    │
│                                            ┌──────────────────────┐         │
│                                            │    EXHAUST SYSTEM    │         │
│                                            └──────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Status

| Status | Meaning |
|--------|---------|
| ✅ | Implemented |
| 🔨 | In Progress |
| ❌ | Not Started |

---

## 1. Feed System Components

### 1.1 Feed Hopper
**Status:** ✅ Implemented  
**Module:** `geometry/components/feed_hopper.py`

```python
@dataclass
class FeedHopperParams:
    """Feed hopper/silo for flour storage."""
    top_diameter: float          # [m] Top opening diameter
    bottom_diameter: float       # [m] Bottom discharge diameter
    cylindrical_height: float    # [m] Height of cylindrical section
    conical_height: float        # [m] Height of conical discharge section
    cone_half_angle: float       # [rad] Angle of hopper cone (typically 30-45°)
    capacity_kg: float           # [kg] Design capacity
    material: str                # Material type (304SS, 316SS)
```

**Design Considerations:**
- Mass flow hopper design (not funnel flow) for consistent discharge
- Cone angle > material's angle of repose + 10-15°
- Vibrator mounting pads for flow aids
- Level sensors (high/low)
- Food-grade interior finish (Ra < 0.8 μm)

---

### 1.2 Rotary Airlock Valve
**Status:** ✅ Implemented  
**Module:** `geometry/components/rotary_airlock.py`

```python
@dataclass
class RotaryAirlockParams:
    """Rotary airlock valve for pressure sealing."""
    rotor_diameter: float        # [m] Rotor diameter
    rotor_length: float          # [m] Rotor length (width)
    num_vanes: int               # Number of vanes (typically 6-10)
    vane_tip_clearance: float    # [m] Gap between vane and housing
    inlet_diameter: float        # [m] Inlet flange diameter
    outlet_diameter: float       # [m] Outlet flange diameter
    rpm: float                   # [rpm] Rotation speed (15-30 typical)
    capacity_m3_hr: float        # [m³/h] Volumetric capacity
```

**Design Considerations:**
- Drop-through or blow-through configuration
- Adjustable tip clearance for air leakage control
- ATEX/explosion-proof motor
- Easy-clean design with removable end plates
- Flexible inlet connection for vibration isolation

---

### 1.3 Screw Feeder / Dosing Unit
**Status:** ✅ Implemented  
**Module:** `geometry/components/screw_feeder.py`

```python
@dataclass
class ScrewFeederParams:
    """Screw feeder for controlled dosing."""
    screw_diameter: float        # [m] Screw/auger diameter
    screw_pitch: float           # [m] Pitch of screw flights
    trough_length: float         # [m] Length of trough
    trough_loading: float        # [%] Fill level (15-45% typical)
    inlet_opening: tuple         # [m] (length, width) of inlet
    outlet_diameter: float       # [m] Outlet diameter
    variable_pitch: bool         # Use variable pitch for uniform withdrawal
    feed_rate_range: tuple       # [kg/h] (min, max) feed rate
```

**Design Considerations:**
- Variable pitch increases near outlet for consistent flow
- Jacketed trough option for temperature control
- Agitator above screw for bridging prevention
- VFD-controlled motor for feed rate adjustment

---

### 1.4 Venturi Eductor / Feeder
**Status:** ❌ Not Started  
**Module:** `geometry/components/venturi_eductor.py`

```python
@dataclass
class VenturiEducatorParams:
    """Venturi eductor for particle entrainment."""
    throat_diameter: float       # [m] Venturi throat diameter
    inlet_diameter: float        # [m] Air inlet diameter
    outlet_diameter: float       # [m] Mixed flow outlet diameter
    convergent_angle: float      # [rad] Inlet convergent angle
    divergent_angle: float       # [rad] Outlet divergent angle
    solids_inlet_diameter: float # [m] Particle feed inlet
    solids_inlet_angle: float    # [rad] Angle of solids entry
    suction_capacity: float      # [Pa] Suction pressure developed
```

**Design Considerations:**
- Throat velocity 20-35 m/s for good entrainment
- Solids entry at or after throat
- Wear-resistant throat lining
- Smooth internal transitions

---

### 1.5 De-agglomerator / Lump Breaker
**Status:** ✅ Implemented  
**Module:** `geometry/components/deagglomerator.py`

```python
@dataclass
class DeagglomeratorParams:
    """De-agglomerator for breaking flour clumps."""
    rotor_diameter: float        # [m] Rotor diameter
    rotor_length: float          # [m] Rotor length
    num_pins: int                # Number of pins/paddles
    pin_diameter: float          # [m] Pin diameter
    screen_aperture: float       # [m] Screen hole size
    rpm: float                   # [rpm] Rotation speed
    inlet_diameter: float        # [m] Inlet size
```

---

## 2. Air Supply System

### 2.1 Centrifugal Blower / Fan
**Status:** ✅ Implemented  
**Module:** `geometry/components/centrifugal_blower.py`

```python
@dataclass
class CentrifugalBlowerParams:
    """Centrifugal blower for air supply."""
    impeller_diameter: float     # [m] Impeller outer diameter
    impeller_width: float        # [m] Impeller width at inlet
    inlet_diameter: float        # [m] Inlet eye diameter
    outlet_width: float          # [m] Outlet width
    outlet_height: float         # [m] Outlet height
    num_blades: int              # Number of impeller blades
    blade_type: str              # "backward_curved", "radial", "forward_curved"
    scroll_radius: float         # [m] Scroll/volute radius
    rpm: float                   # [rpm] Operating speed
    flow_rate: float             # [m³/h] Design flow rate
    pressure_rise: float         # [Pa] Total pressure rise
```

**Design Considerations:**
- Backward-curved blades for efficiency (75-85%)
- Pressure: 2,000-10,000 Pa typical for air classification
- Flow rate: sized for 5-15 m/s superficial velocity
- VFD control for flow adjustment
- Inlet damper for turndown

---

### 2.2 Air Heater (Optional)
**Status:** ❌ Not Started  
**Module:** `geometry/components/air_heater.py`

```python
@dataclass
class AirHeaterParams:
    """Air heater for humidity/temperature control."""
    heater_type: str             # "electric", "steam", "gas"
    duct_diameter: float         # [m] Duct diameter
    heating_length: float        # [m] Length of heating section
    num_elements: int            # Number of heating elements
    power_kw: float              # [kW] Heating power
    max_temperature: float       # [°C] Maximum outlet temperature
    temperature_rise: float      # [°C] Design temperature rise
```

**Design Considerations:**
- Target: 40-60°C inlet air for humidity control
- Food-grade materials (no contamination)
- Explosion-proof for ATEX zones

---

### 2.3 Inlet Air Filter
**Status:** ✅ Implemented  
**Module:** `geometry/components/air_filter.py`

```python
@dataclass
class InletAirFilterParams:
    """Inlet air filter for clean air supply."""
    filter_type: str             # "panel", "bag", "cartridge", "HEPA"
    housing_diameter: float      # [m] Filter housing diameter
    housing_length: float        # [m] Filter housing length
    filter_area: float           # [m²] Total filter area
    efficiency_class: str        # "G4", "M5", "F7", "H13", etc.
    max_pressure_drop: float     # [Pa] Clean filter pressure drop
    change_out_dp: float         # [Pa] Pressure drop for filter change
```

---

### 2.4 Flow Control Damper
**Status:** ✅ Implemented  
**Module:** `geometry/components/damper.py`

```python
@dataclass
class DamperParams:
    """Flow control damper/valve."""
    damper_type: str             # "butterfly", "louver", "iris"
    diameter: float              # [m] Duct diameter
    num_blades: int              # Number of blades (for louver)
    actuator_type: str           # "manual", "pneumatic", "electric"
    cv_max: float                # Flow coefficient at full open
```

---

## 3. Classification Zone (PRIMARY SEPARATION)

### 3.1 Zigzag Air Classifier ⭐ CRITICAL
**Status:** ✅ Implemented  
**Module:** `geometry/components/zigzag_classifier.py`

```python
@dataclass
class ZigzagClassifierParams:
    """Zigzag air classifier for particle separation."""
    channel_width: float         # [m] Width of zigzag channel
    channel_depth: float         # [m] Depth of channel (into page)
    num_stages: int              # Number of zigzag stages (3-7 typical)
    stage_height: float          # [m] Height per stage
    zigzag_angle: float          # [rad] Angle of zigzag (typically 120°)
    feed_position: int           # Stage number for feed entry
    air_inlet_width: float       # [m] Bottom air inlet width
    fines_outlet_width: float    # [m] Top fines outlet width
    coarse_outlet_width: float   # [m] Bottom coarse outlet width
    wall_thickness: float        # [m] Wall thickness
```

**Key Design Parameters:**
- Channel width: 100-300 mm typical
- 3-5 stages for legume protein separation
- Feed enters at middle stage
- Air velocity: 1-5 m/s depending on cut size

**Geometry:**
```
        FINES (Protein-rich)
              ▲
              │
    ┌─────────┴─────────┐
    │    Stage 5        │
    │  ┌─────────┐      │
    │  │         │      │
    │  │    ╱────┤ Stage 4
    │  │   ╱     │      │
    │  ├──╱      │      │
    │  │ ╱   FEED►──────┤ Stage 3 (Feed Entry)
    │  │╱        │      │
    │  ├─────────┤      │
    │  │    ╲    │ Stage 2
    │  │     ╲   │      │
    │  │      ╲──┤      │
    │  │         │ Stage 1
    └──┴─────────┴──────┘
              │
              ▼
        AIR INLET
              │
              ▼
        COARSE (Starch-rich)
```

---

### 3.2 Turbo / Rotor Classifier
**Status:** ❌ Not Started  
**Module:** `geometry/components/turbo_classifier.py`

```python
@dataclass
class TurboClassifierParams:
    """Turbo/rotor classifier for fine separation."""
    rotor_diameter: float        # [m] Classifier rotor diameter
    rotor_height: float          # [m] Rotor height/width
    num_blades: int              # Number of rotor blades (24-48)
    blade_angle: float           # [rad] Blade angle
    housing_diameter: float      # [m] Housing inner diameter
    feed_inlet_diameter: float   # [m] Feed inlet size
    fines_outlet_diameter: float # [m] Fines (overflow) outlet
    coarse_outlet_diameter: float# [m] Coarse (reject) outlet
    rpm: float                   # [rpm] Rotor speed (1000-6000)
    guide_vane_count: int        # Number of stationary guide vanes
```

**Design Considerations:**
- Higher speed = finer cut point
- Cut size range: 2-100 μm adjustable
- VFD for speed control
- Wear-resistant rotor blades

---

### 3.3 Gravitational Counter-Flow Classifier
**Status:** ❌ Not Started  
**Module:** `geometry/components/counterflow_classifier.py`

```python
@dataclass
class CounterflowClassifierParams:
    """Vertical counter-flow gravitational classifier."""
    column_diameter: float       # [m] Column inner diameter
    column_height: float         # [m] Total column height
    feed_height: float           # [m] Height of feed entry
    num_distribution_plates: int # Number of redistribution plates
    plate_spacing: float         # [m] Spacing between plates
    plate_open_area: float       # [%] Open area of plates
    air_inlet_diameter: float    # [m] Bottom air inlet
    fines_outlet_diameter: float # [m] Top outlet for fines
    coarse_outlet_diameter: float# [m] Bottom outlet for coarse
```

---

### 3.4 Elutriator Column
**Status:** ❌ Not Started  
**Module:** `geometry/components/elutriator.py`

```python
@dataclass
class ElutriatorParams:
    """Elutriator for particle classification by terminal velocity."""
    tube_diameter: float         # [m] Elutriator tube diameter
    tube_height: float           # [m] Tube height
    expansion_diameter: float    # [m] Diameter of expansion zone
    expansion_height: float      # [m] Height of expansion zone
    feed_tube_diameter: float    # [m] Central feed tube diameter
    air_distributor_type: str    # "perforated_plate", "sintered", "sparger"
    num_stages: int              # Number of elutriator stages
```

---

## 4. Collection System

### 4.1 Cyclone Separator (Multiple Required)
**Status:** ✅ Implemented  
**Module:** `geometry/assembly.py`, `geometry/components/`

**Current Implementation:**
- Single cyclone with body, cone, vortex finder, inlet, dust outlet

**Required Extensions:**
```python
@dataclass
class CycloneSystemParams:
    """Multi-cyclone collection system."""
    primary_cyclone: CycloneGeometryParams    # Coarse fraction
    secondary_cyclone: CycloneGeometryParams  # Medium fraction
    tertiary_cyclone: CycloneGeometryParams   # Fine fraction
    connecting_ducts: List[DuctParams]        # Interconnecting ducts
    manifold_type: str                        # "parallel", "series"
```

**Design for Protein Separation:**
| Cyclone | Collects | Typical d50 | Product |
|---------|----------|-------------|---------|
| Primary | Coarse | 30-50 μm | Starch fraction (60-70% starch) |
| Secondary | Medium | 15-25 μm | Mixed fraction |
| Tertiary | Fine | 5-15 μm | Protein fraction (50-65% protein) |

---

### 4.2 Bag Filter / Baghouse
**Status:** ✅ Implemented  
**Module:** `geometry/components/bag_filter.py`

```python
@dataclass
class BagFilterParams:
    """Bag filter/baghouse for fine particle collection."""
    housing_width: float         # [m] Housing width
    housing_depth: float         # [m] Housing depth
    housing_height: float        # [m] Housing height
    num_bags: int                # Number of filter bags
    bag_diameter: float          # [m] Individual bag diameter
    bag_length: float            # [m] Bag length
    bag_material: str            # "polyester", "PTFE", "aramid"
    cleaning_type: str           # "pulse_jet", "shaker", "reverse_air"
    pulse_pressure: float        # [bar] Cleaning pulse pressure
    hopper_angle: float          # [rad] Collection hopper angle
    air_to_cloth_ratio: float    # [m³/min/m²] Filtration velocity
```

**Design Considerations:**
- Air-to-cloth ratio: 1.5-3.0 m³/min/m² for food dust
- Pulse-jet cleaning most common
- Food-grade bag materials
- Collection efficiency >99.9% for particles >1 μm

---

### 4.3 Collection Bins / Hoppers
**Status:** ❌ Not Started  
**Module:** `geometry/components/collection_bin.py`

```python
@dataclass
class CollectionBinParams:
    """Product collection bin/hopper."""
    bin_type: str                # "cylindrical", "rectangular"
    volume: float                # [m³] Bin volume
    diameter: float              # [m] Diameter (if cylindrical)
    width: float                 # [m] Width (if rectangular)
    length: float                # [m] Length (if rectangular)
    straight_height: float       # [m] Straight side height
    cone_height: float           # [m] Discharge cone height
    cone_angle: float            # [rad] Cone half-angle
    outlet_diameter: float       # [m] Discharge outlet diameter
    load_cells: bool             # Has load cells for weighing
```

---

## 5. Ductwork and Transitions

### 5.1 Round Duct
**Status:** ✅ Implemented  
**Module:** `geometry/components/ductwork.py`

```python
@dataclass
class RoundDuctParams:
    """Circular duct section."""
    diameter: float              # [m] Inner diameter
    length: float                # [m] Duct length
    wall_thickness: float        # [m] Wall thickness
    material: str                # "galvanized", "304SS", "316SS"
    flanged: bool                # Has flanged connections
    insulated: bool              # Has insulation
```

---

### 5.2 Rectangular Duct
**Status:** ✅ Implemented  
**Module:** `geometry/components/ductwork.py`

```python
@dataclass
class RectangularDuctParams:
    """Rectangular duct section."""
    width: float                 # [m] Internal width
    height: float                # [m] Internal height
    length: float                # [m] Duct length
    corner_radius: float         # [m] Internal corner radius
    wall_thickness: float        # [m] Wall thickness
```

---

### 5.3 Duct Transitions
**Status:** ✅ Implemented  
**Module:** `geometry/components/transitions.py`

```python
@dataclass
class TransitionParams:
    """Duct transitions and reducers."""
    transition_type: str         # "round_to_round", "round_to_rect", "rect_to_rect"
    inlet_dimensions: tuple      # Inlet size(s)
    outlet_dimensions: tuple     # Outlet size(s)
    length: float                # [m] Transition length
    concentric: bool             # Concentric or eccentric
    max_angle: float             # [rad] Maximum expansion/contraction angle
```

**Design Guidelines:**
- Expansion angle: ≤15° to avoid separation
- Contraction angle: ≤30° acceptable
- Smooth internal surfaces
- Radiused corners for rectangular transitions

---

### 5.4 Elbows and Bends
**Status:** ✅ Implemented  
**Module:** `geometry/components/elbows.py`

```python
@dataclass
class ElbowParams:
    """Duct elbow/bend."""
    elbow_type: str              # "round", "rectangular", "mitered"
    diameter: float              # [m] Duct diameter (round)
    width: float                 # [m] Width (rectangular)
    height: float                # [m] Height (rectangular)
    bend_radius: float           # [m] Centerline bend radius
    bend_angle: float            # [rad] Bend angle (typically 90°)
    num_gores: int               # Number of gores (for mitered)
    turning_vanes: bool          # Has internal turning vanes
```

**Design Guidelines:**
- R/D ≥ 1.5 for low pressure drop
- R/D ≥ 2.0 for particle-laden flows (reduce erosion)
- Turning vanes for tight radius bends

---

### 5.5 Y-Branch / Wye
**Status:** ✅ Implemented  
**Module:** `geometry/components/diverter.py` (as DiverterValve)

```python
@dataclass
class WyeBranchParams:
    """Y-branch for flow splitting."""
    inlet_diameter: float        # [m] Main inlet diameter
    branch1_diameter: float      # [m] Branch 1 outlet diameter
    branch2_diameter: float      # [m] Branch 2 outlet diameter
    branch_angle: float          # [rad] Angle between branches
    symmetric: bool              # Symmetric Y or asymmetric
```

---

### 5.6 Diverter Valve
**Status:** ✅ Implemented  
**Module:** `geometry/components/diverter.py`

```python
@dataclass
class DiverterValveParams:
    """Two-way diverter valve."""
    inlet_diameter: float        # [m] Inlet diameter
    outlet1_diameter: float      # [m] Outlet 1 diameter
    outlet2_diameter: float      # [m] Outlet 2 diameter
    outlet_angle: float          # [rad] Angle between outlets
    blade_type: str              # "flap", "rotating", "plug"
    actuator_type: str           # "pneumatic", "electric", "manual"
    seal_type: str               # "flexible", "inflatable", "metal"
```

---

## 6. Exhaust System

### 6.1 Exhaust Fan
**Status:** ❌ Not Started  
**Module:** `geometry/components/exhaust_fan.py`

```python
@dataclass
class ExhaustFanParams:
    """Exhaust/induced draft fan."""
    # Similar to CentrifugalBlowerParams
    fan_type: str                # "centrifugal", "axial"
    impeller_diameter: float     # [m] Impeller diameter
    flow_rate: float             # [m³/h] Design flow
    static_pressure: float       # [Pa] Static pressure
    efficiency: float            # [%] Fan efficiency
```

---

### 6.2 Silencer / Muffler
**Status:** ✅ Implemented  
**Module:** `geometry/components/silencer.py`

```python
@dataclass
class SilencerParams:
    """Acoustic silencer/muffler."""
    silencer_type: str           # "absorptive", "reactive", "combination"
    diameter: float              # [m] Duct diameter
    length: float                # [m] Silencer length
    num_splitters: int           # Number of splitter baffles
    splitter_thickness: float    # [m] Splitter thickness
    absorption_material: str     # "mineral_wool", "foam"
    insertion_loss: float        # [dB] Design insertion loss
```

---

### 6.3 Exhaust Stack
**Status:** ✅ Implemented  
**Module:** `geometry/components/exhaust_stack.py`

```python
@dataclass
class ExhaustStackParams:
    """Exhaust stack/chimney."""
    diameter: float              # [m] Stack diameter
    height: float                # [m] Stack height
    wall_thickness: float        # [m] Wall thickness
    rain_cap: bool               # Has rain cap
    cap_type: str                # "conical", "chinese_hat", "H_cap"
    discharge_velocity: float    # [m/s] Exit velocity
```

---

## 7. Safety and Ancillary Systems

### 7.1 Explosion Vent
**Status:** ✅ Implemented  
**Module:** `geometry/components/safety/explosion_vent.py`

```python
@dataclass
class ExplosionVentParams:
    """Explosion vent panel."""
    vent_area: float             # [m²] Vent area
    vent_type: str               # "rupture_panel", "hinged_door", "recoil"
    static_burst_pressure: float # [bar] Burst pressure (Pstat)
    duct_diameter: float         # [m] Vent duct diameter
    duct_length: float           # [m] Vent duct length (to safe area)
    flame_arrestor: bool         # Has flame arrestor
```

**Design Notes:**
- Legume dust: Kst = 100-200 bar·m/s
- Pmax = 7-9 bar typical
- Vent sizing per NFPA 68 or EN 14491

---

### 7.2 Rotary Airlock (Explosion Isolation)
**Status:** ❌ Not Started  
**Module:** `geometry/components/safety/explosion_isolation.py`

```python
@dataclass
class ExplosionIsolationParams:
    """Explosion isolation rotary valve."""
    # Extended rotary airlock with certified isolation
    valve_type: str              # "rotary", "gate", "pinch"
    certified_Pstat: float       # [bar] Certified isolation pressure
    closing_time: float          # [ms] Valve closing time
    detection_trigger: str       # "pressure", "optical", "thermal"
```

---

### 7.3 Grounding / Bonding Points
**Status:** ✅ Implemented  
**Module:** `geometry/components/safety/grounding.py`

```python
@dataclass
class GroundingPointParams:
    """Static grounding/bonding connection."""
    location: tuple              # (x, y, z) position
    stud_diameter: float         # [m] Grounding stud diameter
    stud_type: str               # "weld_stud", "threaded"
    resistance_max: float        # [Ω] Maximum resistance to ground
```

---

## 8. Process Instrumentation

### 8.1 Pressure Transmitter Port
**Status:** ✅ Implemented  
**Module:** `geometry/components/instrumentation/pressure_port.py`

```python
@dataclass
class PressurePortParams:
    """Pressure measurement port."""
    port_type: str               # "flush_mount", "extended", "averaging"
    connection_size: str         # "1/4 NPT", "1/2 NPT", etc.
    port_diameter: float         # [m] Port inner diameter
    location: tuple              # (x, y, z) position
```

---

### 8.2 Temperature Port
**Status:** ✅ Implemented  
**Module:** `geometry/components/instrumentation/temp_port.py`

```python
@dataclass
class TemperaturePortParams:
    """Temperature measurement port (thermowell)."""
    thermowell_diameter: float   # [m] Thermowell OD
    immersion_length: float      # [m] Immersion depth
    connection_type: str         # "threaded", "flanged", "weld"
    element_type: str            # "RTD", "thermocouple"
```

---

### 8.3 Sample Port
**Status:** ✅ Implemented  
**Module:** `geometry/components/instrumentation/sample_port.py`

```python
@dataclass
class SamplePortParams:
    """In-line sample extraction port."""
    port_diameter: float         # [m] Port diameter
    valve_type: str              # "ball", "plug", "slide"
    sample_type: str             # "isokinetic", "scoop", "thief"
    location: tuple              # Position on equipment
```

---

### 8.4 Sight Glass / Inspection Port
**Status:** ✅ Implemented  
**Module:** `geometry/components/instrumentation/sight_glass.py`

```python
@dataclass
class SightGlassParams:
    """Sight glass/inspection window."""
    glass_diameter: float        # [m] View diameter
    glass_type: str              # "borosilicate", "tempered"
    flange_size: str             # Standard flange size
    light_port: bool             # Has illumination port
    wiper: bool                  # Has internal wiper
```

---

## 9. Support Structures

### 9.1 Equipment Legs / Supports
**Status:** ✅ Implemented  
**Module:** `geometry/components/supports/legs.py`

```python
@dataclass
class EquipmentLegParams:
    """Equipment support legs."""
    leg_type: str                # "tubular", "channel", "adjustable"
    num_legs: int                # Number of legs
    leg_height: float            # [m] Leg height
    leg_diameter: float          # [m] Leg diameter/size
    foot_type: str               # "flat", "leveling", "seismic"
    load_capacity: float         # [kg] Per-leg load capacity
```

---

### 9.2 Structural Frame
**Status:** ✅ Implemented  
**Module:** `geometry/components/supports/frame.py`

```python
@dataclass
class StructuralFrameParams:
    """Structural support frame."""
    frame_type: str              # "bolted", "welded"
    material: str                # "carbon_steel", "304SS", "aluminum"
    width: float                 # [m] Frame width
    depth: float                 # [m] Frame depth
    height: float                # [m] Frame height
    column_size: str             # Column section size
    beam_size: str               # Beam section size
    platform_levels: List[float] # [m] Platform elevations
```

---

## Implementation Priority

### Phase 1: Core Classification (High Priority)
1. ✅ Zigzag Classifier - `geometry/components/zigzag_classifier.py`
2. ✅ Venturi Eductor - `geometry/components/venturi_eductor.py`
3. ✅ Additional Cyclones (Multi-Cyclone System) - `geometry/components/multi_cyclone.py`
4. ✅ Bag Filter - `geometry/components/bag_filter.py`

### Phase 2: Feed System
5. ✅ Feed Hopper - `geometry/components/feed_hopper.py`
6. ✅ Rotary Airlock - `geometry/components/rotary_airlock.py`
7. ✅ Screw Feeder - `geometry/components/screw_feeder.py`
8. ✅ De-agglomerator - `geometry/components/deagglomerator.py`

### Phase 3: Air System
9. ✅ Centrifugal Blower - `geometry/components/centrifugal_blower.py`
10. ✅ Inlet Filter - `geometry/components/air_filter.py`
11. ✅ Flow Dampers - `geometry/components/damper.py`

### Phase 4: Ductwork
12. ✅ Round Ducts - `geometry/components/ductwork.py`
13. ✅ Transitions - `geometry/components/transitions.py`
14. ✅ Elbows - `geometry/components/elbows.py`
15. ✅ Diverters - `geometry/components/diverter.py`

### Phase 5: Safety & Instrumentation
16. ✅ Explosion Vents - `geometry/components/safety/explosion_vent.py`
17. ✅ Grounding Points - `geometry/components/safety/grounding.py`
18. ✅ Pressure/Temp Ports - `geometry/components/instrumentation/`
19. ✅ Sample Ports - `geometry/components/instrumentation/sample_port.py`

### Phase 6: Support & Exhaust
20. ✅ Support Structures - `geometry/components/supports/`
21. ✅ Silencer - `geometry/components/silencer.py`
22. ✅ Exhaust Stack - `geometry/components/exhaust_stack.py`

### Phase 7: System Integration 
23. ✅ CompleteClassifierAssembly - `geometry/assembly/complete_system.py`
    - Integrates all 6 phases into unified system
    - Automatic instrumentation placement (pressure, temp, sample ports)
    - Safety equipment mounting (explosion vents, grounding)
    - Bill of materials generation
    - Factory functions: `create_complete_classifier_system()`, 
      `create_pilot_scale_system()`, `create_production_scale_system()`

---

## Typical System Specifications (Yellow Pea Protein)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Feed rate | 500-2000 kg/h | Milled flour |
| Feed particle size | d50 = 30-50 μm | After pin milling |
| Air flow rate | 2000-8000 m³/h | Depends on classifier |
| Classifier air velocity | 2-5 m/s | In zigzag channel |
| Cut size (d50) | 15-25 μm | Protein vs starch |
| Protein yield | 25-35% of feed | To protein fraction |
| Protein purity | 50-65% | In protein fraction |
| Starch purity | 60-75% | In starch fraction |
| Number of passes | 2-3 | For higher purity |
| Total system pressure drop | 3000-8000 Pa | Feed to exhaust |


┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE AIR CLASSIFIER SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   FEED SYSTEM (Phase 2)          ┌──────────────────────────────────────┐   │
│   ┌──────────────┐               │  CLASSIFICATION (Phase 1)            │   │
│   │  Feed Hopper │◄──Sight Glass │                                      │   │
│   │              │◄──Sample Port │  ┌─────────┐    ┌─────────────────┐  │   │
│   │              │◄──Level Sensor│  │ Zigzag  │───►│ Cyclone System  │  │   │
│   └──────┬───────┘◄──Grounding   │  │Classifier│   │                 │  │   │
│          │                       │  │         │   │ ◄──Explosion Vent│  │   │
│   ┌──────▼───────┐               │  │◄──Press │   │ ◄──Sight Glass   │  │   │
│   │Rotary Airlock│◄──Grounding   │  │   Port  │   │ ◄──Grounding     │  │   │
│   └──────┬───────┘               │  └────┬────┘   └────────┬─────────┘  │   │
│          │                       │       │                  │            │   │
│   ┌──────▼───────┐               └───────┼──────────────────┼────────────┘   │
│   │ Screw Feeder │◄──Grounding           │                  │               │
│   └──────┬───────┘                       ▼                  ▼               │
│          │                        ┌──────────────┐   ┌─────────────┐        │
│   ┌──────▼───────┐                │  Bag Filter  │   │ Dust Bin    │        │
│   │De-agglomerator│◄──Grounding   │              │   │             │        │
│   └──────────────┘                │◄──Explosion  │   │◄──Sample    │        │
│                                   │   Vent       │   │   Port      │        │
│                                   │◄──∆P Ports   │   │◄──Level     │        │
│   AIR SYSTEM (Phase 3)            └──────────────┘   └─────────────┘        │
│   ┌──────────────┐                                                          │
│   │ Inlet Filter │◄──∆P Ports (clean/dirty side pressure drop)              │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│   ┌──────▼───────┐     DUCTWORK (Phase 4)                                   │
│   │   Blower     │────────┬──────────────────────────────────►              │
│   │              │        │  ◄──Pressure Ports along duct                   │
│   │◄──Temp Port  │        │  ◄──Temperature Ports                           │
│   │◄──Grounding  │        │  ◄──Sample Ports at key locations               │
│   └──────────────┘        │                                                  │
│          ▲                ▼                                                  │
│   ┌──────┴───────┐   ┌──────────┐                                           │
│   │ Flow Damper  │   │  Elbows  │                                           │
│   │◄──Position   │   │Transitions│                                           │
│   │   Indicator  │   │ Diverters │                                           │
│   └──────────────┘   └──────────┘                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


## Component Status Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| Venturi Eductor | ✅ Implemented | The doc has inconsistent status - it's in venturi_eductor.py |
| Air Heater | ❌ Not Started | Optional for humidity/temperature control |
| Turbo/Rotor Classifier | ❌ Not Started | Alternative to zigzag (finer cuts) |
| Counterflow Classifier | ❌ Not Started | Alternative classifier type |
| Elutriator Column | ❌ Not Started | Alternative classifier type |
| Collection Bins/Hoppers | ❌ Not Started | Product collection vessels |
| Exhaust Fan | ❌ Not Started | Could reuse CentrifugalBlower |
| Explosion Isolation Valve | ❌ Not Started | Extends rotary airlock concept |

These are secondary/optional components that weren't included in the main 6-phase implementation plan. They provide:
Alternative classifiers (Turbo, Counterflow, Elutriator) - for different separation needs
Auxiliary equipment (Air Heater, Collection Bins) - for complete plant
Safety redundancy (Explosion Isolation) - enhanced protection
Would you like me to implement these remaining components? I can organize them as:
Phase 7A: Alternative Classifiers (Turbo, Counterflow, Elutriator)
Phase 7B: Auxiliary Equipment (Air Heater, Collection Bins, Exhaust Fan, Explosion Isolation)
Or we could proceed directly to the System Integration phase that combines all existing components. Which would you prefer?
---

## References

1. Schubert, H. (1987). "Food Particle Technology. Part I: Properties of Particles and Particulate Food Systems"
2. Dijkink, B.H. & Langelaan, H.C. (2002). "Milling and Air Classification of Legumes"
3. Tyler, R.T. (1984). "Impact Milling Quality of Grain Legumes"
4. Pelgrom, P.J.M. et al. (2013). "Dry Fractionation for Production of Functional Pea Protein"
5. NFPA 652 - Standard on the Fundamentals of Combustible Dust
6. EN 14491 - Dust Explosion Venting Protective Systems
