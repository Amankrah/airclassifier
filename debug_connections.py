"""Debug script to understand connection points for complete system."""
import sys
sys.path.insert(0, 'src')

import numpy as np
from airclassifier.geometry.assembly.complete_system import create_core_connections_system

# Create the system
system = create_core_connections_system()
p = system.params

print("=" * 70)
print("COMPLETE SYSTEM - CONNECTION POINT ANALYSIS")
print("=" * 70)

# Get subsystems and offsets
classification = system._subsystems.get('classification')
class_offset = np.array(system._subsystems.get('classification_offset', (0, 0, 0)))

feed_system = system._subsystems.get('feed_system')
feed_offset = np.array(system._subsystems.get('feed_system_offset', (0, 0, 0)))

air_system = system._subsystems.get('air_system')
air_offset = np.array(system._subsystems.get('air_system_offset', (0, 0, 0)))

print(f"\nSYSTEM OFFSETS:")
print(f"  Classification: {class_offset}")
print(f"  Feed System:    {feed_offset}")
print(f"  Air System:     {air_offset}")

# ============================================================
# CONNECTION 1: Air System -> Venturi air_inlet
# ============================================================
print("\n" + "=" * 70)
print("CONNECTION 1: Air System -> Venturi air_inlet")
print("=" * 70)

# Air system outlet (damper)
if air_system and hasattr(air_system, 'dampers') and air_system.dampers:
    last_damper = air_system.dampers[-1]
    damper_local_pos = np.array(air_system._damper_positions[-1])
    damper_outlet_port = last_damper.ports['outlet']
    damper_outlet_local = damper_local_pos + np.array(damper_outlet_port.position)
    damper_outlet_world = air_offset + damper_outlet_local
    damper_outlet_dir = damper_outlet_port.direction
    damper_outlet_d = damper_outlet_port.diameter

    print(f"\nAIR SYSTEM OUTLET (Damper):")
    print(f"  Air system offset:  {air_offset}")
    print(f"  Damper local pos:   {damper_local_pos}")
    print(f"  Outlet port local:  {damper_outlet_port.position}")
    print(f"  Outlet WORLD pos:   {damper_outlet_world}")
    print(f"  Outlet direction:   {damper_outlet_dir}")
    print(f"  Outlet diameter:    {damper_outlet_d * 1000:.0f} mm")

# Venturi air inlet
if classification:
    venturi = classification.venturi
    class_positions = classification.get_component_positions()
    venturi_local_pos = np.array(class_positions['venturi'])
    venturi_air_port = venturi.ports['air_inlet']
    venturi_air_local = venturi_local_pos + np.array(venturi_air_port.position)
    venturi_air_world = class_offset + venturi_air_local
    venturi_air_dir = venturi_air_port.direction
    venturi_air_d = venturi_air_port.diameter

    print(f"\nVENTURI AIR INLET:")
    print(f"  Class offset:       {class_offset}")
    print(f"  Venturi local pos:  {venturi_local_pos}")
    print(f"  Air inlet local:    {venturi_air_port.position}")
    print(f"  Air inlet WORLD:    {venturi_air_world}")
    print(f"  Inlet direction:    {venturi_air_dir} (expects air from this direction)")
    print(f"  Inlet diameter:     {venturi_air_d * 1000:.0f} mm")

    # Calculate delta
    delta1 = venturi_air_world - damper_outlet_world
    print(f"\n  DELTA (outlet -> inlet): {delta1}")
    print(f"  Distance: {np.linalg.norm(delta1):.3f} m")

# ============================================================
# CONNECTION 2: Feed System -> Venturi solids_inlet
# ============================================================
print("\n" + "=" * 70)
print("CONNECTION 2: Feed System -> Venturi solids_inlet")
print("=" * 70)

# Feed system outlet (deagglomerator)
if feed_system:
    feed_positions = feed_system.get_component_positions()
    deagg_local_pos = np.array(feed_positions['deagglomerator'])
    deagg_outlet_port = feed_system.deagglomerator.ports['outlet']
    deagg_outlet_local = deagg_local_pos + np.array(deagg_outlet_port.position)
    deagg_outlet_world = feed_offset + deagg_outlet_local
    deagg_outlet_dir = deagg_outlet_port.direction
    deagg_outlet_d = deagg_outlet_port.diameter

    print(f"\nFEED SYSTEM OUTLET (Deagglomerator):")
    print(f"  Feed system offset: {feed_offset}")
    print(f"  Deagg local pos:    {deagg_local_pos}")
    print(f"  Outlet port local:  {deagg_outlet_port.position}")
    print(f"  Outlet WORLD pos:   {deagg_outlet_world}")
    print(f"  Outlet direction:   {deagg_outlet_dir}")
    print(f"  Outlet diameter:    {deagg_outlet_d * 1000:.0f} mm")

# Venturi solids inlet
if classification:
    venturi_solids_port = venturi.ports['solids_inlet']
    venturi_solids_local = venturi_local_pos + np.array(venturi_solids_port.position)
    venturi_solids_world = class_offset + venturi_solids_local
    venturi_solids_dir = venturi_solids_port.direction
    venturi_solids_d = venturi_solids_port.diameter

    print(f"\nVENTURI SOLIDS INLET:")
    print(f"  Solids inlet local: {venturi_solids_port.position}")
    print(f"  Solids inlet WORLD: {venturi_solids_world}")
    print(f"  Inlet direction:    {venturi_solids_dir} (expects material from this direction)")
    print(f"  Inlet diameter:     {venturi_solids_d * 1000:.0f} mm")

    # Calculate delta
    delta2 = venturi_solids_world - deagg_outlet_world
    print(f"\n  DELTA (outlet -> inlet): {delta2}")
    print(f"  Distance: {np.linalg.norm(delta2):.3f} m")

# ============================================================
# CONNECTION 3: Bag Filter -> Exhaust Silencer
# ============================================================
print("\n" + "=" * 70)
print("CONNECTION 3: Bag Filter -> Exhaust Silencer")
print("=" * 70)

# Bag filter clean air outlet
if classification:
    bag_filter = classification.bag_filter
    bf_local_pos = np.array(classification._component_positions['bag_filter'])
    bf_outlet_port = bag_filter.ports['clean_air_outlet']
    bf_outlet_local = bf_local_pos + np.array(bf_outlet_port.position)
    bf_outlet_world = class_offset + bf_outlet_local
    bf_outlet_dir = bf_outlet_port.direction
    bf_outlet_d = bf_outlet_port.diameter

    print(f"\nBAG FILTER CLEAN AIR OUTLET:")
    print(f"  Bag filter local:   {bf_local_pos}")
    print(f"  Outlet port local:  {bf_outlet_port.position}")
    print(f"  Outlet WORLD pos:   {bf_outlet_world}")
    print(f"  Outlet direction:   {bf_outlet_dir}")
    print(f"  Outlet diameter:    {bf_outlet_d * 1000:.0f} mm")

# Silencer inlet
silencer = system._components.get('silencer')
if silencer:
    silencer_center = np.array(silencer.params.center)
    silencer_dir = np.array(silencer.params.direction_normalized)
    silencer_len = silencer.params.length
    silencer_d = silencer.params.diameter
    # Inlet is at center - length/2 * direction
    silencer_inlet_world = silencer_center - silencer_dir * (silencer_len / 2)

    print(f"\nSILENCER INLET:")
    print(f"  Silencer center:    {silencer_center}")
    print(f"  Silencer direction: {silencer_dir}")
    print(f"  Silencer length:    {silencer_len * 1000:.0f} mm")
    print(f"  Inlet WORLD pos:    {silencer_inlet_world}")
    print(f"  Inlet diameter:     {silencer_d * 1000:.0f} mm")

    # Calculate delta
    delta3 = silencer_inlet_world - bf_outlet_world
    print(f"\n  DELTA (outlet -> inlet): {delta3}")
    print(f"  Distance: {np.linalg.norm(delta3):.3f} m")

print("\n" + "=" * 70)
print("DUCT ROUTING REQUIREMENTS SUMMARY")
print("=" * 70)

if 'damper_outlet_world' in dir() and 'venturi_air_world' in dir():
    print(f"\n1. AIR -> VENTURI:")
    print(f"   From: {damper_outlet_world} (damper outlet, dir={damper_outlet_dir})")
    print(f"   To:   {venturi_air_world} (venturi air inlet, expects from dir={venturi_air_dir})")
    print(f"   Note: Venturi expects air from BELOW (-Y), so duct must approach from below")

if 'deagg_outlet_world' in dir() and 'venturi_solids_world' in dir():
    print(f"\n2. FEED -> VENTURI:")
    print(f"   From: {deagg_outlet_world} (deagg outlet, dir={deagg_outlet_dir})")
    print(f"   To:   {venturi_solids_world} (venturi solids inlet, expects from dir={venturi_solids_dir})")
    print(f"   Note: Venturi expects solids from SIDE (+X), angled gravity chute")

if 'bf_outlet_world' in dir() and 'silencer_inlet_world' in dir():
    print(f"\n3. BAG FILTER -> SILENCER:")
    print(f"   From: {bf_outlet_world} (bag filter outlet, dir={bf_outlet_dir})")
    print(f"   To:   {silencer_inlet_world} (silencer inlet)")
    print(f"   Note: Bag filter outlet points UP (+Y)")

print("\n" + "=" * 70)
