# Milling Module — Geometry & Material Flow Audit

**Focus:** Geometric components, inlets/outlets, assembly consistency, and real-world material flow.

**Reference:** `src/airclassifier/milling/geometry/`, `physics/coupling.py` (feed step), BUILD_AND_INTEGRATE.md §3.

---

## 1. Coordinate System (Consistent)

All components use the same frame:

| Axis | Direction | Convention |
|------|-----------|------------|
| **X** | Along rotor (shaft) axis | Increase from drive end toward free end |
| **Y** | Vertical (up) | Gravity is -Y |
| **Z** | Lateral (across belt / mill width) | Right-hand rule |

**Origin:** Rotor centerline at Y = 0, Z = 0; X = 0 at drive end.

**mesh_utils:** `box_mesh(x0,y0,z0, lx,ly,lz)` uses minimum corner + dimensions. Cylinders and arcs use center + radius; arc angle 0 = +Y in the YZ plane when axis = "x".

---

## 2. Component-by-Component Summary

### 2.1 Rotor

- **Position:** Centerline at Y = 0, Z = 0; X from disc_start (e.g. 0.05 m) over active_length (0.30 m).
- **Geometry:** Shaft + support discs; discs at `rotor_diameter_m/2` (0.10 m default).
- **Real-world:** Matches horizontal-shaft rotor; rotation around X.

### 2.2 Hammers

- **Position:** Pivot radius from config (e.g. 0.06 m from shaft); rows along X with defined spacing.
- **Tip radius:** `rotor_diameter_m/2 + hammer_length_m` = 0.10 + 0.08 = **0.18 m** (default).
- **Real-world:** Swinging hammers; visualization shows extended position; physics uses tip radius for impact zone.

### 2.3 Screen

- **Position:** `center_x_m` = 0.05, `center_y/z` = 0; **inner_radius_m** = 0.188 m; arc **180°–360°** (bottom half, -Y side).
- **Arc:** `start_angle_deg` = 180 → -Y; 180° arc → bottom semicircle. Axis = "x", so arc is in YZ plane.
- **Real-world:** Curved screen at bottom of chamber; product passes through apertures to the plenum below.

### 2.4 Housing

- **Casing:** Cylindrical shell; **inner_radius_m** = 0.20 m, **length_m** = 0.40 m. Top half arc (0–180°) for visualization; side/end pieces as needed.
- **Feed opening (top):** At **y = inner_radius_m** (0.20 m); X from `feed_opening_x_offset_m` (0.10) over `feed_opening_width_m` (0.15); Z centered.
- **Discharge opening (bottom):** At **y = -inner_radius_m** (bottom); X from `discharge_opening_x_offset_m` (0.08) over width; Z centered. Flange extends to **y ≈ -outer_radius_m - 0.05**.
- **Ports:** `feed_inlet` at top center of feed opening; `discharge_outlet` at bottom center of discharge (below housing).
- **Real-world:** Top feed and bottom discharge match common hammer mill layout.

### 2.5 Feed Chute

- **Alignment:** `FeedChuteParams.from_housing()` ties chute to housing:
  - **outlet** = housing feed opening center: `outlet_x_m` = housing center_x + feed_opening_x_offset + width/2, **outlet_y_m** = **hp.inner_radius_m**, outlet_z = center_z.
  - **inlet** = top of chute: **inlet_y_m** = outlet_y_m + length_m (chute extends upward).
- **Mesh:** Vertical duct; outlet at bottom (at housing top), inlet at top (for pretreatment/hopper connection).
- **Ports:** `inlet` = top (upstream connection); `outlet` = bottom (into housing).
- **Real-world:** Gravity feed from above into the mill; chute outlet correctly aligned with housing feed opening.

### 2.6 Drive

- **Position:** `motor_x_offset_m` (e.g. -0.10), `motor_z_offset_m` = housing outer_radius + 0.10 (beside housing), `motor_y_offset_m` (e.g. -0.15).
- **Real-world:** Motor at drive end, beside casing; no impact on material flow.

---

## 3. Assembly Ports and Pipeline

Assembly exposes:

- **infeed_port:** From `feed_chute_geometry.ports["inlet"]` → **top of feed chute** (where pretreatment/hopper connects). ✓
- **outfeed_port:** From `housing_geometry.ports["discharge_outlet"]` → **below housing** (discharge flange). ✓

So:

- **Upstream (pretreatment)** connects to **infeed_port** (top of chute).
- **Downstream (classifier)** receives from **outfeed_port** (bottom of mill).

Flow direction is top → chamber → screen → bottom; ports are consistent with that.

---

## 4. Material Flow Consistency

### 4.1 Intended flow (real-world)

1. **Feed** enters at **top** of mill (feed chute → housing feed opening).
2. Material falls into **chamber** (rotor + hammers, screen at bottom).
3. **Hammers** impact and break particles; **screen** retains oversize, passes undersize.
4. **Discharge** leaves through **bottom** (through screen into plenum/discharge opening).

### 4.2 Geometry vs flow

| Stage | Geometry | Consistent with flow? |
|-------|----------|------------------------|
| Feed entry | Chute outlet at y = housing inner_radius (top); housing feed opening at same Y | ✓ Top entry |
| Chamber | Rotor at Y=Z=0; screen arc 180–360° (bottom); housing encloses | ✓ |
| Screen position | Screen inner radius 0.188 m; arc below rotor | ✓ Bottom of chamber |
| Discharge | Housing discharge opening at y = -outer_radius; flange below | ✓ Bottom exit |

### 4.3 Physics feed position (coupling._feed_step)

- New particles are created at:
  - **feed_x** = **0.15** (hardcoded),
  - **feed_y** = **chamber_radius * 0.9** (e.g. 0.198 m),
  - **feed_z** = 0 with small random spread.

**Assessment:**

- **feed_y:** Chamber radius 0.22 m → 0.198 m is just below the top (0.22); acceptable as “just inside” the chamber.
- **feed_x:** Housing feed opening center from params is **0.10 + 0.15/2 = 0.175 m**. Using 0.15 is slightly off and **not derived from geometry**. Prefer computing from housing/feed_chute params (e.g. feed opening center X) so that if config changes, physics stays aligned.

**Recommendation:** Derive feed_x (and optionally feed_y) from assembly/housing/feed_chute geometry (e.g. from `HousingParams` or a shared “feed inlet center” from the assembly) instead of a hardcoded 0.15.

---

## 5. Rotor / Screen / Housing Radii

Default config:

- **Rotor (disc) radius:** rotor_diameter_m/2 = **0.10 m**
- **Hammer tip radius:** 0.10 + 0.08 = **0.18 m**
- **Screen inner radius:** **0.188 m** → clearance to tip = **0.008 m** (8 mm) ✓
- **Housing inner radius:** **0.20 m** > screen inner → screen fits inside housing ✓

So:

- Hammer tips do not hit the screen; clearance is positive and matches `hammer_clearance_m`.
- Screen sits inside the housing at the bottom; discharge opening is below the screen.

---

## 6. Feed Chute ↔ Housing Alignment

From **FeedChuteParams.from_housing()**:

- **outlet_width_m** = hp.feed_opening_width_m  
- **outlet_depth_m** = hp.feed_opening_depth_m  
- **outlet_x_m** = hp.center_x_m + hp.feed_opening_x_offset_m + hp.feed_opening_width_m / 2  
- **outlet_y_m** = hp.inner_radius_m  
- **outlet_z_m** = hp.center_z_m  

So the chute outlet **center** and **cross-section** match the housing feed opening. The chute mesh is built with outlet at `(out_x, out_y, out_z)`; the housing feed flange is at the same Y. Alignment is correct for top feed.

---

## 7. Screen Classifier / Kernel Geometry

Screen classifier and kernel use:

- **screen_radius** = 0.21 (in screen_classifier default) vs geometry **screen_inner_radius_m** = 0.188. If classifier is not updated from config, **0.21** is inconsistent with current geometry (screen is at 0.188).
- **screen_x_start** = 0.05, **screen_x_end** = 0.35 → consistent with screen length and position (center_x 0.05, length 0.30).
- **screen_start_angle** = π, **screen_arc_angle** = π → bottom half, consistent with screen geometry.

No change needed when using ScreenClassifier.from_config(); radius and bounds match geometry.

---

## 8. Summary Table

| Item | Status | Note |
|------|--------|------|
| Coordinate system | ✓ | X = rotor, Y = up, Z = lateral; consistent |
| Feed chute → housing | ✓ | Outlet aligned to housing feed opening (top) |
| Infeed port | ✓ | Top of chute (pipeline from pretreatment) |
| Discharge port | ✓ | Below housing (pipeline to classifier) |
| Screen position | ✓ | Bottom half arc, inside housing |
| Rotor/screen clearance | ✓ | tip_radius 0.18, screen 0.188, clearance 8 mm |
| Material flow direction | ✓ | Top → chamber → screen → bottom |
| Physics feed position | ⚠️ | feed_x=0.15 hardcoded; should derive from geometry |
| Screen radius in classifier | ✓ | from_config uses config.screen_inner_radius_m |

---

## 9. Recommendations

1. **Physics feed position:** In `physics/coupling.py` `_feed_step()`, compute feed inlet center (and optionally bounds) from the assembly or from housing/feed_chute params (e.g. feed_opening_x_offset_m + feed_opening_width_m/2, and feed_opening depth for z spread), instead of hardcoded `feed_x=0.15`.
2. **Optional:** Document in BUILD_AND_INTEGRATE.md or in code that **infeed_port** is the top of the feed chute (upstream connection) and **outfeed_port** is the discharge below the housing (downstream connection), so pipeline builders keep the flow direction explicit.

---

*Audit complete. The assembly makes real-world sense; main improvements are deriving physics feed position from geometry and aligning screen classifier radius with config.*
