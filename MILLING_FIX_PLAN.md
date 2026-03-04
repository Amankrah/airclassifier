# Hammer Mill Digital Twin - Fix Plan [COMPLETED]

## Overview

This plan addressed issues identified in the audit of the hammer mill digital twin against real-world physics principles.

---

## Issue 1: Screen Efficiency Bug (3832.4%) ✅ FIXED

### Root Cause
The calculation used **particle count** (70,938 passed / 1,851 fed = 3832%) which is meaningless when fragmentation creates more particles than were fed.

### Fix Applied
- Changed to **mass-based yield** calculation: `(mass_discharged / mass_fed) × 100`
- Added **mass balance display**: shows fed, discharged, holdup, and loss
- Renamed "Screen Efficiency" → "Yield"

### Files Modified
- `results_page.py` lines 565-574, 617-656

---

## Issue 2: Confusing "Seed d50" Terminology ✅ FIXED

### Root Cause
Label said "Seed d50" but showed retained particle d50 (mostly fragments, not seeds).

### Fix Applied
- Renamed "Seed d50" → "Retained d50" (KPI card, summary panel)
- Renamed "Seeds (retained)" → "Holdup (retained)" (PSD toggle)
- Renamed "Retained Seeds" → "Retained Holdup" (analytics column)

### Files Modified
- `results_page.py` lines 58, 144, 367-368, 565-574

---

## Issue 3: Batch-Based Simulation Time ✅ FIXED

### Root Cause
User had to manually set duration; simulation stopped at fixed time regardless of material state.

### Fix Applied
1. Added **"batch_complete"** termination mode to `ConvergenceDetector`
2. **Auto-switches** to batch_complete when `seeds_feed_mass_kg > 0` (in time mode)
3. Simulation runs until:
   - All seeds are fed (feeding phase: 0-50% progress)
   - Mill chamber empties (discharge phase: 50-100% progress)
4. Estimated duration = `feeding_time × 2.5` (allows time for processing)

### Files Modified
- `convergence.py` - Added batch_complete mode, feed tracking, progress calculation
- `coupling.py` - Pass feed_rate to convergence detector
- `milling_page.py` - Auto-switch logic, batch_complete in dropdown
- `control_panel.py` - batch_complete in dropdown and mode handling

---

## Issue 4: Mass Balance Clarity ✅ FIXED

### Fix Applied
Process Summary now shows:
- **Duration**: simulation time
- **Total Mass Fed**: mass input
- **Flour Mass (discharged)**: mass through screen
- **Holdup Mass (retained)**: mass still in mill
- **Yield**: `discharged / fed × 100%`
- **Mass Loss**: `fed - discharged - holdup` (in grams and %)

---

## Summary of Changes

### `results_page.py`
| Change | Description |
|--------|-------------|
| Line 58 | "Seed d50" → "Retained d50" |
| Line 144 | "Seeds (retained)" → "Holdup (retained)" |
| Line 367-368 | "Retained Seeds" → "Retained Holdup" |
| Lines 565-574 | Updated row_data with Yield, Mass Loss |
| Lines 617-656 | Mass-based yield and mass loss calculation |

### `convergence.py`
| Change | Description |
|--------|-------------|
| Docstring | Added batch_complete mode description |
| Lines 71-78 | Added `_cumulative_feed_kg`, `_target_feed_mass_kg`, `_feeding_complete` |
| `reset()` | Clear new tracking fields |
| `update()` | Accept `feed_rate_kg_per_s`, track feed mass, detect feeding complete |
| `should_terminate()` | Handle batch_complete mode |
| `progress_pct` | Feeding (0-50%) + discharge (50-100%) for batch mode |
| `TerminationConfig` | Added `target_feed_mass_kg` parameter |

### `coupling.py`
| Change | Description |
|--------|-------------|
| Line ~480 | Pass `feed_rate_kg_per_s` to convergence detector |

### `milling_page.py`
| Change | Description |
|--------|-------------|
| Lines 371-377 | Added "Batch complete" to dropdown |
| `_on_term_mode_changed()` | Handle index 4 (batch mode) |
| `_get_termination_mode()` | Return "batch_complete" for index 4 |
| `_on_run()` | Auto-switch to batch_complete, pass target_feed_mass_kg |

### `control_panel.py`
| Change | Description |
|--------|-------------|
| Lines 373-378 | Added "Batch complete" to dropdown |
| `_on_term_mode_changed()` | Handle batch mode, auto-set 1 kg default |
| `_on_seeds_feed_mass_changed()` | **NEW** - Auto-switch to batch mode when input mass > 0 |
| Signal connection | Connect `seeds_feed_mass_spin` to new handler |
| `get_recipe()` | Include "batch_complete" in modes |
| `get_termination_mode()` | Include "batch_complete" in modes |

---

## Expected Results After Fix

| Metric | Before | After |
|--------|--------|-------|
| Screen Efficiency | 3832.4% | Removed |
| Yield | (not shown) | 35.0% (mass-based) |
| Mass Loss | (not shown) | X g (Y%) |
| Seed d50 | "Seed d50" | "Retained d50" |
| Retained column | "Retained Seeds" | "Retained Holdup" |
| Batch simulation | Manual duration | Auto until mill empty |
| Progress (batch) | Time-based | Feeding 0-50% → Discharge 50-100% |
| **UX: Input mass > 0** | Shows "Time-based" + Duration 60s | Auto-switches to "Batch complete", hides Duration |

---

## Testing Recommendations

1. **UX auto-switch test**: Set `Input mass (seeds)` to 1.0 kg → termination dropdown should **immediately** switch to "Batch complete" and Duration field should hide
2. **UX revert test**: Set `Input mass (seeds)` back to 0 → should revert to "Time-based" mode
3. **Run a batch simulation**: With input mass = 1.0 kg, click Run → simulation should run until batch complete
4. **Verify progress**: Progress should show 0-50% during feeding, then 50-100% as mill empties
5. **Check mass balance**: `fed = discharged + holdup + loss` should hold (loss < 5%)
6. **Verify yield**: Should be ~35% at 2s (startup), approaching 100% at batch completion
7. **Run longer**: 30-60 seconds to see steady-state behavior
