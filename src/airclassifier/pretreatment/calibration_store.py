"""
Calibration parameter persistence
=================================

Saves the latest calibrated parameters (coupling_factor, k_evap, gap_rate)
to a JSON file so the system can load and apply them without re-running
calibration every time.

Usage::

    from pathlib import Path
    from airclassifier.pretreatment.calibration_store import (
        save_calibration,
        load_calibration,
    )

    path = Path("utility_docs/calibration_latest.json")

    # After calibration:
    save_calibration(cal_result, path)

    # Before simulation (when not calibrating):
    data = load_calibration(path)
    if data:
        config.oscillator_coupling_factor = data["oscillator_coupling_factor"]
        material.k_evap = data["k_evap"]
        # After creating simulator: sim._sim.update_parameters(gap_adjust_rate=data["gap_adjust_rate_mm_s"])
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from .calibration import CalibrationResult

# Keys in the stored JSON
KEY_COUPLING = "oscillator_coupling_factor"
KEY_K_EVAP = "k_evap"
KEY_GAP_RATE = "gap_adjust_rate_mm_s"
KEY_K_DISPERSION = "k_dispersion"
KEY_SAVED_AT = "saved_at"
KEY_SOURCE = "source"

# Fallbacks when JSON is missing — updated from Nelder-Mead calibration
# against Run#1 PLC data (Feb 2026, 289 evals, loss=3.54).
_FALLBACK_COUPLING = 0.138
_FALLBACK_K_EVAP = 1.02e-6
_FALLBACK_GAP_RATE = 0.191
_FALLBACK_K_DISPERSION = 2.10

_cached_defaults: Optional[Tuple[float, float, float, float]] = None


def get_default_calibration_path() -> Path:
    """Path to calibration_latest.json (single source of truth).

    Uses AIRCLASSIFIER_CALIBRATION_FILE if set, else
    utility_docs/calibration_latest.json relative to cwd.
    """
    env = os.environ.get("AIRCLASSIFIER_CALIBRATION_FILE")
    if env:
        return Path(env)
    return Path.cwd() / "utility_docs" / "calibration_latest.json"


def get_calibration_defaults() -> Tuple[float, float, float, float]:
    """Return (coupling, k_evap, gap_rate, k_dispersion) from JSON or fallbacks.

    Single source of truth: config, controller, and CalibrationResult
    all use this. Cached after first load; cache is cleared on save_calibration().
    """
    global _cached_defaults
    if _cached_defaults is not None:
        return _cached_defaults
    data = load_calibration(get_default_calibration_path())
    if data:
        _cached_defaults = (
            data[KEY_COUPLING],
            data[KEY_K_EVAP],
            data[KEY_GAP_RATE],
            data.get(KEY_K_DISPERSION, _FALLBACK_K_DISPERSION),
        )
    else:
        _cached_defaults = (
            _FALLBACK_COUPLING, _FALLBACK_K_EVAP,
            _FALLBACK_GAP_RATE, _FALLBACK_K_DISPERSION,
        )
    return _cached_defaults


def save_calibration(result: "CalibrationResult", path: Path) -> None:
    """Write the latest calibration parameters to a JSON file.

    Call this after a successful calibration run so that subsequent
    runs (without --calibrate) can load and apply these values.

    Args:
        result: The CalibrationResult from the optimizer.
        path: File path (e.g. utility_docs/calibration_latest.json).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        KEY_COUPLING: result.oscillator_coupling_factor,
        KEY_K_EVAP: result.k_evap,
        KEY_GAP_RATE: result.gap_adjust_rate_mm_s,
        KEY_K_DISPERSION: result.k_dispersion,
        KEY_SAVED_AT: datetime.now(tz=timezone.utc).isoformat(),
        KEY_SOURCE: "calibration",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    # Clear cache so next get_calibration_defaults() reads the new file
    global _cached_defaults
    _cached_defaults = None


def load_calibration(path: Path) -> Optional[dict[str, Any]]:
    """Load the latest stored calibration parameters from a JSON file.

    Returns a dict with keys:
        oscillator_coupling_factor, k_evap, gap_adjust_rate_mm_s,
        saved_at (optional), source (optional).
    Returns None if the file does not exist or is invalid.

    Args:
        path: File path (e.g. utility_docs/calibration_latest.json).

    Returns:
        Dict of calibration parameters, or None.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    for key in (KEY_COUPLING, KEY_K_EVAP, KEY_GAP_RATE):
        if key not in data or not isinstance(data[key], (int, float)):
            return None
    return {
        KEY_COUPLING: float(data[KEY_COUPLING]),
        KEY_K_EVAP: float(data[KEY_K_EVAP]),
        KEY_GAP_RATE: float(data[KEY_GAP_RATE]),
        KEY_K_DISPERSION: float(data.get(KEY_K_DISPERSION, _FALLBACK_K_DISPERSION)),
        KEY_SAVED_AT: data.get(KEY_SAVED_AT),
        KEY_SOURCE: data.get(KEY_SOURCE),
    }
