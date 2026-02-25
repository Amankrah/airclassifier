"""
Design Tokens / Theme
=====================

Central color palette and design tokens for the Air Classifier GUI.
This module has NO intra-package imports to avoid circular dependencies.
"""


class _Colors:
    """Central color palette used by the entire application."""
    # --- surfaces ---
    BG_DARKEST    = "#141417"
    BG_DARKER     = "#1a1a1f"
    BG_DARK       = "#1e1e24"
    BG_BASE       = "#242429"
    BG_ELEVATED   = "#2a2a31"
    BG_SURFACE    = "#303038"
    BG_HOVER      = "#363640"

    # --- borders & separators ---
    BORDER        = "#3a3a45"
    BORDER_SUBTLE = "#2f2f39"
    BORDER_FOCUS  = "#4a9eff"

    # --- text ---
    TEXT_PRIMARY   = "#e4e4eb"
    TEXT_SECONDARY = "#a0a0b0"
    TEXT_MUTED     = "#6b6b7b"
    TEXT_DISABLED  = "#505060"
    TEXT_INVERSE   = "#ffffff"

    # --- accent ---
    ACCENT         = "#4a9eff"
    ACCENT_HOVER   = "#6bb3ff"
    ACCENT_PRESSED = "#2d7de0"
    ACCENT_MUTED   = "#2a4a70"

    # --- semantic ---
    SUCCESS        = "#3dd68c"
    SUCCESS_MUTED  = "#1a3d2e"
    WARNING        = "#f0b429"
    WARNING_MUTED  = "#3d3220"
    DANGER         = "#ef5350"
    DANGER_MUTED   = "#3d2020"
    INFO           = "#4fc3f7"

    # --- category (component palette) ---
    CAT_CLASSIFICATION = "#6495ed"
    CAT_CYCLONES       = "#66cdaa"
    CAT_FILTRATION     = "#f48fb1"
    CAT_FEED           = "#ffcc80"
    CAT_AIR            = "#81d4fa"
    CAT_DUCTWORK       = "#b0bec5"
    CAT_EXHAUST        = "#ce93d8"

    # --- milling-specific (digital twin) ---
    MILLING_PRIMARY    = "#4ade80"   # Green - size reduction / milling
    MILLING_SECONDARY  = "#22c55e"   # Darker green
    MILLING_MUTED      = "#14532d"   # Muted green background

    # --- pretreatment-specific (RF heating digital twin) ---
    PRETREAT_PRIMARY   = "#f97316"   # Orange - RF heating / thermal
    PRETREAT_SECONDARY = "#ea580c"   # Darker orange
    PRETREAT_MUTED     = "#431407"   # Muted orange background
    PRETREAT_RF        = "#fbbf24"   # Amber - RF power indicator
    PRETREAT_MOISTURE  = "#38bdf8"   # Sky blue - moisture content
    PRETREAT_THERMAL   = "#ef4444"   # Red - temperature

    # --- KPI semantic colors ---
    KPI_THROUGHPUT     = "#60a5fa"   # Blue - flow rate
    KPI_POWER          = "#f97316"   # Orange - energy consumption
    KPI_QUALITY        = "#a78bfa"   # Purple - d50/PSD quality
    KPI_SIZE           = "#4ade80"   # Green - particle size
    KPI_EFFICIENCY     = "#fbbf24"   # Yellow/gold - efficiency

    # --- Pretreatment KPI colors ---
    KPI_TEMPERATURE    = "#ef4444"   # Red - temperature
    KPI_MOISTURE       = "#38bdf8"   # Sky blue - moisture
    KPI_RF_POWER       = "#f97316"   # Orange - RF power
    KPI_ANODE_CURRENT  = "#fbbf24"   # Amber - anode current
    KPI_ELECTRODE_GAP  = "#a78bfa"   # Purple - electrode gap
    KPI_PROTEIN        = "#4ade80"   # Green - protein quality
    KPI_ENERGY         = "#fb923c"   # Light orange - energy consumed

    # --- gradients (for glassmorphism) ---
    GLASS_START        = "#1e293b"
    GLASS_END          = "#0f172a"
    GLASS_BORDER       = "rgba(255, 255, 255, 0.1)"

    # --- glow effects ---
    GLOW_SUCCESS       = "rgba(74, 222, 128, 0.3)"
    GLOW_ACCENT        = "rgba(74, 158, 255, 0.3)"
    GLOW_WARNING       = "rgba(240, 180, 41, 0.3)"
    GLOW_DANGER        = "rgba(239, 83, 80, 0.3)"

    # --- chart colors ---
    CHART_PRIMARY      = "#4a9eff"
    CHART_SECONDARY    = "#4ade80"
    CHART_TERTIARY     = "#f97316"
    CHART_QUATERNARY   = "#a78bfa"
    CHART_GRID         = "#2a2a35"
    CHART_AXIS         = "#4a4a55"


COLORS = _Colors()


# Animation durations (ms)
class _Animations:
    """Standard animation durations for consistent UX."""
    INSTANT    = 50
    FAST       = 150
    NORMAL     = 250
    SLOW       = 400
    VERY_SLOW  = 600


ANIMATIONS = _Animations()


# Standard shadow styles
SHADOWS = {
    "sm": "0 1px 2px rgba(0, 0, 0, 0.3)",
    "md": "0 4px 6px rgba(0, 0, 0, 0.4)",
    "lg": "0 10px 15px rgba(0, 0, 0, 0.5)",
    "glow": "0 0 20px rgba(74, 222, 128, 0.2)",
}
