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


COLORS = _Colors()
