"""
Desirability Panel Widget
=========================

Modern glassmorphism-styled panel displaying the 5-dimension
process desirability scoring for RF pretreatment.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QFont
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout,
    QLabel, QWidget, QProgressBar, QSizePolicy,
    QGraphicsDropShadowEffect,
)

from ...theme import COLORS


class ScoreGauge(QFrame):
    """Large circular gauge for overall desirability score."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._score = 0.0
        self._target_score = 0.0
        self._animation: Optional[QPropertyAnimation] = None

        self.setFixedSize(140, 140)

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, value: float):
        self._score = value
        self.update()

    def set_score(self, value: float, animate: bool = True):
        """Set the score value with optional animation."""
        self._target_score = value
        if animate:
            if self._animation:
                self._animation.stop()
            self._animation = QPropertyAnimation(self, b"score")
            self._animation.setDuration(600)
            self._animation.setStartValue(self._score)
            self._animation.setEndValue(value)
            self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._animation.start()
        else:
            self._score = value
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = min(self.width(), self.height())
        margin = 8
        diameter = size - 2 * margin

        # Background circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS.BG_DARKEST))
        painter.drawEllipse(margin, margin, diameter, diameter)

        # Score arc
        arc_width = 12
        score_10 = min(10, max(0, self._score))
        sweep_angle = int(-360 * score_10 / 10)

        # Color based on score
        if score_10 >= 7.0:
            color = QColor(COLORS.SUCCESS)
        elif score_10 >= 4.0:
            color = QColor(COLORS.WARNING)
        else:
            color = QColor(COLORS.DANGER)

        # Draw arc background
        painter.setPen(Qt.PenStyle.NoPen)
        arc_rect = (margin + arc_width // 2, margin + arc_width // 2,
                    diameter - arc_width, diameter - arc_width)

        from PySide6.QtGui import QPen
        bg_pen = QPen(QColor(COLORS.BG_SURFACE))
        bg_pen.setWidth(arc_width)
        bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(int(arc_rect[0]), int(arc_rect[1]),
                       int(arc_rect[2]), int(arc_rect[3]),
                       90 * 16, -360 * 16)

        # Draw score arc
        score_pen = QPen(color)
        score_pen.setWidth(arc_width)
        score_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(score_pen)
        painter.drawArc(int(arc_rect[0]), int(arc_rect[1]),
                       int(arc_rect[2]), int(arc_rect[3]),
                       90 * 16, sweep_angle * 16)

        # Draw score text
        painter.setPen(color)
        font = QFont()
        font.setPixelSize(32)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{score_10:.1f}")

        # Draw label
        painter.setPen(QColor(COLORS.TEXT_MUTED))
        font.setPixelSize(10)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        label_rect = self.rect().adjusted(0, 45, 0, 0)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter, "/ 10")


class DimensionCard(QFrame):
    """Individual dimension score card with progress bar."""

    def __init__(
        self,
        key: str,
        title: str,
        color: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._key = key
        self._color = color
        self._score = 0.0

        self._setup_ui(title)

    def _setup_ui(self, title: str):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Score value
        self._value_label = QLabel("--")
        self._value_label.setStyleSheet(f"""
            font-size: 18pt;
            font-weight: 700;
            color: {self._color};
        """)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._value_label)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background: {COLORS.BG_DARKEST};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {self._color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self._bar)

        # Title
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"""
            font-size: 8pt;
            color: {COLORS.TEXT_MUTED};
        """)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

    def set_score(self, score: float, animate: bool = True):
        """Set the dimension score (0-1)."""
        self._score = score
        pct = int(round(score * 100))
        self._value_label.setText(f"{pct}%")
        self._bar.setValue(pct)

    def clear(self):
        """Reset the card."""
        self._score = 0.0
        self._value_label.setText("--")
        self._bar.setValue(0)


class DesirabilityPanel(QFrame):
    """Panel displaying 5-dimension process desirability scoring.

    Dimensions:
    - Thermal Treatment
    - Flavour Improvement
    - Protein Preservation
    - Moisture Retention
    - Energy Efficiency

    Signals:
        dimension_clicked(str): Emitted when a dimension card is clicked
    """

    dimension_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, compact: bool = False):
        super().__init__(parent)
        self._compact = compact
        self._setup_style()
        self._setup_ui()

    def _setup_style(self):
        self.setObjectName("desirabilityPanel")
        self.setStyleSheet(f"""
            QFrame#desirabilityPanel {{
                background: {COLORS.BG_BASE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 10px;
            }}
        """)

        # Add subtle glow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Process Desirability")
        title.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
        """)
        header.addWidget(title)

        subtitle = QLabel("Protein Separation & Flavour")
        subtitle.setStyleSheet(f"""
            font-size: 8pt;
            color: {COLORS.TEXT_MUTED};
        """)
        header.addWidget(subtitle)
        header.addStretch()
        layout.addLayout(header)

        # Main content
        content = QHBoxLayout()
        content.setSpacing(16)

        # Overall score gauge
        self._overall_gauge = ScoreGauge()
        content.addWidget(self._overall_gauge)

        # Dimension cards grid
        dims_layout = QVBoxLayout()
        dims_layout.setSpacing(8)

        # Row 1
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self._cards: Dict[str, DimensionCard] = {}

        dim_defs = [
            ("d_thermal", "Thermal\nTreatment", COLORS.DANGER),
            ("d_flavour", "Flavour\nImprovement", COLORS.WARNING),
            ("d_protein", "Protein\nPreservation", COLORS.SUCCESS),
            ("d_moisture", "Moisture\nRetention", COLORS.INFO),
            ("d_energy", "Energy\nEfficiency", COLORS.KPI_ENERGY),
        ]

        for i, (key, title, color) in enumerate(dim_defs):
            card = DimensionCard(key, title, color)
            self._cards[key] = card

            if i < 3:
                row1.addWidget(card)
            else:
                if i == 3:
                    dims_layout.addLayout(row1)
                    row2 = QHBoxLayout()
                    row2.setSpacing(8)
                row2.addWidget(card)

        # Add stretch to row2 to balance with row1
        row2.addStretch()
        dims_layout.addLayout(row2)

        content.addLayout(dims_layout, 1)
        layout.addLayout(content)

    def update_scores(
        self,
        overall: Optional[float] = None,
        d_thermal: Optional[float] = None,
        d_flavour: Optional[float] = None,
        d_protein: Optional[float] = None,
        d_moisture: Optional[float] = None,
        d_energy: Optional[float] = None,
        animate: bool = True,
    ):
        """Update the desirability scores.

        Args:
            overall: Overall score (0-10)
            d_thermal: Thermal treatment score (0-1)
            d_flavour: Flavour improvement score (0-1)
            d_protein: Protein preservation score (0-1)
            d_moisture: Moisture retention score (0-1)
            d_energy: Energy efficiency score (0-1)
            animate: Whether to animate value changes
        """
        if overall is not None:
            self._overall_gauge.set_score(overall, animate=animate)

        scores = {
            "d_thermal": d_thermal,
            "d_flavour": d_flavour,
            "d_protein": d_protein,
            "d_moisture": d_moisture,
            "d_energy": d_energy,
        }

        for key, score in scores.items():
            if score is not None and key in self._cards:
                self._cards[key].set_score(score, animate=animate)

    def update_from_result(self, desirability_result: Any, animate: bool = True):
        """Update from a DesirabilityResult object.

        Args:
            desirability_result: Object with overall_10, d_thermal, d_flavour, etc.
            animate: Whether to animate changes
        """
        if desirability_result is None:
            return

        overall = getattr(desirability_result, "overall_10", None)
        self.update_scores(
            overall=overall,
            d_thermal=getattr(desirability_result, "d_thermal", None),
            d_flavour=getattr(desirability_result, "d_flavour", None),
            d_protein=getattr(desirability_result, "d_protein", None),
            d_moisture=getattr(desirability_result, "d_moisture", None),
            d_energy=getattr(desirability_result, "d_energy", None),
            animate=animate,
        )

    def clear(self):
        """Reset all scores."""
        self._overall_gauge.set_score(0, animate=False)
        for card in self._cards.values():
            card.clear()

    @property
    def overall_score(self) -> float:
        return self._overall_gauge.score
