"""
Timeline Widget
===============

Playback control widget with timeline scrubbing, speed control,
and waveform visualization for milling simulations.
"""

from __future__ import annotations

from typing import List, Optional, Callable
from enum import Enum

from PySide6.QtCore import (
    Qt, Signal, Slot, QTimer, QRectF, QPointF,
    Property, QPropertyAnimation, QEasingCurve,
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QPainterPath, QFont, QMouseEvent,
)
from PySide6.QtWidgets import (
    QWidget, QFrame, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QSizePolicy,
    QGraphicsDropShadowEffect,
)

from ...theme import COLORS, ANIMATIONS


class PlaybackState(Enum):
    """Playback state machine."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class WaveformWidget(QWidget):
    """Mini waveform display for timeline visualization."""

    cursor_moved = Signal(float)  # Emits normalized position [0, 1]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._values: List[float] = []
        self._cursor_pos: float = 0.0  # Normalized [0, 1]
        self._is_dragging: bool = False
        self._highlight_color = QColor(COLORS.MILLING_PRIMARY)
        self._track_color = QColor(COLORS.BORDER_SUBTLE)

        self.setMinimumHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_values(self, values: List[float]):
        """Set waveform data."""
        self._values = values
        self.update()

    def set_cursor(self, position: float):
        """Set cursor position (normalized 0-1)."""
        self._cursor_pos = max(0.0, min(1.0, position))
        self.update()

    def clear(self):
        """Clear waveform data."""
        self._values = []
        self._cursor_pos = 0.0
        self.update()

    def paintEvent(self, event):
        """Render the waveform."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        padding = 4

        # Background
        painter.fillRect(rect, QColor(COLORS.BG_DARKEST))

        if not self._values:
            # Draw placeholder
            painter.setPen(QColor(COLORS.TEXT_MUTED))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No data")
            return

        # Calculate bar dimensions
        n = len(self._values)
        bar_width = max(2, (rect.width() - 2 * padding) / n)
        chart_height = rect.height() - 2 * padding

        min_val = min(self._values)
        max_val = max(self._values)
        val_range = max_val - min_val if max_val != min_val else 1

        # Draw bars
        cursor_x = padding + self._cursor_pos * (rect.width() - 2 * padding)

        for i, val in enumerate(self._values):
            x = padding + i * bar_width
            norm_val = (val - min_val) / val_range
            bar_height = norm_val * chart_height * 0.8 + chart_height * 0.1

            # Color based on cursor position
            if x < cursor_x:
                color = self._highlight_color
            else:
                color = self._track_color

            painter.fillRect(
                int(x), int(rect.height() - padding - bar_height),
                int(bar_width - 1), int(bar_height),
                color
            )

        # Draw cursor line
        pen = QPen(QColor(COLORS.TEXT_PRIMARY), 2)
        painter.setPen(pen)
        painter.drawLine(
            int(cursor_x), padding,
            int(cursor_x), rect.height() - padding
        )

    def mousePressEvent(self, event: QMouseEvent):
        """Start dragging cursor."""
        self._is_dragging = True
        self._update_cursor_from_mouse(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent):
        """Update cursor while dragging."""
        if self._is_dragging:
            self._update_cursor_from_mouse(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop dragging."""
        self._is_dragging = False

    def _update_cursor_from_mouse(self, x: float):
        """Update cursor position from mouse x coordinate."""
        padding = 4
        width = self.width() - 2 * padding
        if width > 0:
            pos = (x - padding) / width
            pos = max(0.0, min(1.0, pos))
            self._cursor_pos = pos
            self.cursor_moved.emit(pos)
            self.update()


class TimelineWidget(QFrame):
    """Full timeline control widget with playback and scrubbing.

    Signals:
        play_clicked(): Play button pressed
        pause_clicked(): Pause button pressed
        stop_clicked(): Stop button pressed
        speed_changed(float): Playback speed changed
        seek(float): User seeked to position [0, 1]
    """

    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    speed_changed = Signal(float)
    seek = Signal(float)

    # Playback speeds
    SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = PlaybackState.STOPPED
        self._current_time: float = 0.0
        self._total_time: float = 0.0
        self._speed_index: int = 3  # 1.0x default

        self._setup_style()
        self._setup_ui()
        self._connect_signals()

    def _setup_style(self):
        """Apply glassmorphism styling."""
        self.setObjectName("timelineWidget")
        self.setStyleSheet(f"""
            QFrame#timelineWidget {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS.GLASS_START},
                    stop:1 {COLORS.GLASS_END});
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
            }}
        """)
        self.setFixedHeight(70)

        # Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(-2)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

    def _setup_ui(self):
        """Build the timeline UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Waveform
        self._waveform = WaveformWidget()
        layout.addWidget(self._waveform)

        # Controls row
        controls = QHBoxLayout()
        controls.setSpacing(8)

        # Playback buttons
        btn_style = f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12pt;
                color: {COLORS.TEXT_PRIMARY};
                min-width: 32px;
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                border-color: {COLORS.TEXT_MUTED};
            }}
            QPushButton:pressed {{
                background: {COLORS.ACCENT_MUTED};
            }}
            QPushButton:disabled {{
                color: {COLORS.TEXT_DISABLED};
                background: {COLORS.BG_DARK};
            }}
        """

        self._skip_back_btn = QPushButton("<<")
        self._skip_back_btn.setStyleSheet(btn_style)
        self._skip_back_btn.setToolTip("Skip to start")
        controls.addWidget(self._skip_back_btn)

        self._play_btn = QPushButton(">")
        self._play_btn.setStyleSheet(btn_style.replace(
            f"background: {COLORS.BG_SURFACE}",
            f"background: {COLORS.SUCCESS_MUTED}"
        ))
        self._play_btn.setToolTip("Play (F5)")
        controls.addWidget(self._play_btn)

        self._pause_btn = QPushButton("||")
        self._pause_btn.setStyleSheet(btn_style)
        self._pause_btn.setToolTip("Pause (F6)")
        self._pause_btn.setEnabled(False)
        controls.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("[]")
        self._stop_btn.setStyleSheet(btn_style.replace(
            f"background: {COLORS.BG_SURFACE}",
            f"background: {COLORS.DANGER_MUTED}"
        ))
        self._stop_btn.setToolTip("Stop (Shift+F5)")
        self._stop_btn.setEnabled(False)
        controls.addWidget(self._stop_btn)

        self._skip_fwd_btn = QPushButton(">>")
        self._skip_fwd_btn.setStyleSheet(btn_style)
        self._skip_fwd_btn.setToolTip("Skip to end")
        controls.addWidget(self._skip_fwd_btn)

        controls.addSpacing(16)

        # Speed control
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        controls.addWidget(speed_label)

        self._speed_label = QLabel("1.0x")
        self._speed_label.setStyleSheet(f"""
            color: {COLORS.TEXT_PRIMARY};
            font-size: 9pt;
            font-weight: 600;
            min-width: 40px;
        """)
        controls.addWidget(self._speed_label)

        self._speed_down_btn = QPushButton("-")
        self._speed_down_btn.setStyleSheet(btn_style)
        self._speed_down_btn.setFixedWidth(28)
        controls.addWidget(self._speed_down_btn)

        self._speed_up_btn = QPushButton("+")
        self._speed_up_btn.setStyleSheet(btn_style)
        self._speed_up_btn.setFixedWidth(28)
        controls.addWidget(self._speed_up_btn)

        controls.addStretch()

        # Time display
        self._time_label = QLabel("0.0s / 0.0s")
        self._time_label.setStyleSheet(f"""
            color: {COLORS.TEXT_SECONDARY};
            font-size: 10pt;
            font-family: monospace;
        """)
        controls.addWidget(self._time_label)

        layout.addLayout(controls)

    def _connect_signals(self):
        """Connect button signals."""
        self._play_btn.clicked.connect(self._on_play)
        self._pause_btn.clicked.connect(self._on_pause)
        self._stop_btn.clicked.connect(self._on_stop)
        self._skip_back_btn.clicked.connect(lambda: self.seek.emit(0.0))
        self._skip_fwd_btn.clicked.connect(lambda: self.seek.emit(1.0))
        self._speed_down_btn.clicked.connect(self._decrease_speed)
        self._speed_up_btn.clicked.connect(self._increase_speed)
        self._waveform.cursor_moved.connect(self.seek.emit)

    def _on_play(self):
        """Handle play button."""
        self._state = PlaybackState.PLAYING
        self._update_button_states()
        self.play_clicked.emit()

    def _on_pause(self):
        """Handle pause button."""
        self._state = PlaybackState.PAUSED
        self._update_button_states()
        self.pause_clicked.emit()

    def _on_stop(self):
        """Handle stop button."""
        self._state = PlaybackState.STOPPED
        self._current_time = 0.0
        self._update_button_states()
        self._update_time_display()
        self.stop_clicked.emit()

    def _update_button_states(self):
        """Update button enabled states based on playback state."""
        is_playing = self._state == PlaybackState.PLAYING
        is_stopped = self._state == PlaybackState.STOPPED

        self._play_btn.setEnabled(not is_playing)
        self._pause_btn.setEnabled(is_playing)
        self._stop_btn.setEnabled(not is_stopped)

    def _increase_speed(self):
        """Increase playback speed."""
        if self._speed_index < len(self.SPEEDS) - 1:
            self._speed_index += 1
            self._update_speed_display()
            self.speed_changed.emit(self.SPEEDS[self._speed_index])

    def _decrease_speed(self):
        """Decrease playback speed."""
        if self._speed_index > 0:
            self._speed_index -= 1
            self._update_speed_display()
            self.speed_changed.emit(self.SPEEDS[self._speed_index])

    def _update_speed_display(self):
        """Update speed label."""
        speed = self.SPEEDS[self._speed_index]
        self._speed_label.setText(f"{speed}x")

    def _update_time_display(self):
        """Update time label."""
        self._time_label.setText(f"{self._current_time:.1f}s / {self._total_time:.1f}s")

    # Public API

    def set_time(self, current: float, total: float):
        """Update current and total time."""
        self._current_time = current
        self._total_time = total
        self._update_time_display()

        # Update waveform cursor
        if total > 0:
            self._waveform.set_cursor(current / total)

    def set_waveform_data(self, values: List[float]):
        """Set waveform visualization data."""
        self._waveform.set_values(values)

    def set_state(self, state: PlaybackState):
        """Set playback state."""
        self._state = state
        self._update_button_states()

    @property
    def speed(self) -> float:
        """Get current playback speed."""
        return self.SPEEDS[self._speed_index]

    @property
    def state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state

    def reset(self):
        """Reset timeline to initial state."""
        self._state = PlaybackState.STOPPED
        self._current_time = 0.0
        self._total_time = 0.0
        self._speed_index = 3
        self._waveform.clear()
        self._update_button_states()
        self._update_speed_display()
        self._update_time_display()
