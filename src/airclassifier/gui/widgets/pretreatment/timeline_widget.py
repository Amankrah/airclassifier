"""
Pretreatment Timeline Widget
============================

Simulation progress bar with playback-style controls
and RF power waveform visualization.
"""

from __future__ import annotations

from typing import Optional, List

from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient, QPainterPath
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout,
    QLabel, QWidget, QSlider, QPushButton,
    QSizePolicy,
)

from ...theme import COLORS


class WaveformDisplay(QFrame):
    """Mini RF power waveform visualization."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._values: List[float] = []
        self._max_points = 200
        self._max_value = 30.0  # Default max power (kW)
        self._current_index = 0

        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def add_value(self, value: float):
        """Add a new power value to the waveform."""
        self._values.append(value)
        if len(self._values) > self._max_points:
            self._values.pop(0)
        self._max_value = max(self._max_value, value * 1.2)
        self.update()

    def set_position(self, index: int):
        """Set the current playback position."""
        self._current_index = min(index, len(self._values) - 1)
        self.update()

    def clear(self):
        """Clear all values."""
        self._values.clear()
        self._current_index = 0
        self._max_value = 30.0
        self.update()

    def paintEvent(self, event):
        if not self._values:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background gradient
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0, QColor(COLORS.BG_DARKEST))
        bg_grad.setColorAt(1, QColor(COLORS.BG_DARK))
        painter.fillRect(0, 0, w, h, bg_grad)

        # Draw waveform
        n = len(self._values)
        if n < 2:
            return

        x_step = w / max(n - 1, 1)
        y_scale = (h - 10) / max(self._max_value, 1)

        # Build path for fill
        path = QPainterPath()
        path.moveTo(0, h)

        for i, val in enumerate(self._values):
            x = i * x_step
            y = h - 5 - val * y_scale
            if i == 0:
                path.lineTo(x, y)
            else:
                path.lineTo(x, y)

        path.lineTo((n - 1) * x_step, h)
        path.closeSubpath()

        # Fill gradient
        fill_grad = QLinearGradient(0, 0, 0, h)
        fill_grad.setColorAt(0, QColor(COLORS.PRETREAT_PRIMARY + "80"))
        fill_grad.setColorAt(1, QColor(COLORS.PRETREAT_PRIMARY + "20"))
        painter.fillPath(path, fill_grad)

        # Draw line
        pen = QPen(QColor(COLORS.PRETREAT_PRIMARY))
        pen.setWidth(2)
        painter.setPen(pen)

        for i in range(1, n):
            x1 = (i - 1) * x_step
            y1 = h - 5 - self._values[i - 1] * y_scale
            x2 = i * x_step
            y2 = h - 5 - self._values[i] * y_scale
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw position marker
        if 0 <= self._current_index < n:
            marker_x = self._current_index * x_step
            pen = QPen(QColor(COLORS.TEXT_PRIMARY))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(int(marker_x), 0, int(marker_x), h)


class PretreatmentTimelineWidget(QFrame):
    """Timeline widget with progress bar and playback controls.

    Features:
    - Progress bar showing simulation progress
    - Play/pause/stop controls (for future replay feature)
    - RF power waveform preview
    - Time labels (elapsed / total)

    Signals:
        play_clicked(): Emitted when play button is clicked
        pause_clicked(): Emitted when pause button is clicked
        stop_clicked(): Emitted when stop button is clicked
        seek_requested(float): Emitted with position (0-1) when seeking
    """

    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    seek_requested = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._duration = 0.0
        self._current_time = 0.0
        self._is_playing = False

        self._setup_style()
        self._setup_ui()

    def _setup_style(self):
        self.setObjectName("pretreatTimeline")
        self.setStyleSheet(f"""
            QFrame#pretreatTimeline {{
                background: {COLORS.BG_ELEVATED};
                border-top: 1px solid {COLORS.BORDER};
            }}
        """)
        self.setFixedHeight(100)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(6)

        # Waveform display
        self._waveform = WaveformDisplay()
        layout.addWidget(self._waveform)

        # Progress slider
        slider_row = QHBoxLayout()
        slider_row.setSpacing(12)

        self._time_label = QLabel("0:00")
        self._time_label.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
            min-width: 50px;
        """)
        slider_row.addWidget(self._time_label)

        self._progress_slider = QSlider(Qt.Orientation.Horizontal)
        self._progress_slider.setRange(0, 1000)
        self._progress_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 8px;
                background: {COLORS.BG_DARKEST};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS.PRETREAT_PRIMARY};
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLORS.ACCENT_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS.PRETREAT_PRIMARY}, stop:1 {COLORS.PRETREAT_SECONDARY});
                border-radius: 4px;
            }}
        """)
        self._progress_slider.sliderMoved.connect(self._on_seek)
        slider_row.addWidget(self._progress_slider, 1)

        self._duration_label = QLabel("0:00")
        self._duration_label.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_MUTED};
            min-width: 50px;
        """)
        self._duration_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        slider_row.addWidget(self._duration_label)

        layout.addLayout(slider_row)

        # Control buttons
        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)

        btn_style = f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER};
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 10pt;
                font-weight: 600;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                border-color: {COLORS.ACCENT};
            }}
            QPushButton:disabled {{
                color: {COLORS.TEXT_DISABLED};
            }}
        """

        self._stop_btn = QPushButton("\u25a0 Stop")
        self._stop_btn.setStyleSheet(btn_style)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        controls_row.addWidget(self._stop_btn)

        self._play_btn = QPushButton("\u25b6 Play")
        self._play_btn.setStyleSheet(btn_style)
        self._play_btn.clicked.connect(self._on_play_pause)
        controls_row.addWidget(self._play_btn)

        controls_row.addStretch()

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"""
            font-size: 9pt;
            color: {COLORS.TEXT_MUTED};
        """)
        controls_row.addWidget(self._status_label)

        controls_row.addStretch()

        # Percentage label
        self._pct_label = QLabel("0%")
        self._pct_label.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: 700;
            color: {COLORS.PRETREAT_PRIMARY};
        """)
        controls_row.addWidget(self._pct_label)

        layout.addLayout(controls_row)

    def _on_play_pause(self):
        if self._is_playing:
            self._is_playing = False
            self._play_btn.setText("\u25b6 Play")
            self.pause_clicked.emit()
        else:
            self._is_playing = True
            self._play_btn.setText("\u23f8 Pause")
            self.play_clicked.emit()

    def _on_seek(self, value: int):
        position = value / 1000.0
        self.seek_requested.emit(position)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as M:SS or H:MM:SS."""
        seconds = max(0, seconds)
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"

    def set_duration(self, duration_s: float):
        """Set the total simulation duration."""
        self._duration = duration_s
        self._duration_label.setText(self._format_time(duration_s))

    def set_progress(self, current_time: float, power_kw: Optional[float] = None):
        """Update the current progress.

        Args:
            current_time: Current simulation time in seconds
            power_kw: Optional RF power value to add to waveform
        """
        self._current_time = current_time

        # Update slider position
        if self._duration > 0:
            position = int(1000 * current_time / self._duration)
            self._progress_slider.blockSignals(True)
            self._progress_slider.setValue(position)
            self._progress_slider.blockSignals(False)

            pct = int(100 * current_time / self._duration)
            self._pct_label.setText(f"{pct}%")
        else:
            self._pct_label.setText("0%")

        # Update time label
        self._time_label.setText(self._format_time(current_time))

        # Add to waveform
        if power_kw is not None:
            self._waveform.add_value(power_kw)
            self._waveform.set_position(len(self._waveform._values) - 1)

    def set_status(self, status: str, color: Optional[str] = None):
        """Set the status label text."""
        self._status_label.setText(status)
        if color:
            self._status_label.setStyleSheet(f"font-size: 9pt; color: {color};")

    def set_running(self, running: bool):
        """Update UI state based on running status."""
        self._is_playing = running
        if running:
            self._play_btn.setText("\u23f8 Pause")
            self.set_status("Running...", COLORS.WARNING)
        else:
            self._play_btn.setText("\u25b6 Play")

    def set_complete(self, success: bool = True):
        """Mark simulation as complete."""
        self._is_playing = False
        self._play_btn.setText("\u25b6 Play")
        if success:
            self.set_status("Complete", COLORS.SUCCESS)
            self._pct_label.setText("100%")
        else:
            self.set_status("Stopped", COLORS.DANGER)

    def clear(self):
        """Reset the timeline."""
        self._duration = 0.0
        self._current_time = 0.0
        self._is_playing = False
        self._progress_slider.setValue(0)
        self._time_label.setText("0:00")
        self._duration_label.setText("0:00")
        self._pct_label.setText("0%")
        self._play_btn.setText("\u25b6 Play")
        self._waveform.clear()
        self.set_status("Ready", COLORS.TEXT_MUTED)
