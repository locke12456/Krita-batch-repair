"""Main repair plugin docker UI."""

from __future__ import annotations

from typing import Any

from .detector_model_manager import DetectorModelManager
from .repair_compat import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

try:
    from krita import DockWidget
except Exception:
    DockWidget = QWidget


class RepairDocker(DockWidget):
    """UI shell for detector residency and candidate workflow controls."""

    def __init__(self, parent: Any = None) -> None:
        try:
            super().__init__(parent)
        except TypeError:
            super().__init__()
        self.detector_manager = DetectorModelManager()
        self._status_label = QLabel("Detector: unloaded")
        self._mode_combo = QComboBox()
        self._load_button = QPushButton("Load Detector")
        self._unload_button = QPushButton("Unload Detector")
        self._detect_button = QPushButton("Detect")
        self._select_all_button = QPushButton("Select All")
        self._clear_selected_button = QPushButton("Clear Selected")
        self._row_scroll = QScrollArea()
        self._row_container = QWidget()
        self._row_layout = QVBoxLayout()
        self._candidate_rows: list[dict[str, Any]] = []
        self._build_ui()
        self._connect_signals()
        self._refresh_status()

    def canvasChanged(self, canvas: Any) -> None:
        """Krita docker callback kept intentionally lightweight."""
        return None

    def _build_ui(self) -> None:
        """Build the Phase 2 shell widgets."""
        if hasattr(self, "setWindowTitle"):
            self.setWindowTitle("Auto Detect Repair")

        for mode in ("all", "head", "censor"):
            self._mode_combo.addItem(mode)

        root = QWidget()
        layout = QVBoxLayout()
        root.setLayout(layout)

        detector_row = QHBoxLayout()
        detector_row.addWidget(self._load_button)
        detector_row.addWidget(self._unload_button)
        layout.addLayout(detector_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        mode_row.addWidget(self._mode_combo)
        mode_row.addWidget(self._detect_button)
        layout.addLayout(mode_row)

        selection_row = QHBoxLayout()
        selection_row.addWidget(self._select_all_button)
        selection_row.addWidget(self._clear_selected_button)
        layout.addLayout(selection_row)

        layout.addWidget(self._status_label)

        self._row_container.setLayout(self._row_layout)
        self._row_scroll.setWidget(self._row_container)
        self._row_scroll.setWidgetResizable(True)
        layout.addWidget(QLabel("Candidate Layers"))
        layout.addWidget(self._row_scroll)
        self._refresh_candidate_rows()

        if hasattr(self, "setWidget"):
            self.setWidget(root)
        elif hasattr(self, "setLayout"):
            self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Wire button callbacks."""
        self._load_button.clicked.connect(self._load_detector)
        self._unload_button.clicked.connect(self._unload_detector)
        self._detect_button.clicked.connect(self._detect_placeholder)
        self._select_all_button.clicked.connect(self._select_all_rows)
        self._clear_selected_button.clicked.connect(self._clear_selected_rows)
        self._mode_combo.currentTextChanged.connect(self._refresh_candidate_rows)

    def _current_mode(self) -> str:
        """Return the current detector mode filter."""
        text = self._mode_combo.currentText()
        return str(text or "all").strip().lower() or "all"

    def _load_detector(self) -> None:
        """Load or warm the detector backend for the selected mode."""
        try:
            self.detector_manager.load(self._current_mode())
            self._refresh_status()
        except Exception as exc:
            self._show_error(str(exc))

    def _unload_detector(self) -> None:
        """Unload detector backend references for the selected mode."""
        try:
            self.detector_manager.unload(self._current_mode())
            self._refresh_status()
        except Exception as exc:
            self._show_error(str(exc))

    def _detect_placeholder(self) -> None:
        """Phase 2 UI shell placeholder for the later detection service."""
        self._show_info("Detection service is implemented in the next phase.")

    def add_candidate_row(
        self,
        layer_id: str,
        layer_name: str,
        mode: str,
        active: bool = True,
        selected: bool = True,
    ) -> None:
        """Add a candidate row to the Phase 2 row list."""
        self._candidate_rows.append(
            {
                "layer_id": layer_id,
                "layer_name": layer_name,
                "mode": mode,
                "selected": bool(selected),
                "active": bool(active),
            }
        )
        self._refresh_candidate_rows()

    def selected_active_rows(self, filter_mode: str | None = None) -> list[dict[str, Any]]:
        """Return currently selected and active rows."""
        active_filter = filter_mode or self._mode_combo.currentText()
        return [
            row
            for row in self._candidate_rows
            if row.get("selected")
            and row.get("active")
            and (active_filter == "all" or row.get("mode") == active_filter)
        ]

    def _select_all_rows(self) -> None:
        """Select all rows in the current mode filter."""
        active_filter = self._mode_combo.currentText()
        for row in self._candidate_rows:
            if active_filter == "all" or row.get("mode") == active_filter:
                row["selected"] = True
        self._refresh_candidate_rows()

    def _clear_selected_rows(self) -> None:
        """Clear selected rows in the current mode filter."""
        active_filter = self._mode_combo.currentText()
        for row in self._candidate_rows:
            if active_filter == "all" or row.get("mode") == active_filter:
                row["selected"] = False
        self._refresh_candidate_rows()

    def _refresh_candidate_rows(self) -> None:
        """Render candidate rows with selected and active controls."""
        while self._row_layout.count():
            item = self._row_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        active_filter = self._mode_combo.currentText()
        visible_rows = [
            row for row in self._candidate_rows
            if active_filter == "all" or row.get("mode") == active_filter
        ]

        if not visible_rows:
            self._row_layout.addWidget(QLabel("No candidate rows yet."))
            return

        for row in visible_rows:
            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_widget.setLayout(row_layout)

            selected = QCheckBox()
            selected.setChecked(bool(row.get("selected")))
            selected.stateChanged.connect(
                lambda _state, target=row, widget=selected: self._set_row_selected(
                    target,
                    widget.isChecked(),
                )
            )

            active = QCheckBox("Active")
            active.setChecked(bool(row.get("active")))
            active.stateChanged.connect(
                lambda _state, target=row, widget=active: self._set_row_active(
                    target,
                    widget.isChecked(),
                )
            )

            label = QLabel(f"{row.get('mode', '')}: {row.get('layer_name', row.get('layer_id', ''))}")
            row_layout.addWidget(selected)
            row_layout.addWidget(active)
            row_layout.addWidget(label)
            self._row_layout.addWidget(row_widget)

    def _set_row_selected(self, row: dict[str, Any], selected: bool) -> None:
        """Update a row selected flag from the UI."""
        row["selected"] = bool(selected)

    def _set_row_active(self, row: dict[str, Any], active: bool) -> None:
        """Update a row active flag from the UI."""
        row["active"] = bool(active)

    def _refresh_status(self) -> None:
        """Refresh detector status text."""
        status = self.detector_manager.status()
        modes = ", ".join(status.loaded_modes) if status.loaded_modes else "none"
        self._status_label.setText(f"Detector: {status.state}; loaded modes: {modes}")

    def _show_info(self, message: str) -> None:
        """Display an informational message."""
        try:
            QMessageBox.information(self, "Auto Detect Repair", message)
        except Exception:
            print(message)

    def _show_error(self, message: str) -> None:
        """Display an error message."""
        try:
            QMessageBox.critical(self, "Auto Detect Repair", message)
        except Exception:
            print(message)