"""Main repair plugin docker UI."""

from __future__ import annotations

from typing import Any

from .bbox_generation_service import BBoxGenerationService
from .detection_service import DetectionOptions
from .detector_model_manager import DetectorModelManager
from .group_batch_detection_service import GroupBatchDetectionService, GroupDetectionReport
from .group_selection_model import GroupSelectionModel, RepairGroupRow
from .group_sync_source import GroupSyncSource
from .layer_metadata_service import LayerMetadataService
from .prompt_extraction_service import PromptExtractionService
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
    """UI shell for SyncRecord group batch detection."""

    def __init__(self, parent: Any = None) -> None:
        try:
            super().__init__(parent)
        except TypeError:
            super().__init__()

        self.detector_manager = DetectorModelManager()
        self.group_selection_model = GroupSelectionModel()
        self.metadata_service = LayerMetadataService()
        self.prompt_extraction_service = PromptExtractionService(
            metadata_service=self.metadata_service,
        )
        self.bbox_generation_service = BBoxGenerationService()
        self.group_batch_detection_service = GroupBatchDetectionService(
            self.detector_manager,
            self.metadata_service,
            self.prompt_extraction_service,
        )

        self._status_label = QLabel("Detector: unloaded")
        self._mode_combo = QComboBox()
        self._refresh_groups_button = QPushButton("Refresh Groups")
        self._load_button = QPushButton("Load Detector")
        self._unload_button = QPushButton("Unload Detector")
        self._detect_button = QPushButton("Batch Detect Selected Groups")
        self._image2tagger_checkbox = QCheckBox("Use image2tagger prompt")
        self._generation_checkbox = QCheckBox("Generate bbox repair")
        self._select_all_button = QPushButton("Select All Groups")
        self._clear_selected_button = QPushButton("Clear Groups")
        self._row_scroll = QScrollArea()
        self._row_container = QWidget()
        self._row_layout = QVBoxLayout()
        self._report_label = QLabel("Batch Report: no run yet.")

        self._build_ui()
        self._connect_signals()
        self._refresh_status()

    def canvasChanged(self, canvas: Any) -> None:
        """Krita docker callback kept intentionally lightweight."""
        return None

    def _build_ui(self) -> None:
        """Build the SyncRecord group batch UI."""
        if hasattr(self, "setWindowTitle"):
            self.setWindowTitle("Auto Detect Repair")

        for mode in ("all", "head", "censor"):
            self._mode_combo.addItem(mode)

        root = QWidget()
        layout = QVBoxLayout()
        root.setLayout(layout)

        layout.addWidget(self._refresh_groups_button)

        detector_row = QHBoxLayout()
        detector_row.addWidget(self._load_button)
        detector_row.addWidget(self._unload_button)
        layout.addLayout(detector_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        mode_row.addWidget(self._mode_combo)
        layout.addLayout(mode_row)

        selection_row = QHBoxLayout()
        selection_row.addWidget(self._select_all_button)
        selection_row.addWidget(self._clear_selected_button)
        layout.addLayout(selection_row)

        option_row = QHBoxLayout()
        option_row.addWidget(self._image2tagger_checkbox)
        option_row.addWidget(self._generation_checkbox)
        layout.addLayout(option_row)

        layout.addWidget(self._detect_button)
        layout.addWidget(self._status_label)

        self._row_container.setLayout(self._row_layout)
        self._row_scroll.setWidget(self._row_container)
        self._row_scroll.setWidgetResizable(True)
        layout.addWidget(QLabel("Group List"))
        layout.addWidget(self._row_scroll)
        layout.addWidget(self._report_label)
        self._refresh_group_rows()

        if hasattr(self, "setWidget"):
            self.setWidget(root)
        elif hasattr(self, "setLayout"):
            self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Wire button callbacks."""
        self._refresh_groups_button.clicked.connect(self._refresh_groups)
        self._load_button.clicked.connect(self._load_detector)
        self._unload_button.clicked.connect(self._unload_detector)
        self._detect_button.clicked.connect(self._batch_detect_selected_groups)
        self._select_all_button.clicked.connect(self._select_all_groups)
        self._clear_selected_button.clicked.connect(self._clear_groups)

    def _current_mode(self) -> str:
        """Return the current detector mode filter."""
        text = self._mode_combo.currentText()
        return str(text or "all").strip().lower() or "all"

    def _refresh_groups(self) -> None:
        """Load group-backed SyncRecord rows from the active document."""
        try:
            rows = GroupSyncSource().load_rows()
            self.group_selection_model.replace_rows(rows)
            self._refresh_group_rows()
            if not rows:
                self._show_info("No group-backed SyncRecord rows were found.")
        except Exception as exc:
            self._show_error(str(exc))

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

    def _batch_detect_selected_groups(self) -> None:
        """Run detection for selected active group rows only."""
        try:
            if self._generation_checkbox.isChecked():
                self._show_error(
                    "BBox-only generation is not implemented; refusing full-canvas redraw."
                )
                return

            rows = self.group_selection_model.selected_active_groups()
            reports = self.group_batch_detection_service.detect_rows(
                rows,
                self._current_mode(),
                DetectionOptions(),
                extract_prompts=self._image2tagger_checkbox.isChecked(),
            )
            self._refresh_group_rows()
            self._refresh_report(reports)
            if not rows:
                self._show_info("No selected resolved groups are available.")
            elif not reports:
                self._show_info("No detector results were created.")
        except Exception as exc:
            self._show_error(str(exc))

    def _select_all_groups(self) -> None:
        """Select all resolved group rows."""
        self.group_selection_model.select_all()
        self._refresh_group_rows()

    def _clear_groups(self) -> None:
        """Clear selected group rows."""
        self.group_selection_model.clear_selected()
        self._refresh_group_rows()

    def _refresh_group_rows(self) -> None:
        """Render SyncRecord group rows with selected and active controls."""
        while self._row_layout.count():
            item = self._row_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        rows = self.group_selection_model.rows
        if not rows:
            self._row_layout.addWidget(QLabel("No group rows yet. Click Refresh Groups."))
            return

        for row in rows:
            self._row_layout.addWidget(self._build_group_row_widget(row))

    def _build_group_row_widget(self, row: RepairGroupRow) -> QWidget:
        """Build one group row widget."""
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_widget.setLayout(row_layout)

        selected = QCheckBox()
        selected.setChecked(bool(row.selected))
        selected.setEnabled(row.is_resolved)
        selected.stateChanged.connect(
            lambda _state, target=row, widget=selected: self._set_group_selected(
                target,
                widget.isChecked(),
            )
        )

        active = QCheckBox("Active")
        active.setChecked(bool(row.active))
        active.stateChanged.connect(
            lambda _state, target=row, widget=active: self._set_group_active(
                target,
                widget.isChecked(),
            )
        )

        resolved = "resolved" if row.is_resolved else "unresolved"
        warnings = "; ".join(row.warnings)
        label = QLabel(
            f"#{row.sync_index} | {row.display_name} | {row.export_key} | "
            f"layers={len(row.layer_ids)} | {resolved} | created={row.detected_count}"
            + (f" | {warnings}" if warnings else "")
        )

        row_layout.addWidget(selected)
        row_layout.addWidget(active)
        row_layout.addWidget(label)
        return row_widget

    def _set_group_selected(self, row: RepairGroupRow, selected: bool) -> None:
        """Update a group row selected flag from the UI."""
        row.selected = bool(selected)

    def _set_group_active(self, row: RepairGroupRow, active: bool) -> None:
        """Update a group row active flag from the UI."""
        row.active = bool(active)

    def _refresh_report(self, reports: list[GroupDetectionReport]) -> None:
        """Render a traceable batch report."""
        if not reports:
            self._report_label.setText("Batch Report: no detector results.")
            return

        created = [report for report in reports if report.created_layer_id]
        failed = [report for report in reports if report.error]
        lines = [
            f"Batch Report: created={len(created)}, failed={len(failed)}, total={len(reports)}"
        ]

        for report in reports[:12]:
            bbox = report.bbox or {}
            bbox_text = (
                f"{bbox.get('x', '?')},{bbox.get('y', '?')},"
                f"{bbox.get('width', '?')}x{bbox.get('height', '?')}"
            )
            if report.error:
                lines.append(
                    f"[x] {report.group_name or report.export_key} | "
                    f"{report.source_layer_name} | {bbox_text} | {report.error}"
                )
            else:
                lines.append(
                    f"[+] {report.group_name or report.export_key} | "
                    f"{report.source_layer_name} | {bbox_text} | "
                    f"{report.created_layer_name or report.created_layer_id}"
                )

        remaining = len(reports) - 12
        if remaining > 0:
            lines.append(f"... {remaining} more result(s)")

        self._report_label.setText("\n".join(lines))

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
