"""Main repair plugin docker UI."""

from __future__ import annotations

from typing import Any

from .bbox_generation_service import BBoxGenerationService
from .detection_service import DetectionOptions
from .detector_model_manager import DetectorModelManager
from .group_batch_detection_service import GroupBatchDetectionService, GroupDetectionReport
from .group_refine_service import GroupRefineService
from .group_selection_model import GroupSelectionModel, RepairGroupRow
from .group_sync_source import GroupSyncSource
from .layer_metadata_service import LayerMetadataService
from .prompt_extraction_service import PromptExtractionService
from .prompt_extraction_worker import PromptExtractionProgress, PromptExtractionWorker
from .repair_result_model import RepairResultSelectionModel, RepairResultRow
from .repair_state_store import RepairStateRecord, RepairStateStore
from .repair_compat import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    active_krita_document,
    all_krita_nodes,
    delete_layer,
    find_krita_node_by_id,
    merge_layer_into_target,
    QVBoxLayout,
    QWidget,
    set_layer_visible,
)
from .repair_log_docker import RepairLogDocker

try:
    from krita import DockWidget
except Exception:
    DockWidget = QWidget


PROMPT_TYPE_OPTIONS = (
    ("All", ""),
    ("Head", "detailed head repair, natural face structure"),
    ("Penis", "anatomically consistent penis repair"),
    ("Pussy", "anatomically consistent pussy repair"),
    ("Censor", "remove censorship artifact, restore natural detail"),
    ("Other", "localized repair, coherent texture and lighting"),
)


def prompt_type_prompt(prompt_type: str) -> str:
    """Return the committed prompt fragment for a prompt type label."""
    for label, fragment in PROMPT_TYPE_OPTIONS:
        if label == prompt_type:
            return fragment
    return ""


class RepairDocker(DockWidget):
    """UI shell for SyncRecord group batch detection."""

    def __init__(self, parent: Any = None) -> None:
        try:
            super().__init__(parent)
        except TypeError:
            super().__init__()

        self.detector_manager = DetectorModelManager()
        self.group_selection_model = GroupSelectionModel()
        self.result_selection_model = RepairResultSelectionModel()
        self.metadata_service = LayerMetadataService()
        self.repair_state_store: RepairStateStore | None = None
        self.prompt_worker: PromptExtractionWorker | None = None
        self.prompt_extraction_service = PromptExtractionService(
            metadata_service=self.metadata_service,
        )
        self.bbox_generation_service = BBoxGenerationService(
            metadata_service=self.metadata_service,
            on_row_finished=self._on_generation_row_finished,
        )
        self.group_batch_detection_service = GroupBatchDetectionService(
            self.detector_manager,
            self.metadata_service,
            self.prompt_extraction_service,
            self.result_selection_model,
        )
        self.group_refine_service = GroupRefineService(
            bbox_generation_service=self.bbox_generation_service,
            metadata_service=self.metadata_service,
            repair_state_store=self.repair_state_store,
        )

        self._log_docker = RepairLogDocker()

        self._status_label = QLabel("Detector: unloaded")
        self._mode_combo = QComboBox()
        self._censor_filter_combo = QComboBox()
        self._censor_filter_label = QLabel("Censor filter")
        self._refresh_groups_button = QPushButton("Refresh Groups")
        self._load_button = QPushButton("Load Detector")
        self._unload_button = QPushButton("Unload Detector")
        self._detect_button = QPushButton("Batch Detect Selected Groups")
        self._refine_groups_button = QPushButton("Refine Selected Groups")
        self._sync_all_button = QPushButton("Sync All")
        self._image2tagger_checkbox = QCheckBox("Use image2tagger prompt")
        self._image2tagger_threshold_input = QLineEdit("0.8")
        self._generation_checkbox = QCheckBox("Generate bbox repair")
        self._force_rect_checkbox = QCheckBox("Force rect crop")
        self._rect_width_input = QLineEdit("260")
        self._rect_height_input = QLineEdit("340")
        self._clamp_rect_checkbox = QCheckBox("Clamp to source bounds")
        self._clamp_rect_checkbox.setChecked(True)
        self._extract_tags_button = QPushButton("Extract Tags for Selected Results")
        self._cancel_tags_button = QPushButton("Cancel Tag Extraction")
        self._select_all_results_button = QPushButton("Select All Results")
        self._clear_results_button = QPushButton("Clear Results")
        self._generate_results_button = QPushButton("Generate Selected Results")
        self._attach_mask_checkbox = QCheckBox("Attach transparency mask")
        self._attach_mask_checkbox.setChecked(True)
        self._batch_merge_button = QPushButton("Batch Remove Selected Results")
        self._result_filter_checkbox = QCheckBox("filter")
        self._result_filter_prompt_combo = QComboBox()
        self._result_visible_checkbox = QCheckBox("visible")
        self._result_invisible_checkbox = QCheckBox("invisible")
        self._apply_result_filter_button = QPushButton("apply")
        self._committed_result_filter_enabled = False
        self._committed_result_filter_prompt_type = ""
        self._prompt_progress_label = QLabel("Prompt extraction: 0 / 0")
        self._select_all_button = QPushButton("Select All Groups")
        self._clear_selected_button = QPushButton("Clear Groups")
        self._row_scroll = QScrollArea()
        self._row_container = QWidget()
        self._row_layout = QVBoxLayout()
        self._result_scroll = QScrollArea()
        self._result_container = QWidget()
        self._result_layout = QVBoxLayout()
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
        for prompt_type, _fragment in PROMPT_TYPE_OPTIONS:
            self._result_filter_prompt_combo.addItem(prompt_type)

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

        for censor_label in ("All", "penis", "pussy"):
            self._censor_filter_combo.addItem(censor_label)
        censor_filter_row = QHBoxLayout()
        censor_filter_row.addWidget(self._censor_filter_label)
        censor_filter_row.addWidget(self._censor_filter_combo)
        layout.addLayout(censor_filter_row)
        self._refresh_detector_filter_visibility()

        selection_row = QHBoxLayout()
        selection_row.addWidget(self._select_all_button)
        selection_row.addWidget(self._clear_selected_button)
        layout.addLayout(selection_row)

        option_row = QHBoxLayout()
        option_row.addWidget(self._image2tagger_checkbox)
        option_row.addWidget(QLabel("Threshold"))
        option_row.addWidget(self._image2tagger_threshold_input)
        option_row.addWidget(self._generation_checkbox)
        layout.addLayout(option_row)

        rect_row = QHBoxLayout()
        rect_row.addWidget(self._force_rect_checkbox)
        rect_row.addWidget(QLabel("Width"))
        rect_row.addWidget(self._rect_width_input)
        rect_row.addWidget(QLabel("Height"))
        rect_row.addWidget(self._rect_height_input)
        rect_row.addWidget(self._clamp_rect_checkbox)
        layout.addLayout(rect_row)

        refine_sync_row = QHBoxLayout()
        refine_sync_row.addWidget(self._refine_groups_button)
        refine_sync_row.addWidget(self._sync_all_button)
        layout.addLayout(refine_sync_row)
        layout.addWidget(self._detect_button)
        layout.addWidget(self._status_label)

        self._row_container.setLayout(self._row_layout)
        self._row_scroll.setWidget(self._row_container)
        self._row_scroll.setWidgetResizable(True)
        layout.addWidget(QLabel("Group List"))
        layout.addWidget(self._row_scroll)

        result_action_row = QHBoxLayout()
        result_action_row.addWidget(self._select_all_results_button)
        result_action_row.addWidget(self._clear_results_button)
        layout.addLayout(result_action_row)

        tag_row = QHBoxLayout()
        tag_row.addWidget(self._extract_tags_button)
        tag_row.addWidget(self._cancel_tags_button)
        layout.addLayout(tag_row)

        self._result_container.setLayout(self._result_layout)
        self._result_scroll.setWidget(self._result_container)
        self._result_scroll.setWidgetResizable(True)
        layout.addWidget(QLabel("Detection Results"))

        result_filter_row = QHBoxLayout()
        result_filter_row.addWidget(self._result_filter_checkbox)
        result_filter_row.addWidget(self._result_filter_prompt_combo)
        result_filter_row.addWidget(self._result_visible_checkbox)
        result_filter_row.addWidget(self._result_invisible_checkbox)
        result_filter_row.addWidget(self._apply_result_filter_button)
        layout.addLayout(result_filter_row)

        layout.addWidget(self._result_scroll)
        layout.addWidget(self._attach_mask_checkbox)
        layout.addWidget(self._generate_results_button)
        layout.addWidget(self._batch_merge_button)

        self._refresh_group_rows()
        self._refresh_result_rows()

        if hasattr(self, "setWidget"):
            self.setWidget(root)
        elif hasattr(self, "setLayout"):
            self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Wire button callbacks."""
        self._refresh_groups_button.clicked.connect(self._refresh_groups)
        self._load_button.clicked.connect(self._load_detector)
        self._unload_button.clicked.connect(self._unload_detector)
        self._mode_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_detector_filter_visibility()
        )
        self._detect_button.clicked.connect(self._batch_detect_selected_groups)
        self._refine_groups_button.clicked.connect(self._refine_selected_groups)
        self._sync_all_button.clicked.connect(self._sync_all_refined_groups)
        self._select_all_button.clicked.connect(self._select_all_groups)
        self._clear_selected_button.clicked.connect(self._clear_groups)
        self._select_all_results_button.clicked.connect(self._select_all_results)
        self._clear_results_button.clicked.connect(self._clear_results)
        self._extract_tags_button.clicked.connect(self._extract_tags_for_selected_results)
        self._cancel_tags_button.clicked.connect(self._cancel_tag_extraction)
        self._generate_results_button.clicked.connect(self._generate_selected_results)
        self._batch_merge_button.clicked.connect(self._batch_remove_selected_results)
        self._result_visible_checkbox.stateChanged.connect(
            lambda _state: self._set_toolbar_visibility_exclusive("visible")
        )
        self._result_invisible_checkbox.stateChanged.connect(
            lambda _state: self._set_toolbar_visibility_exclusive("invisible")
        )
        self._apply_result_filter_button.clicked.connect(
            self._apply_result_list_visibility_filter
        )

    def _current_mode(self) -> str:
        """Return the current detector mode filter."""
        text = self._mode_combo.currentText()
        return str(text or "all").strip().lower() or "all"

    def _detection_options(self) -> DetectionOptions:
        """Return detection options from the current UI controls."""
        filter_label: str | None = None
        if self._current_mode() == "censor":
            raw = str(self._censor_filter_combo.currentText() or "").strip()
            if raw and raw != "All":
                filter_label = raw
        return DetectionOptions(
            filter_label=filter_label,
            force_rect_crop=self._force_rect_checkbox.isChecked(),
            rect_width=max(1, int(self._rect_width_input.text() or "260")),
            rect_height=max(1, int(self._rect_height_input.text() or "340")),
            clamp_rect_to_source_bounds=self._clamp_rect_checkbox.isChecked(),
        )

    def _current_repair_state_store(self) -> RepairStateStore:
        """Return a RepairStateStore for the active Krita document."""
        document_ref = active_krita_document()
        if document_ref is None:
            raise RuntimeError("No active Krita document.")
        if self.repair_state_store is None:
            self.repair_state_store = RepairStateStore(document_ref)
        return self.repair_state_store

    def _refresh_groups(self) -> None:
        """Load group-backed SyncRecord rows from the active document."""
        try:
            rows = GroupSyncSource(
                repair_state_store=self._current_repair_state_store(),
            ).load_rows()
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
            rows = self.group_selection_model.selected_active_groups()
            reports = self.group_batch_detection_service.detect_rows(
                rows,
                self._current_mode(),
                self._detection_options(),
                extract_prompts=False,
            )
            self._refresh_group_rows()
            self._refresh_result_rows()
            self._refresh_report(reports)
            if self._image2tagger_checkbox.isChecked():
                self._start_prompt_worker([report.result_row for report in reports if report.result_row])
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
        refine_tag = row.refine_reason
        label = QLabel(
            f"#{row.sync_index} | {row.display_name} | {row.export_key} | "
            f"layers={len(row.layer_ids)} | {resolved} | created={row.detected_count}"
            + (f" | refine={refine_tag}" if refine_tag else "")
            + (f" | {warnings}" if warnings else "")
        )

        sync_button = QPushButton("Sync")
        sync_button.setEnabled(row.is_resolved)
        sync_button.clicked.connect(
            lambda _checked=False, target=row: self._sync_group_row(target)
        )

        row_layout.addWidget(selected)
        row_layout.addWidget(active)
        row_layout.addWidget(label)
        row_layout.addWidget(sync_button)
        return row_widget

    def _set_group_selected(self, row: RepairGroupRow, selected: bool) -> None:
        """Update a group row selected flag from the UI."""
        row.selected = bool(selected)

    def _set_group_active(self, row: RepairGroupRow, active: bool) -> None:
        """Update a group row active flag from the UI."""
        row.active = bool(active)

    def _select_all_results(self) -> None:
        """Select all active detection result rows."""
        self.result_selection_model.select_all()
        self._refresh_result_rows()

    def _clear_results(self) -> None:
        """Clear selected detection result rows."""
        self.result_selection_model.clear_selected()
        self._refresh_result_rows()

    def _refresh_result_rows(self) -> None:
        """Render detection result rows with selected and active controls."""
        while self._result_layout.count():
            item = self._result_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        rows = self._visible_result_rows_for_current_filter()
        if not rows:
            self._result_layout.addWidget(QLabel("No detection results yet."))
            return

        for row in rows:
            self._result_layout.addWidget(self._build_result_row_widget(row))

    def _build_result_row_widget(self, row: RepairResultRow) -> QWidget:
        """Build one detection result row widget."""
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_widget.setLayout(row_layout)

        selected = QCheckBox()
        selected.setChecked(bool(row.selected))
        selected.stateChanged.connect(
            lambda _state, target=row, widget=selected: self._set_result_selected(
                target,
                widget.isChecked(),
            )
        )

        visible = QCheckBox("Visible")
        visible.setChecked(bool(getattr(row, "visible", True)))
        visible.stateChanged.connect(
            lambda _state, target=row, widget=visible: self._set_result_visible(
                target,
                widget.isChecked(),
            )
        )

        active = QCheckBox("Active")
        active.setChecked(bool(row.active))
        active.stateChanged.connect(
            lambda _state, target=row, widget=active: self._set_result_active(
                target,
                widget.isChecked(),
            )
        )

        label = QLabel(self._result_row_label(row))
        tag_label = QLabel(f"Tag: {row.effective_prompt_type() or 'Unclassified'}")

        merge_button = QPushButton("Merge")
        merge_button.setEnabled(bool(row.generation_result_layer_id))
        merge_button.clicked.connect(
            lambda _checked=False, target=row: self._merge_result_generation_layer(target)
        )

        delete_button = QPushButton("X")
        delete_button.clicked.connect(
            lambda _checked=False, target=row: self._delete_result_row(target)
        )

        row_layout.addWidget(selected)
        row_layout.addWidget(visible)
        row_layout.addWidget(active)
        row_layout.addWidget(label)
        row_layout.addWidget(tag_label)
        row_layout.addWidget(merge_button)
        row_layout.addWidget(delete_button)
        return row_widget

    def _set_result_selected(self, row: RepairResultRow, selected: bool) -> None:
        """Update a detection result row selected flag from the UI."""
        row.selected = bool(selected)

    def _set_result_active(self, row: RepairResultRow, active: bool) -> None:
        """Update a detection result row active flag from the UI."""
        row.active = bool(active)

    def _set_result_visible(self, row: RepairResultRow, visible: bool) -> None:
        """Set row visibility through the adapter helper."""
        try:
            self._apply_row_visibility(row, visible)
        except Exception as exc:
            self._show_error(str(exc))
        self._refresh_result_rows()

    def _apply_row_visibility(self, row: RepairResultRow, visible: bool) -> None:
        """Apply visibility to the primary layer target for one row."""
        target = self._resolve_row_visibility_target(row)
        if target is None:
            raise RuntimeError("No layer target exists for this result row.")
        set_layer_visible(target, bool(visible))
        row.visible = bool(visible)
        self._attach_row_ui_metadata(row, "result")

    def _apply_result_prompt_type(self, row: RepairResultRow, prompt_type: str) -> None:
        """Commit the prompt type dropdown to the row model and metadata."""
        prompt_type = str(prompt_type or "").strip()
        if not prompt_type or prompt_type == "All":
            row.prompt_type = ""
            row.prompt_type_prompt = ""
            row.prompt_type_applied = False
        else:
            row.prompt_type = prompt_type
            row.prompt_type_prompt = prompt_type_prompt(prompt_type)
            row.prompt_type_applied = True
        self._attach_row_ui_metadata(row, "prompt")
        self._refresh_result_rows()

    def _set_toolbar_visibility_exclusive(self, changed: str) -> None:
        """Keep visible and invisible toolbar operations mutually exclusive."""
        if changed == "visible" and self._result_visible_checkbox.isChecked():
            self._result_invisible_checkbox.setChecked(False)
        elif changed == "invisible" and self._result_invisible_checkbox.isChecked():
            self._result_visible_checkbox.setChecked(False)

    def _apply_result_list_visibility_filter(self) -> None:
        """Commit the toolbar filter and optional layer visibility operation."""
        filter_enabled = bool(self._result_filter_checkbox.isChecked())
        prompt_type = str(self._result_filter_prompt_combo.currentText() or "").strip()
        apply_visible = bool(self._result_visible_checkbox.isChecked())
        apply_invisible = bool(self._result_invisible_checkbox.isChecked())

        if apply_visible and apply_invisible:
            self._show_error("Visible and invisible cannot both be selected.")
            return

        self._committed_result_filter_enabled = filter_enabled
        self._committed_result_filter_prompt_type = prompt_type

        errors: list[str] = []
        if apply_visible or apply_invisible:
            target_visible = bool(apply_visible)
            for row in self.result_selection_model.visibility_target_rows(
                prompt_type=prompt_type,
                filter_enabled=filter_enabled,
            ):
                try:
                    self._apply_row_visibility(row, target_visible)
                except Exception as exc:
                    errors.append(f"{row.display_name}: {exc}")

        self._refresh_result_rows()
        if errors:
            self._show_error("\n".join(errors))

    def _visible_result_rows_for_current_filter(self) -> list[RepairResultRow]:
        """Return rows visible in the Docker result list."""
        return self.result_selection_model.visibility_target_rows(
            prompt_type=self._committed_result_filter_prompt_type,
            filter_enabled=self._committed_result_filter_enabled,
        )

    def _resolve_row_visibility_target(self, row: RepairResultRow) -> Any | None:
        """Prefer generated layer, then fallback to detection crop layer."""
        document_ref = getattr(row.source_layer, "document_ref", None)
        if row.generation_result_layer_id and document_ref is not None:
            generated = find_krita_node_by_id(document_ref, row.generation_result_layer_id)
            if generated is not None:
                return generated
        if row.created_layer is not None:
            return row.created_layer
        return None

    def _delete_result_row(self, row: RepairResultRow) -> None:
        """Delete this row's Krita layers, then remove the row from the model."""
        if row.generation_status == "running":
            self._show_info("Cannot delete a result row while generation is running.")
            return

        try:
            for target in self._delete_targets_for_result_row(row):
                delete_layer(target)
            self.result_selection_model.remove_result(row.result_id)
        except Exception as exc:
            self._show_error(str(exc))
        self._refresh_result_rows()

    def _delete_targets_for_result_row(self, row: RepairResultRow) -> list[Any]:
        """Return unique layer targets owned by this result row.

        Priority:
        1. merged layer, when merge already produced a target id
        2. generated repair layer
        3. detection / repair row layer

        Targets are deduplicated by Krita node id so merged rows do not try to
        delete the same layer twice.
        """
        targets: list[Any] = []
        seen_ids: set[str] = set()

        def add_target(layer: Any | None) -> None:
            if layer is None:
                return
            node = getattr(layer, "node", layer)
            parent_node = getattr(node, "parentNode", None)
            if not callable(parent_node) or parent_node() is None:
                # Skip stale wrappers after mergeDown(); they no longer belong
                # to the document tree and cannot be deleted safely.
                return

            layer_id = str(getattr(layer, "id_string", "") or "")
            if not layer_id:
                unique_id = getattr(node, "uniqueId", None)
                if callable(unique_id):
                    layer_id = str(unique_id().toString())
            if layer_id and layer_id in seen_ids:
                return
            if layer_id:
                seen_ids.add(layer_id)
            targets.append(layer)

        document_ref = getattr(row.source_layer, "document_ref", None)
        is_merged = getattr(row, "merge_status", "") == "merged"

        merged_layer_id = str(getattr(row, "merged_layer_id", "") or "")
        if merged_layer_id and document_ref is not None:
            add_target(find_krita_node_by_id(document_ref, merged_layer_id))

        if is_merged and not targets:
            # Backward-compatible fallback for rows merged before the live-id fix:
            # the saved id may be stale, but the layer name is often still exact.
            group_node = getattr(row.group_layer, "node", row.group_layer)
            target_names = {
                str(getattr(row, "merged_layer_name", "") or ""),
                str(getattr(row, "display_name", "") or ""),
            }
            target_names.discard("")
            try:
                children = list(group_node.childNodes() or [])
            except Exception:
                children = []
            for child in children:
                child_name = str(getattr(child, "name", lambda: "")() or "")
                if child_name in target_names:
                    add_target(child)
                    if targets:
                        break

        if not is_merged:
            generated_layer_id = str(getattr(row, "generation_result_layer_id", "") or "")
            if generated_layer_id and document_ref is not None:
                add_target(find_krita_node_by_id(document_ref, generated_layer_id))

        add_target(row.created_layer)

        if not targets:
            raise RuntimeError("No live Krita layer target exists for this result row.")
        return targets

    def _merge_result_generation_layer(self, row: RepairResultRow) -> None:
        """Merge the generated layer into the correct row-specific target layer."""
        try:
            if not row.generation_result_layer_id:
                raise RuntimeError("No generated layer to merge.")
            if row.source_layer is None:
                raise RuntimeError("Source layer is required for merge.")
            document_ref = getattr(row.source_layer, "document_ref", None)
            if document_ref is None:
                raise RuntimeError("Source layer document_ref is required for merge.")

            generated = find_krita_node_by_id(document_ref, row.generation_result_layer_id)
            if generated is None:
                raise RuntimeError("Generated layer could not be resolved.")

            # Correct merge target is the row-specific repair/detection target layer.
            # Fall back to the original source layer only when no created row layer exists.
            target_layer = row.created_layer or row.source_layer
            if target_layer is None:
                raise RuntimeError("No row-specific target layer exists for merge.")

            group_node = getattr(row.group_layer, "node", row.group_layer)
            generated_node = getattr(generated, "node", generated)
            target_node = getattr(target_layer, "node", target_layer)
            if group_node is None:
                raise RuntimeError("Group layer is required for merge.")
            if generated_node.parentNode() != group_node or target_node.parentNode() != group_node:
                raise RuntimeError("Generated and target layers must share the result group parent.")

            merged = merge_layer_into_target(document_ref, generated, target_layer)
            merged_layer_id = str(getattr(merged, "id_string", "") or "")
            if not merged_layer_id:
                raise RuntimeError("Merged layer id could not be resolved after merge.")

            row.merge_status = "merged"
            row.merge_error = ""
            row.merged_layer_id = merged_layer_id
            row.merged_layer_name = str(getattr(merged, "name", "") or getattr(target_layer, "name", "") or "")
            row.created_layer = merged

            # The generated layer was consumed by mergeDown(); keeping this id
            # makes [X] try to delete a stale / detached layer after merge.
            row.generation_result_layer_id = ""
            row.generation_result_layer_name = ""
            self._attach_row_ui_metadata(row, "result")
        except Exception as exc:
            row.merge_status = "failed"
            row.merge_error = str(exc)
            self._attach_row_ui_metadata(row, "result")
            self._show_error(row.merge_error)
        self._refresh_result_rows()

    def _batch_remove_selected_results(self) -> None:
        """Batch remove all eligible selected result rows (delete layers + remove from model)."""
        rows = self.result_selection_model.selected_active_results()
        eligible = [
            row for row in rows
            if row.generation_status == "done"
        ]
        if not eligible:
            self._show_info("No selected results have completed generation for cleanup.")
            return

        success_count = 0
        failed_count = 0
        for row in eligible:
            try:
                # Delete detection/crop cache layer, keep generated repair layer
                if row.created_layer is not None and row.created_layer is not row.source_layer:
                    try:
                        delete_layer(row.created_layer)
                    except Exception:
                        pass  # Already removed or stale
                self.result_selection_model.remove_result(row.result_id)
                success_count += 1
            except Exception as exc:
                failed_count += 1
                print(f"[RepairDocker] WARNING: batch remove failed for {row.display_name}: {exc}")

        self._refresh_result_rows()
        self._log_docker.set_report(
            f"Batch Remove Report: success={success_count}, "
            f"failed={failed_count}, total={len(eligible)}"
        )

    def _result_row_label(self, row: RepairResultRow) -> str:
        """Return compact row text so action widgets have horizontal room."""
        prompt = row.effective_prompt_type() or "unclassified"
        merge = getattr(row, "merge_status", "not_started")
        return (
            f"{row.display_name} | type={prompt} | tag={row.prompt_status} | "
            f"gen={row.generation_status} | merge={merge}"
        )

    def _set_combo_text(self, combo: QComboBox, text: str) -> None:
        """Best-effort select a QComboBox item by display text."""
        find_text = getattr(combo, "findText", None)
        set_index = getattr(combo, "setCurrentIndex", None)
        if callable(find_text) and callable(set_index):
            index = int(find_text(text))
            if index >= 0:
                set_index(index)

    def _attach_row_ui_metadata(self, row: RepairResultRow, namespace: str) -> None:
        """Attach row UI metadata to the best available layer metadata target."""
        target = row.generation_result_layer_id or row.created_layer
        if not target:
            return
        if namespace == "prompt":
            attach = getattr(self.metadata_service, "attach_prompt_metadata", None)
        else:
            attach = getattr(self.metadata_service, "attach_result_metadata", None)
        if callable(attach):
            attach(target, row.to_metadata())

    def _extract_tags_for_selected_results(self) -> None:
        """Start async image2tagger extraction for selected result rows."""
        rows = self.result_selection_model.selected_active_results()
        if not rows:
            self._show_info("No selected detection results are available.")
            return
        self._start_prompt_worker(rows)

    def _start_prompt_worker(self, rows: list[RepairResultRow]) -> None:
        """Create and start a prompt extraction worker."""
        threshold = self._parse_image2tagger_threshold()
        self.prompt_worker = PromptExtractionWorker(
            self.prompt_extraction_service,
            on_progress=self._on_prompt_progress,
            on_row_finished=self._on_prompt_row_finished,
            on_completed=self._on_prompt_completed,
            threshold=threshold,
        )
        self.prompt_worker.enqueue(rows)
        self._refresh_result_rows()
        self.prompt_worker.start()

    def _cancel_tag_extraction(self) -> None:
        """Cancel queued prompt extraction work."""
        if self.prompt_worker is not None:
            self.prompt_worker.cancel()
        self._refresh_result_rows()

    def _on_prompt_progress(self, progress: PromptExtractionProgress) -> None:
        """Update visible prompt extraction progress."""
        self._log_docker.append_log(
            f"Prompt extraction: {progress.completed} / {progress.total}"
        )

    def _on_prompt_row_finished(self, row: RepairResultRow, _result: Any | None) -> None:
        """Refresh one completed prompt row."""
        if getattr(row, "removed", False):
            return
        self._refresh_result_rows()

    def _on_prompt_completed(self, progress: PromptExtractionProgress) -> None:
        """Refresh prompt extraction completion state."""
        self._log_docker.append_log(
            f"Prompt extraction: {progress.completed} / {progress.total}"
            + (" cancelled" if progress.cancelled else "")
        )
        self._refresh_result_rows()

    def _on_generation_row_finished(self, row: Any, result: Any) -> None:
        """Refresh rows after async bbox generation completes."""
        if getattr(row, "is_refine_proxy", False):
            self._refresh_group_rows()
            status = "success" if getattr(result, "success", False) else "failed"
            error = str(getattr(result, "error", "") or "")
            self._log_docker.append_log(
                f"Refine generation {status}" + (f": {error}" if error else "")
            )
            return
        self._refresh_result_rows()
        if result is not None and not getattr(result, "success", False):
            error = str(getattr(result, "error", "") or "")
            if error:
                self._show_error(error)

    def _generate_selected_results(self) -> None:
        """Generate bbox repairs for selected detection result rows (sequential)."""
        rows = self.result_selection_model.selected_active_results()
        if not rows:
            self._show_info("No selected detection results are available.")
            return

        base_positive, base_negative = self.bbox_generation_service.active_model_prompt_snapshot()

        tasks_and_rows: list[tuple] = []
        for row in rows:
            try:
                task = self.bbox_generation_service.task_from_result_row(
                    row,
                    base_positive=base_positive,
                    base_negative=base_negative,
                )
                task.attach_transparency_mask = self._attach_mask_checkbox.isChecked()
                tasks_and_rows.append((task, row))
            except Exception as exc:
                row.mark_generation_failed(str(exc))

        if tasks_and_rows:
            self.bbox_generation_service.enqueue_batch_sequential(tasks_and_rows)
        self._refresh_result_rows()

    def _refresh_report(self, reports: list[GroupDetectionReport]) -> None:
        """Render a traceable batch report."""
        if not reports:
            self._log_docker.set_report("Batch Report: no detector results.")
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

        self._log_docker.set_report("\n".join(lines))

    def _refresh_status(self) -> None:
        """Refresh detector status text."""
        status = self.detector_manager.status()
        modes = ", ".join(status.loaded_modes) if status.loaded_modes else "none"
        self._status_label.setText(f"Detector: {status.state}; loaded modes: {modes}")

    def _refresh_detector_filter_visibility(self) -> None:
        """Show censor filter controls only when detector mode is censor."""
        is_censor = self._current_mode() == "censor"
        self._censor_filter_combo.setVisible(is_censor)
        self._censor_filter_label.setVisible(is_censor)

    def _parse_image2tagger_threshold(self) -> float | None:
        """Parse the image2tagger threshold input, returning None on invalid."""
        raw = str(self._image2tagger_threshold_input.text() or "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
            if not (0.0 <= value <= 1.0):
                self._show_error(
                    f"Image2tagger threshold must be between 0.0 and 1.0, got {value}"
                )
                return None
            return value
        except ValueError:
            self._show_error(f"Image2tagger threshold is not a valid number: {raw}")
            return None

    def _refine_selected_groups(self) -> None:
        """Run group refine for selected eligible group rows."""
        try:
            rows = self.group_selection_model.selected_active_groups()
            if not rows:
                self._show_info("No selected resolved groups are available.")
                return
            eligible = [row for row in rows if row.refine_eligible]
            if not eligible:
                reasons = set(row.refine_reason for row in rows)
                self._show_info(
                    "No selected groups are eligible for refine. "
                    f"Reasons: {', '.join(reasons)}"
                )
                return
            self.group_refine_service.repair_state_store = self._current_repair_state_store()
            reports = self.group_refine_service.refine_rows(eligible)
            self._refresh_group_rows()
            success = sum(1 for r in reports if r.get("status") == "success")
            skipped = sum(1 for r in reports if r.get("status") == "skipped")
            failed = sum(1 for r in reports if r.get("status") == "failed")
            self._log_docker.set_report(
                f"Refine Report: success={success}, skipped={skipped}, "
                f"failed={failed}, total={len(reports)}"
            )
        except Exception as exc:
            self._show_error(str(exc))

    def _sync_group_row(self, row: RepairGroupRow) -> bool:
        """Sync one group row active source into RepairStateStore."""
        try:
            if not row.is_resolved or row.group_layer is None:
                self._show_info("Group is not resolved; cannot sync.")
                return False
            if len(row.source_layers) != 1:
                self._show_info("Group must have exactly one resolved active source layer to sync.")
                return False

            canonical_layer_ids = [
                str(layer_id or "")
                for layer_id in list(getattr(row.record, "layer_ids", []) or [])
                if str(layer_id or "")
            ]
            if len(canonical_layer_ids) != 1:
                self._show_info("SyncRecord.layer_ids must contain exactly one canonical layer id.")
                return False

            canonical_layer_id = canonical_layer_ids[0]
            active_layer = row.source_layers[0]
            active_layer_id = str(getattr(active_layer, "id_string", "") or "")
            if not active_layer_id:
                self._show_info("Active source layer id could not be resolved.")
                return False

            store = self._current_repair_state_store()
            record = store.resolve_by_canonical_layer_id(canonical_layer_id)
            if record is None:
                record = RepairStateRecord(canonical_layer_id=canonical_layer_id)

            record.export_key = str(getattr(row.record, "export_key", "") or record.export_key or "")
            record.group_id = getattr(row.record, "group_id", None) or record.group_id
            record.group_name = getattr(row.record, "group_name", None) or record.group_name
            record.active_layer_id = active_layer_id
            record.active_layer_name = str(getattr(active_layer, "name", "") or "")

            if canonical_layer_id != active_layer_id:
                record.replacements[canonical_layer_id] = active_layer_id
                if canonical_layer_id not in record.deleted_layer_ids:
                    record.deleted_layer_ids.append(canonical_layer_id)

            store.upsert_record(record)

            self._log_docker.append_log(
                f"Synced group {row.display_name}: active repair layer state updated"
            )
            self._refresh_group_rows()
            return True
        except Exception as exc:
            self._show_error(f"Sync failed for {row.display_name}: {exc}")
            return False

    def _sync_all_refined_groups(self) -> None:
        """Sync all selected resolved groups (persist current layer_ids)."""
        try:
            rows = self.group_selection_model.selected_active_groups()
            if not rows:
                self._show_info("No selected resolved groups are available.")
                return

            success = 0
            failed = 0
            for row in rows:
                try:
                    if self._sync_group_row(row):
                        success += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

            self._log_docker.set_report(
                f"Sync All Report: success={success}, failed={failed}, "
                f"total={len(rows)}"
            )
        except Exception as exc:
            self._show_error(str(exc))

    def _is_group_node_check(self, layer: Any) -> bool:
        """Check if a layer is a group node (for sync filtering)."""
        from .repair_compat import is_group_layer
        try:
            if is_group_layer(layer):
                return True
        except Exception:
            pass
        node = getattr(layer, "node", layer)
        layer_type = str(getattr(layer, "type", "") or "").lower()
        node_type = getattr(node, "type", None)
        if callable(node_type):
            try:
                layer_type = str(node_type() or "").lower()
            except Exception:
                pass
        return layer_type in {"grouplayer", "group_layer", "group"}

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

    def closeEvent(self, event: Any) -> None:
        """Close log docker when main docker closes."""
        if hasattr(self, "_log_docker") and self._log_docker is not None:
            self._log_docker.close()
        super().closeEvent(event)
