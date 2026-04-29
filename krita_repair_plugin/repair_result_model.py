from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4

from krita_ai_metadata.sync_map_store import SyncRecord


PROMPT_NOT_STARTED = "not_started"
PROMPT_QUEUED = "queued"
PROMPT_RUNNING = "running"
PROMPT_DONE = "done"
PROMPT_FAILED = "failed"
PROMPT_CANCELLED = "cancelled"

GENERATION_NOT_STARTED = "not_started"
GENERATION_QUEUED = "queued"
GENERATION_RUNNING = "running"
GENERATION_DONE = "done"
GENERATION_FAILED = "failed"
GENERATION_CANCELLED = "cancelled"

MERGE_NOT_STARTED = "not_started"
MERGE_READY = "ready"
MERGE_MERGED = "merged"
MERGE_FAILED = "failed"


def _safe_layer_id(layer: Any | None) -> str:
    if layer is None:
        return ""
    return str(getattr(layer, "id_string", "") or "")


def _safe_layer_name(layer: Any | None) -> str:
    if layer is None:
        return ""
    return str(getattr(layer, "name", "") or "")


def _bbox_copy(value: dict[str, int] | None) -> dict[str, int]:
    if not value:
        return {}
    return {str(key): int(raw) for key, raw in value.items()}


@dataclass(slots=True)
class RepairResultRow:
    result_id: str = field(default_factory=lambda: uuid4().hex)
    selected: bool = True
    active: bool = True
    visible: bool = True
    removed: bool = False
    prompt_type: str = ""
    prompt_type_prompt: str = ""
    prompt_type_applied: bool = False
    merge_status: str = MERGE_NOT_STARTED
    merge_error: str = ""
    merged_layer_id: str = ""
    merged_layer_name: str = ""

    record: SyncRecord | None = None
    group_layer: Any | None = None
    source_layer: Any | None = None
    created_layer: Any | None = None

    detector_bbox: dict[str, int] = field(default_factory=dict)
    crop_bbox: dict[str, int] = field(default_factory=dict)
    crop_png_bytes: bytes = b""

    detector_mode: str = ""
    detector_label: str = ""
    detector_score: float = 0.0

    force_rect_crop: bool = False
    rect_width: int = 0
    rect_height: int = 0

    prompt_text: str = ""
    prompt_raw_output: Any | None = None
    prompt_success: bool = False
    prompt_error: str = ""
    prompt_status: str = PROMPT_NOT_STARTED
    prompt_progress_index: int = 0
    prompt_progress_total: int = 0

    generation_status: str = GENERATION_NOT_STARTED
    generation_job_id: str = ""
    generation_result_layer_id: str = ""
    generation_result_layer_name: str = ""
    generation_error: str = ""

    @property
    def group_id(self) -> str | None:
        return None if self.record is None else self.record.group_id

    @property
    def group_name(self) -> str | None:
        return None if self.record is None else self.record.group_name

    @property
    def export_key(self) -> str:
        return "" if self.record is None else self.record.export_key

    @property
    def source_layer_id(self) -> str:
        return _safe_layer_id(self.source_layer)

    @property
    def source_layer_name(self) -> str:
        return _safe_layer_name(self.source_layer)

    @property
    def created_layer_id(self) -> str:
        return _safe_layer_id(self.created_layer)

    @property
    def created_layer_name(self) -> str:
        return _safe_layer_name(self.created_layer)

    @property
    def display_name(self) -> str:
        group = self.group_name or self.export_key or "group"
        source = self.source_layer_name or "source"
        bbox = self.crop_bbox or self.detector_bbox
        if bbox:
            x = int(bbox.get("x", bbox.get("x1", 0)))
            y = int(bbox.get("y", bbox.get("y1", 0)))
            width = int(bbox.get("width", max(0, int(bbox.get("x2", 0)) - x)))
            height = int(bbox.get("height", max(0, int(bbox.get("y2", 0)) - y)))
            return f"{group} | {source} | {x},{y} {width}x{height}"
        return f"{group} | {source}"

    def mark_prompt_queued(self, index: int = 0, total: int = 0) -> None:
        self.prompt_status = PROMPT_QUEUED
        self.prompt_progress_index = int(index)
        self.prompt_progress_total = int(total)
        self.prompt_error = ""

    def mark_prompt_running(self, index: int = 0, total: int = 0) -> None:
        self.prompt_status = PROMPT_RUNNING
        self.prompt_progress_index = int(index)
        self.prompt_progress_total = int(total)
        self.prompt_error = ""

    def mark_prompt_done(
        self,
        prompt_text: str,
        raw_output: Any | None = None,
        index: int = 0,
        total: int = 0,
    ) -> None:
        self.prompt_status = PROMPT_DONE
        self.prompt_text = str(prompt_text or "")
        self.prompt_raw_output = raw_output
        self.prompt_success = True
        self.prompt_error = ""
        self.prompt_progress_index = int(index)
        self.prompt_progress_total = int(total)

    def mark_prompt_failed(self, error: str, index: int = 0, total: int = 0) -> None:
        self.prompt_status = PROMPT_FAILED
        self.prompt_success = False
        self.prompt_error = str(error or "")
        self.prompt_progress_index = int(index)
        self.prompt_progress_total = int(total)

    def mark_prompt_cancelled(self, index: int = 0, total: int = 0) -> None:
        self.prompt_status = PROMPT_CANCELLED
        self.prompt_success = False
        self.prompt_error = "Cancelled"
        self.prompt_progress_index = int(index)
        self.prompt_progress_total = int(total)

    def mark_generation_running(self, job_id: str = "") -> None:
        self.generation_status = GENERATION_RUNNING
        self.generation_job_id = str(job_id or "")
        self.generation_error = ""

    def mark_generation_done(
        self,
        layer_id: str,
        layer_name: str,
        job_id: str = "",
    ) -> None:
        self.generation_status = GENERATION_DONE
        self.generation_job_id = str(job_id or self.generation_job_id or "")
        self.generation_result_layer_id = str(layer_id or "")
        self.generation_result_layer_name = str(layer_name or "")
        self.generation_error = ""

    def mark_generation_failed(self, error: str, job_id: str = "") -> None:
        self.generation_status = GENERATION_FAILED
        self.generation_job_id = str(job_id or self.generation_job_id or "")
        self.generation_error = str(error or "")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "repair_plugin.result_id": self.result_id,
            "repair_plugin.selected": bool(self.selected),
            "repair_plugin.active": bool(self.active),
            "repair_plugin.visible": bool(self.visible),
            "repair_plugin.removed": bool(self.removed),
            "repair_plugin.prompt_type": self.prompt_type,
            "repair_plugin.prompt_type_prompt": self.prompt_type_prompt,
            "repair_plugin.prompt_type_applied": bool(self.prompt_type_applied),
            "repair_plugin.merge_status": self.merge_status,
            "repair_plugin.merge_error": self.merge_error,
            "repair_plugin.merged_layer_id": self.merged_layer_id,
            "repair_plugin.merged_layer_name": self.merged_layer_name,
            "repair_plugin.source_group_id": self.group_id,
            "repair_plugin.source_group_name": self.group_name,
            "repair_plugin.export_key": self.export_key,
            "repair_plugin.source_layer_id": self.source_layer_id,
            "repair_plugin.source_layer_name": self.source_layer_name,
            "repair_plugin.detector_mode": self.detector_mode,
            "repair_plugin.detector_label": self.detector_label,
            "repair_plugin.detector_bbox": _bbox_copy(self.detector_bbox),
            "repair_plugin.crop_bbox": _bbox_copy(self.crop_bbox),
            "repair_plugin.detector_score": float(self.detector_score),
            "repair_plugin.force_rect_crop": bool(self.force_rect_crop),
            "repair_plugin.rect_width": int(self.rect_width),
            "repair_plugin.rect_height": int(self.rect_height),
            "repair_plugin.prompt_text": self.prompt_text,
            "repair_plugin.prompt_raw_output": self.prompt_raw_output,
            "repair_plugin.prompt_success": bool(self.prompt_success),
            "repair_plugin.prompt_error": self.prompt_error,
            "repair_plugin.prompt_status": self.prompt_status,
            "repair_plugin.created_layer_id": self.created_layer_id,
            "repair_plugin.created_layer_name": self.created_layer_name,
            "repair_plugin.generation_status": self.generation_status,
            "repair_plugin.generation_job_id": self.generation_job_id,
            "repair_plugin.generation_result_layer_id": self.generation_result_layer_id,
            "repair_plugin.generation_result_layer_name": self.generation_result_layer_name,
            "repair_plugin.generation_error": self.generation_error,
        }


class RepairResultSelectionModel:
    def __init__(self) -> None:
        self.rows: list[RepairResultRow] = []

    def replace_rows(self, rows: Iterable[RepairResultRow]) -> None:
        self.rows = list(rows)

    def append_rows(self, rows: Iterable[RepairResultRow]) -> None:
        self.rows.extend(list(rows))

    def non_removed_rows(self) -> list[RepairResultRow]:
        return [row for row in self.rows if not row.removed]

    def rows_for_prompt_type(self, prompt_type: str) -> list[RepairResultRow]:
        prompt_type = str(prompt_type or "").strip()
        if not prompt_type or prompt_type.lower() == "all":
            return self.non_removed_rows()
        return [
            row
            for row in self.non_removed_rows()
            if row.prompt_type_applied and row.prompt_type == prompt_type
        ]

    def visibility_target_rows(
        self,
        prompt_type: str = "",
        filter_enabled: bool = False,
    ) -> list[RepairResultRow]:
        if filter_enabled:
            return self.rows_for_prompt_type(prompt_type)
        return self.non_removed_rows()

    def remove_result(self, result_id: str) -> RepairResultRow | None:
        for row in self.rows:
            if row.result_id == result_id:
                row.removed = True
                row.selected = False
                row.active = False
                return row
        return None

    def selected_active_results(self) -> list[RepairResultRow]:
        return [
            row
            for row in self.rows
            if row.selected and row.active and not row.removed
        ]

    def select_all(self) -> None:
        for row in self.rows:
            if row.active and not row.removed:
                row.selected = True

    def clear_selected(self) -> None:
        for row in self.rows:
            row.selected = False

    def clear(self) -> None:
        self.rows.clear()

    def find_by_result_id(self, result_id: str) -> RepairResultRow | None:
        for row in self.rows:
            if row.result_id == result_id:
                return row
        return None

    def update_prompt_progress(self, completed: int, total: int) -> None:
        for index, row in enumerate(self.non_removed_rows(), start=1):
            if row.prompt_status in {PROMPT_QUEUED, PROMPT_RUNNING}:
                row.prompt_progress_index = int(completed or index)
                row.prompt_progress_total = int(total)

    def counts(self) -> dict[str, int]:
        rows = self.non_removed_rows()
        return {
            "total": len(rows),
            "removed": len([row for row in self.rows if row.removed]),
            "selected": len([row for row in rows if row.selected]),
            "active": len([row for row in rows if row.active]),
            "prompt_done": len([row for row in rows if row.prompt_status == PROMPT_DONE]),
            "generated": len([row for row in rows if row.generation_status == GENERATION_DONE]),
            "failed": len(
                [
                    row
                    for row in rows
                    if row.prompt_status == PROMPT_FAILED
                    or row.generation_status == GENERATION_FAILED
                ]
            ),
        }