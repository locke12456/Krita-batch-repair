"""Record-based group refine orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .bbox_generation_service import (
    BBoxGenerationService,
    RepairGenerationTask,
)
from .group_selection_model import RepairGroupRow
from .repair_compat import find_krita_node_by_id, render_node_projection
from .repair_state_store import RepairStateStore


@dataclass(slots=True)
class GroupRefineReport:
    """Report for one group refine attempt."""

    group_name: str
    export_key: str
    source_layer_name: str
    status: str  # "success" | "skipped" | "failed"
    reason: str = ""
    created_layer_id: str = ""
    created_layer_name: str = ""
    job_id: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "export_key": self.export_key,
            "source_layer_name": self.source_layer_name,
            "status": self.status,
            "reason": self.reason,
            "created_layer_id": self.created_layer_id,
            "created_layer_name": self.created_layer_name,
            "job_id": self.job_id,
            "error": self.error,
        }


class RefineRowProxy:
    """Duck-typed row proxy for BBoxGenerationService.enqueue_task().

    After async generation completes, replaces the original source layer
    in RepairGroupRow.source_layers with the newly created refine layer.
    """

    is_refine_proxy: bool = True

    def __init__(
        self,
        group_row: RepairGroupRow,
        source_layer: Any,
        document_ref: Any,
        repair_state_store: RepairStateStore | None = None,
    ) -> None:
        self.group_row = group_row
        self.source_layer = source_layer
        self.document_ref = document_ref
        self.repair_state_store = repair_state_store
        self.generation_status: str = "not_started"
        self.generation_job_id: str = ""
        self.generation_result_layer_id: str = ""
        self.generation_result_layer_name: str = ""
        self.generation_error: str = ""

    def mark_generation_running(self, job_id: str = "") -> None:
        self.generation_status = "running"
        self.generation_job_id = str(job_id or "")
        self.generation_error = ""

    def mark_generation_done(
        self,
        layer_id: str,
        layer_name: str,
        job_id: str = "",
    ) -> None:
        self.generation_status = "done"
        self.generation_job_id = str(job_id or self.generation_job_id or "")
        self.generation_result_layer_id = str(layer_id or "")
        self.generation_result_layer_name = str(layer_name or "")
        self.generation_error = ""
        self._replace_source_layer(layer_id)

    def mark_generation_failed(self, error: str, job_id: str = "") -> None:
        self.generation_status = "failed"
        self.generation_job_id = str(job_id or self.generation_job_id or "")
        self.generation_error = str(error or "")

    def _replace_source_layer(self, layer_id: str) -> None:
        """Replace the original source layer in group_row.source_layers and persist."""
        if not layer_id or self.document_ref is None:
            return
        created_layer = find_krita_node_by_id(self.document_ref, layer_id)
        if created_layer is None:
            print(
                f"[RefineRowProxy] WARNING: could not resolve created layer "
                f"id={layer_id}; source_layers not replaced."
            )
            return

        # 1. In-memory replacement
        source_layers = self.group_row.source_layers
        old_id = getattr(self.source_layer, "id_string", None)
        replaced = False
        for i, existing in enumerate(source_layers):
            if existing is self.source_layer:
                source_layers[i] = created_layer
                replaced = True
                break
        if not replaced:
            print(
                "[RefineRowProxy] WARNING: original source_layer not found in "
                "group_row.source_layers (may have been refreshed); skipping replacement."
            )

        record = self.group_row.record
        canonical_layer_id = ""
        layer_ids = list(getattr(record, "layer_ids", []) or [])
        if len(layer_ids) == 1:
            canonical_layer_id = str(layer_ids[0] or "")

        if old_id and layer_id and canonical_layer_id and self.repair_state_store is not None:
            self.repair_state_store.record_refine_success(
                canonical_layer_id=canonical_layer_id,
                old_layer_id=str(old_id),
                new_layer_id=str(layer_id),
                export_key=str(getattr(record, "export_key", "") or ""),
                group_id=getattr(record, "group_id", None),
                group_name=getattr(record, "group_name", None),
                active_layer_name=str(getattr(created_layer, "name", "") or ""),
                job_id=str(self.generation_job_id or ""),
                seed=getattr(record, "seed", None),
            )
        elif old_id and layer_id:
            print(
                "[RefineRowProxy] WARNING: repair state was not persisted; "
                "old source layer will not be deleted."
            )
            return

        # Delete old source layer only after RepairStateStore persistence succeeds.
        if old_id and replaced:
            try:
                from .repair_compat import delete_layer as _del_old
                old_node = find_krita_node_by_id(self.document_ref, old_id)
                if old_node is not None:
                    _del_old(old_node)
            except Exception as exc:
                print(
                    f"[RefineRowProxy] WARNING: failed to delete old source "
                    f"layer id={old_id}: {exc}"
                )


class GroupRefineService:
    """Orchestrate record-based group refine without detection result rows.

    Converts selected group rows with stored prompt snapshots into generation
    tasks accepted by the existing BBoxGenerationService in whole-crop refine
    mode (bbox == detector_bbox).
    """

    def __init__(
        self,
        bbox_generation_service: BBoxGenerationService,
        metadata_service: Any | None = None,
        on_row_finished: Callable[[RepairGroupRow, GroupRefineReport], None] | None = None,
        repair_state_store: RepairStateStore | None = None,
    ) -> None:
        self.bbox_generation_service = bbox_generation_service
        self.metadata_service = metadata_service
        self.on_row_finished = on_row_finished
        self.repair_state_store = repair_state_store

    def refine_rows(self, rows: list[RepairGroupRow]) -> list[dict[str, Any]]:
        """Refine eligible group rows and return per-row report dicts."""
        reports: list[dict[str, Any]] = []
        for row in rows:
            if not row.refine_eligible:
                report = GroupRefineReport(
                    group_name=row.group_name or "",
                    export_key=row.export_key,
                    source_layer_name="",
                    status="skipped",
                    reason=row.refine_reason,
                )
                reports.append(report.to_dict())
                self._notify_row_finished(row, report)
                continue

            if not row.source_layers:
                report = GroupRefineReport(
                    group_name=row.group_name or "",
                    export_key=row.export_key,
                    source_layer_name="",
                    status="skipped",
                    reason="no source layers",
                )
                reports.append(report.to_dict())
                self._notify_row_finished(row, report)
                continue

            for source_layer in row.source_layers:
                report = self._refine_one_source(row, source_layer)
                reports.append(report.to_dict())
                self._notify_row_finished(row, report)

        return reports

    def _refine_one_source(
        self,
        row: RepairGroupRow,
        source_layer: Any,
    ) -> GroupRefineReport:
        """Refine one source layer within a group row."""
        source_name = str(getattr(source_layer, "name", "") or "source")
        try:
            rendered = render_node_projection(source_layer)
            projection_bounds = rendered.bounds
            projection_png = bytes(rendered.to_bytes())
            if not projection_png:
                return GroupRefineReport(
                    group_name=row.group_name or "",
                    export_key=row.export_key,
                    source_layer_name=source_name,
                    status="failed",
                    error="Source layer projection rendered empty bytes.",
                )

            bbox = {
                "x": int(getattr(projection_bounds, "x", 0)),
                "y": int(getattr(projection_bounds, "y", 0)),
                "width": int(getattr(projection_bounds, "width", 0)),
                "height": int(getattr(projection_bounds, "height", 0)),
            }
            if bbox["width"] <= 0 or bbox["height"] <= 0:
                return GroupRefineReport(
                    group_name=row.group_name or "",
                    export_key=row.export_key,
                    source_layer_name=source_name,
                    status="failed",
                    error="Source layer projection bounds are empty.",
                )

            prompt_text = row.refine_source_text

            task = RepairGenerationTask(
                record=row.record,
                group_layer=row.group_layer,
                source_layer=source_layer,
                bbox=bbox,
                detector_bbox=bbox,
                crop_png_bytes=projection_png,
                prompt_text=prompt_text,
                detector_mode="refine",
                detector_label="group-refine",
            )

            document_ref = getattr(source_layer, "document_ref", None)
            proxy = RefineRowProxy(
                group_row=row,
                source_layer=source_layer,
                document_ref=document_ref,
                repair_state_store=self.repair_state_store,
            )
            result = self.bbox_generation_service.enqueue_task(task, row=proxy)
            return GroupRefineReport(
                group_name=row.group_name or "",
                export_key=row.export_key,
                source_layer_name=source_name,
                status="success" if result.success else "failed",
                created_layer_id=result.created_layer_id,
                created_layer_name=result.created_layer_name,
                job_id=result.job_id,
                error=result.error,
            )

        except Exception as exc:
            return GroupRefineReport(
                group_name=row.group_name or "",
                export_key=row.export_key,
                source_layer_name=source_name,
                status="failed",
                error=str(exc),
            )

    def _notify_row_finished(
        self,
        row: RepairGroupRow,
        report: GroupRefineReport,
    ) -> None:
        """Notify the optional row-finished callback."""
        callback = self.on_row_finished
        if callable(callback):
            callback(row, report)