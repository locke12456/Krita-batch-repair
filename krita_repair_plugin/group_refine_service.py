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
from .repair_diagnostics import emit_log, exception_detail
from .repair_state_store import RepairStateStore


def _indent_block(text: str, prefix: str = "    ") -> str:
    """Indent every line of a multi-line detail block."""
    return "\n".join(f"{prefix}{line}" for line in str(text or "").splitlines())


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

    def to_log_line(self) -> str:
        """Return a traceable log entry including the full failure detail."""
        marker = {"success": "[+]", "skipped": "[-]"}.get(self.status, "[x]")
        parts = [
            f"{marker} {self.group_name or self.export_key or '<unnamed>'}",
            f"status={self.status}",
        ]
        if self.source_layer_name:
            parts.append(f"source={self.source_layer_name}")
        if self.job_id:
            parts.append(f"job={self.job_id}")
        if self.created_layer_name or self.created_layer_id:
            parts.append(f"layer={self.created_layer_name or self.created_layer_id}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        line = " | ".join(parts)
        if self.error:
            line = f"{line}\n{_indent_block(self.error)}"
        return line


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
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.group_row = group_row
        self.source_layer = source_layer
        self.document_ref = document_ref
        self.repair_state_store = repair_state_store
        self.log_callback = log_callback
        self.generation_status: str = "not_started"
        self.generation_job_id: str = ""
        self.generation_result_layer_id: str = ""
        self.generation_result_layer_name: str = ""
        self.generation_error: str = ""

    def _row_label(self) -> str:
        row = self.group_row
        return str(
            getattr(row, "display_name", "") or getattr(row, "export_key", "") or "group"
        )

    def _log(self, text: str) -> None:
        emit_log(self.log_callback, f"[refine:{self._row_label()}] {text}")

    def mark_generation_running(self, job_id: str = "") -> None:
        self.generation_status = "running"
        self.generation_job_id = str(job_id or "")
        self.generation_error = ""
        self._log(f"generation running (job={self.generation_job_id or '?'})")

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
        self._log(
            f"generation done (job={self.generation_job_id or '?'}, "
            f"layer={layer_name or layer_id or '?'})"
        )
        self._replace_source_layer(layer_id)

    def mark_generation_failed(self, error: str, job_id: str = "") -> None:
        self.generation_status = "failed"
        self.generation_job_id = str(job_id or self.generation_job_id or "")
        self.generation_error = str(error or "") or "<no error detail reported>"
        self._log(
            f"generation FAILED (job={self.generation_job_id or '?'})\n"
            f"{_indent_block(self.generation_error)}"
        )

    def _replace_source_layer(self, layer_id: str) -> None:
        """Replace the original source layer in group_row.source_layers and persist."""
        if not layer_id or self.document_ref is None:
            return
        created_layer = find_krita_node_by_id(self.document_ref, layer_id)
        if created_layer is None:
            self._log(
                f"WARNING: could not resolve created layer id={layer_id}; "
                "source_layers not replaced."
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
            self._log(
                "WARNING: original source_layer not found in group_row.source_layers "
                "(may have been refreshed); skipping replacement."
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
            self._log(
                "WARNING: repair state was not persisted; old source layer will not be "
                f"deleted (canonical_layer_id={canonical_layer_id or '<empty>'}, "
                f"record_layer_ids={layer_ids}, "
                f"state_store={'set' if self.repair_state_store is not None else 'None'})."
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
                self._log(
                    f"WARNING: failed to delete old source layer id={old_id}\n"
                    f"{_indent_block(exception_detail(exc))}"
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
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.bbox_generation_service = bbox_generation_service
        self.metadata_service = metadata_service
        self.on_row_finished = on_row_finished
        self.repair_state_store = repair_state_store
        self.log_callback = log_callback

    def _log(self, text: str) -> None:
        """Send one traceable line to the log docker."""
        emit_log(self.log_callback, str(text))

    def refine_rows(self, rows: list[RepairGroupRow]) -> list[dict[str, Any]]:
        """Refine eligible group rows and return per-row report dicts."""
        reports: list[dict[str, Any]] = []
        self._log(f"[refine] start: {len(rows)} row(s) requested")

        base_positive = ""
        try:
            base_positive, _base_negative = self.bbox_generation_service.active_model_prompt_snapshot()
        except Exception as exc:
            self._log(
                "[refine] WARNING: failed to read active prompt snapshot\n"
                f"{_indent_block(exception_detail(exc))}"
            )
        if not base_positive:
            self._log(
                "[refine] note: active Krita AI positive prompt is empty; only the "
                "per-group refine fragment will be used."
            )

        for row in rows:
            if not row.refine_eligible:
                report = GroupRefineReport(
                    group_name=row.group_name or "",
                    export_key=row.export_key,
                    source_layer_name="",
                    status="skipped",
                    reason=row.refine_reason or "<no reason reported>",
                )
                reports.append(report.to_dict())
                self._log(report.to_log_line())
                self._notify_row_finished(row, report)
                continue

            report = self._refine_one_group(row, base_positive=base_positive)
            reports.append(report.to_dict())
            self._log(report.to_log_line())
            self._notify_row_finished(row, report)

        return reports

    def _compose_refine_prompt(
        self,
        base_positive: str,
        refine_fragment: str,
    ) -> str:
        """Compose active Krita AI prompt with the selected refine fragment.

        Style prompt is not appended here. The active krita-ai-diffusion
        workflow remains responsible for style prompt merging.
        """
        base = str(base_positive or "").strip()
        fragment = str(refine_fragment or "").strip()
        if base and fragment:
            return f"{base}, {fragment}"
        return base or fragment

    def _refine_one_group(
        self,
        row: RepairGroupRow,
        base_positive: str = "",
    ) -> GroupRefineReport:
        """Refine one group row using the full group composite as input."""
        source_layer = row.source_layers[0] if row.source_layers else row.group_layer
        source_name = str(getattr(row.group_layer, "name", "") or row.display_name or "group")
        label = row.display_name or row.export_key or "group"
        stage = "init"
        try:
            self._log(
                f"[refine:{label}] begin | mode={row.refine_source_mode} | "
                f"source_layers={len(row.source_layers)} | "
                f"fragment_chars={len(row.refine_source_text or '')}"
            )
            stage = "render group projection"
            rendered = render_node_projection(row.group_layer)
            projection_bounds = rendered.bounds
            projection_png = bytes(rendered.to_bytes())
            if not projection_png:
                return GroupRefineReport(
                    group_name=row.group_name or "",
                    export_key=row.export_key,
                    source_layer_name=source_name,
                    status="failed",
                    error=(
                        "Group projection rendered empty bytes "
                        f"(group_layer={source_name!r}, bounds={projection_bounds!r}). "
                        "The group is likely hidden, empty, or fully transparent."
                    ),
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
                    error=(
                        f"Group projection bounds are empty (bbox={bbox}). "
                        "The group has no visible pixel content to refine."
                    ),
                )

            stage = "compose prompt"
            prompt_text = self._compose_refine_prompt(
                base_positive=base_positive,
                refine_fragment=row.refine_source_text,
            )
            self._log(
                f"[refine:{label}] bbox={bbox['x']},{bbox['y']},"
                f"{bbox['width']}x{bbox['height']} | png_bytes={len(projection_png)} | "
                f"prompt_chars={len(prompt_text)}"
            )
            if not prompt_text.strip():
                return GroupRefineReport(
                    group_name=row.group_name or "",
                    export_key=row.export_key,
                    source_layer_name=source_name,
                    status="failed",
                    error=(
                        "Composed refine prompt is empty: both the active Krita AI "
                        "positive prompt and the group refine fragment "
                        f"(mode={row.refine_source_mode}) are blank."
                    ),
                )

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
            if document_ref is None:
                self._log(
                    f"[refine:{label}] WARNING: source layer exposes no document_ref; "
                    "the generated layer cannot replace the old source layer."
                )
            proxy = RefineRowProxy(
                group_row=row,
                source_layer=source_layer,
                document_ref=document_ref,
                repair_state_store=self.repair_state_store,
                log_callback=self.log_callback,
            )
            if self.repair_state_store is None:
                self._log(
                    f"[refine:{label}] WARNING: no RepairStateStore is attached; "
                    "the refine result will not be persisted."
                )

            stage = "enqueue generation task"
            result = self.bbox_generation_service.enqueue_task(task, row=proxy)
            error = str(getattr(result, "error", "") or "")
            if not result.success and not error:
                error = "enqueue_task reported failure without an error message."
            return GroupRefineReport(
                group_name=row.group_name or "",
                export_key=row.export_key,
                source_layer_name=source_name,
                status="success" if result.success else "failed",
                created_layer_id=result.created_layer_id,
                created_layer_name=result.created_layer_name,
                job_id=result.job_id,
                error=error,
            )

        except Exception as exc:
            return GroupRefineReport(
                group_name=row.group_name or "",
                export_key=row.export_key,
                source_layer_name=source_name,
                status="failed",
                error=f"while {stage}: {exception_detail(exc)}",
            )

    def _notify_row_finished(
        self,
        row: RepairGroupRow,
        report: GroupRefineReport,
    ) -> None:
        """Notify the optional row-finished callback."""
        callback = self.on_row_finished
        if not callable(callback):
            return
        try:
            callback(row, report)
        except Exception as exc:
            self._log(
                "[refine] WARNING: on_row_finished callback raised\n"
                f"{_indent_block(exception_detail(exc))}"
            )