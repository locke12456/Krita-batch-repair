"""Group-level image-to-tag extraction service."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from krita_ai_metadata.sync_map_store import SyncMapStore

from .group_selection_model import RepairGroupRow, _normalize_threshold_key
from .prompt_extraction_service import PromptExtractionService
from .repair_compat import render_node_projection


@dataclass(slots=True)
class GroupTagExtractionReport:
    group_name: str
    export_key: str
    threshold: float
    status: str
    reason: str = ""
    tag_text: str = ""
    error: str = ""


class GroupTagExtractionService:
    """Extract group-level tag text and persist it into SyncRecord tag cache."""

    def __init__(
        self,
        prompt_service: PromptExtractionService,
        sync_map_store: SyncMapStore,
    ) -> None:
        self.prompt_service = prompt_service
        self.sync_map_store = sync_map_store

    def extract_for_rows(
        self,
        rows: list[RepairGroupRow],
        threshold: float,
    ) -> list[GroupTagExtractionReport]:
        """Extract tags for resolved one-source group rows."""
        reports: list[GroupTagExtractionReport] = []
        threshold_value = float(threshold)
        threshold_key = _normalize_threshold_key(threshold_value)

        for row in rows:
            if not row.is_resolved:
                reports.append(
                    self._report(row, threshold_value, "skipped", reason="group unresolved")
                )
                continue
            if len(row.source_layers) != 1:
                reports.append(
                    self._report(
                        row,
                        threshold_value,
                        "skipped",
                        reason="requires exactly one source layer",
                    )
                )
                continue

            source_layer = row.source_layers[0]
            try:
                rendered = render_node_projection(source_layer)
                projection_png = bytes(rendered.to_bytes())
                if not projection_png:
                    reports.append(
                        self._report(
                            row,
                            threshold_value,
                            "failed",
                            error="Source layer projection rendered empty bytes.",
                        )
                    )
                    continue

                result = self.prompt_service.extract_prompt_from_bytes(
                    layer_id=row.export_key,
                    image_bytes=projection_png,
                    threshold=threshold_value,
                )
                if not result.success:
                    reports.append(
                        self._report(
                            row,
                            threshold_value,
                            "failed",
                            error=result.error_message,
                        )
                    )
                    continue

                snapshot = copy.deepcopy(getattr(row.record, "params_snapshot", None) or {})
                metadata = snapshot.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                    snapshot["metadata"] = metadata
                tag_cache = metadata.get("tag_cache")
                if not isinstance(tag_cache, dict):
                    tag_cache = {}
                    metadata["tag_cache"] = tag_cache

                tag_cache[threshold_key] = result.prompt_text
                row.record.params_snapshot = snapshot
                self.sync_map_store.record_apply(row.record)
                row.refine_threshold = threshold_value

                reports.append(
                    self._report(
                        row,
                        threshold_value,
                        "success",
                        tag_text=result.prompt_text,
                    )
                )
            except Exception as exc:
                reports.append(
                    self._report(row, threshold_value, "failed", error=str(exc))
                )

        return reports

    def _report(
        self,
        row: RepairGroupRow,
        threshold: float,
        status: str,
        *,
        reason: str = "",
        tag_text: str = "",
        error: str = "",
    ) -> GroupTagExtractionReport:
        """Build a report for one group row."""
        return GroupTagExtractionReport(
            group_name=row.group_name or "",
            export_key=row.export_key,
            threshold=float(threshold),
            status=status,
            reason=reason,
            tag_text=tag_text,
            error=error,
        )
