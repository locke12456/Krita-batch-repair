from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .group_selection_model import RepairGroupRow
from .repair_result_model import (
    GENERATION_DONE,
    GENERATION_FAILED,
    GENERATION_RUNNING,
    GENERATION_QUEUED,
    MERGE_FAILED,
    MERGE_MERGED,
    PROMPT_FAILED,
    PROMPT_QUEUED,
    PROMPT_RUNNING,
    RepairResultRow,
)


@dataclass(frozen=True, slots=True)
class RepairGroupRowInfo:
    summary: str
    tooltip: str
    warning_badge: str


@dataclass(frozen=True, slots=True)
class RepairResultRowInfo:
    summary: str
    tooltip: str
    error_badge: str


class RepairRowInfoPresenter:
    """Format repair rows into compact visible text and detailed tooltips."""

    def for_group(self, row: RepairGroupRow) -> RepairGroupRowInfo:
        warning_badge = self._warning_badge(row)
        return RepairGroupRowInfo(
            summary=self._group_summary(row),
            tooltip=self._group_tooltip(row),
            warning_badge=warning_badge,
        )

    def for_result(self, row: RepairResultRow) -> RepairResultRowInfo:
        error_badge = self._error_badge(row)
        return RepairResultRowInfo(
            summary=self._result_summary(row),
            tooltip=self._result_tooltip(row),
            error_badge=error_badge,
        )

    def _group_summary(self, row: RepairGroupRow) -> str:
        resolved = "resolved" if row.is_resolved else "unresolved"
        parts = [
            f"#{row.sync_index}",
            resolved,
            f"layers={len(row.layer_ids)}",
            f"created={row.detected_count}",
        ]
        badge = self._warning_badge(row)
        if badge:
            parts.append(badge)
        return " · ".join(parts)

    def _group_tooltip(self, row: RepairGroupRow) -> str:
        lines = [
            self._line("Display name", row.display_name),
            self._line("Export key", row.export_key),
            self._line("Group id", row.group_id),
            self._line("Group name", row.group_name),
            self._line("Sync index", row.sync_index),
            self._line("Resolved", "yes" if row.is_resolved else "no"),
            self._line("Layer count", len(row.layer_ids)),
            self._line("Layer ids", ", ".join(row.layer_ids) if row.layer_ids else "none"),
            self._line("Source layer count", len(row.source_layers)),
            self._line("Created count", row.detected_count),
            self._line("Created layer ids", ", ".join(row.created_layer_ids) if row.created_layer_ids else "none"),
            self._line("Refine reason", row.refine_reason),
        ]
        if row.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in row.warnings)
        return "\n".join(lines)

    def _result_summary(self, row: RepairResultRow) -> str:
        prompt_type = row.effective_prompt_type() or "unclassified"
        parts = [
            "result",
            prompt_type,
            self._pipeline_state(row),
        ]
        display_name = self._result_display_name(row)
        if display_name.startswith("[Generated]"):
            parts.append(display_name)
        badge = self._error_badge(row)
        if badge and badge not in parts[-1]:
            parts.append(badge)
        return " · ".join(part for part in parts if part)

    def _result_tooltip(self, row: RepairResultRow) -> str:
        lines = [
            self._line("Display name", self._result_display_name(row)),
            self._line("Result id", row.result_id),
            self._line("Group id", row.group_id),
            self._line("Group name", row.group_name),
            self._line("Export key", row.export_key),
            self._line("Source layer id", row.source_layer_id),
            self._line("Source layer name", row.source_layer_name),
            self._line("Created layer id", row.created_layer_id),
            self._line("Created layer name", row.created_layer_name),
            self._line("Generated layer id", row.generation_result_layer_id),
            self._line("Generated layer name", row.generation_result_layer_name),
            self._line("Merged layer id", row.merged_layer_id),
            self._line("Merged layer name", row.merged_layer_name),
            self._line("Prompt type", row.effective_prompt_type() or "unclassified"),
            self._line("Prompt status", row.prompt_status),
            self._line("Generation status", row.generation_status),
            self._line("Merge status", row.merge_status),
            self._line("Generation job id", row.generation_job_id),
            self._line("Detector mode", row.detector_mode),
            self._line("Detector label", row.detector_label),
            self._line("Detector score", row.detector_score),
            self._line("Detector bbox", self._format_bbox(row.detector_bbox)),
            self._line("Crop bbox", self._format_bbox(row.crop_bbox)),
            self._line("Force rect crop", "yes" if row.force_rect_crop else "no"),
            self._line("Rect size", self._format_rect_size(row)),
            self._line("Prompt text", row.prompt_text),
            self._line("Prompt raw output", self._compact_value(row.prompt_raw_output)),
        ]

        errors = [
            ("Prompt error", row.prompt_error),
            ("Generation error", row.generation_error),
            ("Merge error", row.merge_error),
        ]
        for label, error in errors:
            if error:
                lines.append(self._line(label, error))

        return "\n".join(lines)

    def _pipeline_state(self, row: RepairResultRow) -> str:
        if row.merge_status == MERGE_FAILED:
            return "merge failed ❌"
        if row.generation_status == GENERATION_FAILED:
            return "generation failed ❌"
        if row.prompt_status == PROMPT_FAILED:
            return "tag failed ⚠"
        if row.generation_status in {GENERATION_RUNNING, GENERATION_QUEUED}:
            return "generating..."
        if row.prompt_status in {PROMPT_RUNNING, PROMPT_QUEUED}:
            return "tagging..."
        if row.merge_status == MERGE_MERGED:
            return "merged"
        if row.generation_status == GENERATION_DONE:
            return "generated"
        return "ready"

    def _warning_badge(self, row: RepairGroupRow) -> str:
        return "⚠" if row.warnings else ""

    def _error_badge(self, row: RepairResultRow) -> str:
        if row.merge_status == MERGE_FAILED or row.generation_status == GENERATION_FAILED:
            return "❌"
        if row.prompt_status == PROMPT_FAILED or row.prompt_error:
            return "⚠"
        return ""

    def _result_display_name(self, row: RepairResultRow) -> str:
        candidates = [
            getattr(row, "generation_result_layer_name", ""),
            getattr(row, "created_layer_name", ""),
            getattr(row, "display_name", ""),
        ]
        for candidate in candidates:
            text = self._safe_text(candidate, "")
            if text.startswith("[Generated"):
                short_name, seed_number = self._generated_name_parts(text)
                if seed_number:
                    return f"[Generated] {short_name} ({seed_number})"
                return f"[Generated] {short_name}"
        return self._safe_text(getattr(row, "display_name", ""), "unknown")

    def _generated_name_parts(self, value: str) -> tuple[str, str]:
        raw = str(value or "").strip()
        seed_match = re.search(r"\((\d+)\)\s*$", raw)
        seed_number = seed_match.group(1) if seed_match else ""
        if seed_match:
            raw = raw[:seed_match.start()].strip()

        raw = re.sub(r"^\[Generated[^\]]*\]\s*", "", raw).strip()
        raw = re.sub(r"\s+", " ", raw).strip(" -_")
        if not raw:
            raw = "generated"

        max_chars = 36
        if len(raw) > max_chars:
            raw = raw[:max_chars].rstrip(" -_") + "..."
        return raw, seed_number

    def _format_bbox(self, value: dict[str, int]) -> str:
        if not value:
            return "none"
        keys = ("x", "y", "width", "height", "x1", "y1", "x2", "y2")
        parts = [f"{key}={value[key]}" for key in keys if key in value]
        return ", ".join(parts) if parts else str(value)

    def _format_rect_size(self, row: RepairResultRow) -> str:
        if not row.rect_width and not row.rect_height:
            return "none"
        return f"{row.rect_width}x{row.rect_height}"

    def _compact_value(self, value: Any) -> str:
        if value is None:
            return "none"
        text = str(value).strip()
        if not text:
            return "none"
        if len(text) > 500:
            return text[:500] + "... (truncated)"
        return text

    def _line(self, label: str, value: object) -> str:
        return f"{label}: {self._safe_text(value)}"

    def _safe_text(self, value: object, fallback: str = "unknown") -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text if text else fallback