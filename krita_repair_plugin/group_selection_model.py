from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from krita_ai_metadata.sync_map_store import SyncRecord


def _extract_refine_prompt(params_snapshot: dict) -> str:
    """Extract prompt text from a SyncRecord params_snapshot for group refine."""
    if not params_snapshot:
        return ""
    prompt = str(params_snapshot.get("prompt", "") or "").strip()
    if prompt:
        return prompt
    prompt = str(params_snapshot.get("positive_prompt", "") or "").strip()
    if prompt:
        return prompt
    prompt = str(params_snapshot.get("name", "") or "").strip()
    if prompt:
        return prompt
    regions = params_snapshot.get("regions")
    if isinstance(regions, (list, dict)):
        region_list = regions if isinstance(regions, list) else list(regions.values())
        parts = []
        for region in region_list:
            if isinstance(region, dict):
                rp = str(region.get("positive", "") or region.get("prompt", "") or "").strip()
                if rp:
                    parts.append(rp)
        if parts:
            return ", ".join(parts)
    return ""


@dataclass(slots=True)
class RepairGroupRow:
    record: SyncRecord
    selected: bool = False
    active: bool = True
    group_layer: Any | None = None
    source_layers: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_layer_ids: list[str] = field(default_factory=list)
    detected_count: int = 0

    @property
    def export_key(self) -> str:
        return self.record.export_key

    @property
    def group_id(self) -> str | None:
        return self.record.group_id

    @property
    def group_name(self) -> str | None:
        return self.record.group_name

    @property
    def layer_ids(self) -> list[str]:
        return list(self.record.layer_ids)

    @property
    def sync_index(self) -> int:
        return self.record.sync_index

    @property
    def is_resolved(self) -> bool:
        return self.group_layer is not None

    @property
    def display_name(self) -> str:
        return self.group_name or self.export_key or f"sync-{self.sync_index:04d}"

    @property
    def refine_prompt(self) -> str:
        return _extract_refine_prompt(self.record.params_snapshot)

    @property
    def refine_eligible(self) -> bool:
        return (
            self.record.target_type == "group"
            and self.is_resolved
            and bool(self.refine_prompt)
        )

    @property
    def refine_reason(self) -> str:
        if self.record.target_type != "group":
            return "not a group record"
        if not self.is_resolved:
            return "group unresolved"
        if not self.refine_prompt:
            return "no prompt in params_snapshot"
        return "refine-ready"


class GroupSelectionModel:
    def __init__(self) -> None:
        self.rows: list[RepairGroupRow] = []

    def replace_rows(self, rows: Iterable[RepairGroupRow]) -> None:
        self.rows = list(rows)

    def selected_active_groups(self) -> list[RepairGroupRow]:
        return [
            row
            for row in self.rows
            if row.selected and row.active and row.is_resolved
        ]

    def select_all(self) -> None:
        for row in self.rows:
            if row.is_resolved:
                row.selected = True

    def clear_selected(self) -> None:
        for row in self.rows:
            row.selected = False

    def update_result(
        self,
        record: SyncRecord,
        created_layer_ids: list[str],
        warning: str = "",
    ) -> None:
        for row in self.rows:
            if row.record is record:
                row.created_layer_ids.extend(created_layer_ids)
                row.detected_count += len(created_layer_ids)
                if warning:
                    row.warnings.append(warning)
                return