from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from krita_ai_metadata.sync_map_store import SyncRecord


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