from __future__ import annotations

from typing import Any

from krita_ai_metadata.sync_map_store import SyncMapStore, SyncRecord

from .group_selection_model import RepairGroupRow
from .repair_compat import active_krita_document, all_krita_nodes, is_group_layer


class GroupSyncSource:
    def __init__(
        self,
        document_ref: Any | None = None,
        sync_map_store: SyncMapStore | None = None,
    ) -> None:
        self.document_ref = document_ref or active_krita_document()
        if self.document_ref is None:
            raise RuntimeError("No active Krita document is available")
        self.sync_map_store = sync_map_store or SyncMapStore(self.document_ref)

    def load_rows(self) -> list[RepairGroupRow]:
        self.sync_map_store.load()
        rows: list[RepairGroupRow] = []
        seen_records: set[tuple[str, str, str, int]] = set()

        for record in self.sync_map_store.all_records():
            if record.target_type != "group":
                continue

            record_key = self._record_key(record)
            if record_key in seen_records:
                continue
            seen_records.add(record_key)

            warnings: list[str] = []
            group_layer = self.resolve_group_layer(record)
            if group_layer is None:
                warnings.append(
                    f"Group unresolved: {record.group_name or record.group_id or record.export_key}"
                )

            source_layers = self.resolve_source_layers(record)
            missing_count = len(record.layer_ids) - len(source_layers)
            if missing_count > 0:
                warnings.append(f"{missing_count} source layer(s) from SyncRecord were not found.")

            rows.append(
                RepairGroupRow(
                    record=record,
                    group_layer=group_layer,
                    source_layers=source_layers,
                    warnings=warnings,
                )
            )

        return rows

    def resolve_group_layer(self, record: SyncRecord) -> Any | None:
        for layer in all_krita_nodes(self.document_ref):
            if (
                record.group_id
                and layer.id_string == record.group_id
                and self._is_group_node(layer)
            ):
                return layer

        for layer in all_krita_nodes(self.document_ref):
            if (
                record.group_name
                and layer.name == record.group_name
                and self._is_group_node(layer)
            ):
                return layer

        return None

    def resolve_source_layers(self, record: SyncRecord) -> list[Any]:
        wanted = set(record.layer_ids)
        if not wanted:
            return []

        return [
            layer
            for layer in all_krita_nodes(self.document_ref)
            if layer.id_string in wanted and not self._is_group_node(layer)
        ]

    def _record_key(self, record: SyncRecord) -> tuple[str, str, str, int]:
        return (
            str(record.group_id or ""),
            str(record.group_name or ""),
            str(record.export_key or ""),
            int(record.sync_index or 0),
        )

    def _is_group_node(self, layer: Any) -> bool:
        try:
            if is_group_layer(layer):
                return True
        except Exception:
            pass

        node = getattr(layer, "node", layer)
        try:
            if is_group_layer(node):
                return True
        except Exception:
            pass

        layer_type = str(getattr(layer, "type", "") or "").lower()
        node_type = getattr(node, "type", None)
        if callable(node_type):
            try:
                layer_type = str(node_type() or "").lower()
            except Exception:
                pass

        return layer_type in {"grouplayer", "group_layer", "group"}
