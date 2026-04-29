from __future__ import annotations

from typing import Any

from krita_ai_metadata.sync_map_store import SyncMapStore, SyncRecord

from .group_selection_model import RepairGroupRow
from .repair_compat import active_krita_document, all_krita_nodes, is_group_layer
from .repair_state_resolver import resolve_active_source
from .repair_state_store import RepairStateStore


class GroupSyncSource:
    def __init__(
        self,
        document_ref: Any | None = None,
        sync_map_store: SyncMapStore | None = None,
        repair_state_store: RepairStateStore | None = None,
    ) -> None:
        self.document_ref = document_ref or active_krita_document()
        if self.document_ref is None:
            raise RuntimeError("No active Krita document is available")
        self.sync_map_store = sync_map_store or SyncMapStore(self.document_ref)
        self.repair_state_store = repair_state_store or RepairStateStore(self.document_ref)

    def load_rows(self) -> list[RepairGroupRow]:
        self.sync_map_store.load()
        records = self.sync_map_store.all_records()
        try:
            self.repair_state_store.migrate_from_sync_records(records)
        except Exception:
            pass

        rows: list[RepairGroupRow] = []
        seen_records: set[tuple[str, str, str, int]] = set()

        for record in records:
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

            resolution = resolve_active_source(
                record=record,
                document_ref=self.document_ref,
                state_store=self.repair_state_store,
            )
            source_layers = [resolution.layer_ref] if resolution.resolved else []
            if resolution.unresolved_reason:
                warnings.append(resolution.unresolved_reason)

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

    def resolve_group_child_layers(self, group_layer: Any) -> list[Any]:
        """Resolve non-group child layers directly from a live group layer."""
        child_layers = getattr(group_layer, "child_layers", None)
        if isinstance(child_layers, list) and child_layers:
            return [
                layer
                for layer in child_layers
                if not self._is_group_node(layer)
            ]

        group_node = getattr(group_layer, "node", group_layer)
        try:
            raw_children = list(group_node.childNodes() or [])
        except Exception:
            raw_children = []

        if not raw_children:
            return []

        layers_by_id = {
            str(getattr(layer, "id_string", "") or ""): layer
            for layer in all_krita_nodes(self.document_ref)
        }
        result: list[Any] = []
        seen_ids: set[str] = set()

        for child in raw_children:
            child_id = ""
            unique_id = getattr(child, "uniqueId", None)
            if callable(unique_id):
                try:
                    child_id = str(unique_id().toString())
                except Exception:
                    child_id = ""
            if not child_id:
                child_id = str(getattr(child, "id_string", "") or "")
            if not child_id or child_id in seen_ids:
                continue
            layer = layers_by_id.get(child_id)
            if layer is None or self._is_group_node(layer):
                continue
            seen_ids.add(child_id)
            result.append(layer)

        return result

    def _select_active_source_layers(
        self,
        record: SyncRecord,
        source_layers: list[Any],
    ) -> list[Any]:
        """Return the single active source layer for repair refine."""
        if not source_layers:
            return []

        snapshot = dict(getattr(record, "params_snapshot", {}) or {})
        refine_state = dict(snapshot.get("repair_plugin_refine", {}) or {})
        current_layer_id = str(refine_state.get("current_layer_id", "") or "")
        if current_layer_id:
            for layer in source_layers:
                if str(getattr(layer, "id_string", "") or "") == current_layer_id:
                    return [layer]

        existing_ids = [
            str(layer_id or "")
            for layer_id in getattr(record, "layer_ids", []) or []
            if str(layer_id or "")
        ]
        if existing_ids:
            preferred_id = existing_ids[-1]
            for layer in source_layers:
                if str(getattr(layer, "id_string", "") or "") == preferred_id:
                    return [layer]

        return [source_layers[-1]]

    def _persist_group_record_from_layers(
        self,
        record: SyncRecord,
        group_layer: Any,
        source_layers: list[Any],
    ) -> None:
        """Persist the current live group mapping into SyncMapStore."""
        layer_ids = [
            str(getattr(layer, "id_string", "") or "")
            for layer in source_layers
            if str(getattr(layer, "id_string", "") or "")
        ]
        if not layer_ids:
            return

        group_id = str(getattr(group_layer, "id_string", "") or "")
        group_name = str(getattr(group_layer, "name", "") or "")

        record.target_type = "group"
        record.group_id = group_id or record.group_id
        record.group_name = group_name or record.group_name
        record.layer_ids = layer_ids

        try:
            self.sync_map_store.record_apply(record)
        except Exception:
            return

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
