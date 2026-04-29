from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .repair_compat import find_krita_node_by_id
from .repair_state_store import RepairStateRecord, RepairStateStore


@dataclass
class ActiveSourceResolution:
    canonical_layer_id: str = ""
    active_layer_id: str = ""
    layer_ref: Any | None = None
    state_record: RepairStateRecord | None = None
    unresolved_reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.layer_ref is not None and not self.unresolved_reason


def select_canonical_layer_id(record: Any) -> str | None:
    layer_ids = [str(layer_id) for layer_id in list(getattr(record, "layer_ids", []) or []) if str(layer_id)]
    if len(layer_ids) == 1:
        return layer_ids[0]
    return None


def _live_layer(document_ref: Any, layer_id: str | None) -> Any | None:
    if not layer_id:
        return None
    return find_krita_node_by_id(document_ref, str(layer_id))


def resolve_active_source(
    record: Any,
    document_ref: Any,
    state_store: RepairStateStore,
) -> ActiveSourceResolution:
    canonical_layer_id = select_canonical_layer_id(record)
    if canonical_layer_id is None:
        return ActiveSourceResolution(
            unresolved_reason="SyncRecord.layer_ids is missing or ambiguous."
        )

    state_record = state_store.resolve_by_canonical_layer_id(canonical_layer_id)
    if state_record is None:
        state_record = state_store.resolve_by_export_key(getattr(record, "export_key", ""))

    if state_record is not None:
        active_layer_id = state_record.active_layer_id or canonical_layer_id
        layer_ref = _live_layer(document_ref, active_layer_id)
        if layer_ref is not None:
            return ActiveSourceResolution(
                canonical_layer_id=canonical_layer_id,
                active_layer_id=active_layer_id,
                layer_ref=layer_ref,
                state_record=state_record,
            )

        replacement_id = state_store.resolve_replacement(active_layer_id)
        layer_ref = _live_layer(document_ref, replacement_id)
        if layer_ref is not None:
            return ActiveSourceResolution(
                canonical_layer_id=canonical_layer_id,
                active_layer_id=str(replacement_id),
                layer_ref=layer_ref,
                state_record=state_record,
            )

        return ActiveSourceResolution(
            canonical_layer_id=canonical_layer_id,
            active_layer_id=active_layer_id,
            state_record=state_record,
            unresolved_reason="RepairStateStore active layer is missing from the live Krita document.",
        )

    layer_ref = _live_layer(document_ref, canonical_layer_id)
    if layer_ref is not None:
        return ActiveSourceResolution(
            canonical_layer_id=canonical_layer_id,
            active_layer_id=canonical_layer_id,
            layer_ref=layer_ref,
            state_record=None,
        )

    replacement_id = state_store.resolve_replacement(canonical_layer_id)
    layer_ref = _live_layer(document_ref, replacement_id)
    if layer_ref is not None:
        return ActiveSourceResolution(
            canonical_layer_id=canonical_layer_id,
            active_layer_id=str(replacement_id),
            layer_ref=layer_ref,
            state_record=None,
        )

    return ActiveSourceResolution(
        canonical_layer_id=canonical_layer_id,
        unresolved_reason="No live layer matches SyncRecord.layer_ids fallback.",
    )


def resolve_replacement_layer(
    document_ref: Any,
    state_store: RepairStateStore,
    old_layer_id: str | None,
) -> Any | None:
    replacement_id = state_store.resolve_replacement(old_layer_id)
    if not replacement_id:
        return None
    return _live_layer(document_ref, replacement_id)