from __future__ import annotations

import json
from typing import Any

from .krita_core_adapter import find_krita_node_by_id


REPAIR_STATE_ANNOTATION_KEY = "repair_plugin/state.json"


def load_repair_state_payload(document_ref: Any) -> dict[str, Any]:
    if document_ref is None:
        return {}

    find_annotation = getattr(document_ref, "find_annotation", None)
    if not callable(find_annotation):
        return {}

    annotation = find_annotation(REPAIR_STATE_ANNOTATION_KEY)
    if annotation is None:
        return {}

    try:
        payload = bytes(annotation).decode("utf-8")
        raw = json.loads(payload)
    except Exception:
        return {}

    if isinstance(raw, dict):
        return raw
    return {}


def resolve_repair_replacement_id(
    document_ref: Any,
    old_layer_id: str | None,
) -> str | None:
    if not old_layer_id:
        return None

    target_id = str(old_layer_id)
    payload = load_repair_state_payload(document_ref)
    records = payload.get("records_by_canonical_layer_id", {})
    if not isinstance(records, dict):
        return None

    for record_data in records.values():
        if not isinstance(record_data, dict):
            continue
        replacements = record_data.get("replacements", {})
        if not isinstance(replacements, dict):
            continue

        current = replacements.get(target_id)
        seen = {target_id}
        while current and current not in seen:
            current = str(current)
            seen.add(current)
            next_id = replacements.get(current)
            if not next_id:
                return current
            current = str(next_id)

    return None


def resolve_repair_replacement_layer(
    document_ref: Any,
    old_layer_id: str | None,
) -> Any | None:
    replacement_id = resolve_repair_replacement_id(document_ref, old_layer_id)
    if not replacement_id:
        return None
    return find_krita_node_by_id(document_ref, replacement_id)