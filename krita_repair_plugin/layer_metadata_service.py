"Layer-only metadata service for repair plugin candidate and result layers."

from __future__ import annotations

import json
from typing import Any

from . import metadata_schema as schema

try:
    from krita_ai_metadata.sync_map_store import SyncMapStore
except Exception:
    SyncMapStore = None  # type: ignore[assignment]


class LayerMetadataService:
    "Attach detector, prompt, and generation metadata without creating groups."

    def __init__(self) -> None:
        self._metadata_by_layer_id: dict[str, dict[str, Any]] = {}
        self._sync_store_by_document_id: dict[int, Any] = {}
        self._layer_ref_by_layer_id: dict[str, Any] = {}

    def attach_detector_metadata(self, layer_ref: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        "Attach detector metadata to a layer reference or layer id."
        return self._merge_metadata(layer_ref, "detector", metadata)

    def attach_prompt_metadata(self, layer_ref: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        "Attach prompt extraction metadata to a layer reference or layer id."
        return self._merge_metadata(layer_ref, "prompt", metadata)

    def attach_generation_metadata(self, layer_ref: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        "Attach generation handoff metadata to a layer reference or layer id."
        return self._merge_metadata(layer_ref, "generation", metadata)

    def attach_result_metadata(self, layer_ref: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        "Attach post-completion generation metadata to a generated result layer."
        return self._merge_metadata(layer_ref, "result", metadata)

    def metadata_for_layer(self, layer_ref: Any) -> dict[str, Any]:
        "Return a copy of metadata for a layer reference or layer id."
        layer_id = self._layer_id(layer_ref)
        return dict(self._metadata_by_layer_id.get(layer_id, {}))

    def apply_candidate_context(self, model: Any, metadata: dict[str, Any]) -> None:
        "Apply candidate context to the active native ai-diffusion model when possible."
        candidate_ids = metadata.get("candidate_layer_ids")
        if not isinstance(candidate_ids, list) or not candidate_ids:
            return
        try_set_preview_layer = getattr(model, "try_set_preview_layer", None)
        if callable(try_set_preview_layer):
            try_set_preview_layer(str(candidate_ids[0]))

    def _merge_metadata(
        self,
        layer_ref: Any,
        namespace: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        "Merge namespace metadata into the layer-only metadata payload."
        raw_layer_id = self._layer_id(layer_ref)
        resolved_layer_ref = self._resolve_layer_ref(layer_ref)
        layer_id = self._layer_id(resolved_layer_ref) or raw_layer_id
        self._remember_layer_ref(layer_id, resolved_layer_ref)
        layer_name = self._layer_name(resolved_layer_ref)
        current = dict(self._metadata_by_layer_id.get(layer_id, {}))
        if not current:
            current.update(schema.base_payload(layer_id, layer_name))
        current[namespace] = self._json_safe(metadata)
        current.update(self._flatten_known_fields(metadata))
        self._metadata_by_layer_id[layer_id] = current
        self._reuse_existing_sync_record(resolved_layer_ref, layer_id, current)
        self._write_layer_annotation(resolved_layer_ref, current)
        return dict(current)

    def _remember_layer_ref(self, layer_id: str, layer_ref: Any) -> None:
        "Remember live layer references for later string-only metadata updates."
        if not layer_id or isinstance(layer_ref, str):
            return
        self._layer_ref_by_layer_id[layer_id] = layer_ref

    def _resolve_layer_ref(self, layer_ref: Any) -> Any:
        "Resolve a cached live layer reference when only a layer id is supplied."
        if isinstance(layer_ref, str):
            return self._layer_ref_by_layer_id.get(layer_ref, layer_ref)
        return layer_ref

    def _reuse_existing_sync_record(
        self,
        layer_ref: Any,
        layer_id: str,
        payload: dict[str, Any],
    ) -> None:
        "Reuse an existing export-compatible sync record when one is present."
        if SyncMapStore is None:
            return

        document = self._document_for_layer(layer_ref)
        if document is None:
            return

        try:
            document_key = id(document)
            store = self._sync_store_by_document_id.get(document_key)
            if store is None:
                store = SyncMapStore(document)
                self._sync_store_by_document_id[document_key] = store
            record = store.resolve_layer(layer_id)
        except Exception:
            return

        if record is None:
            return

        try:
            snapshot = dict(getattr(record, "params_snapshot", {}) or {})
            repair_payload = dict(snapshot.get("repair_plugin_metadata", {}) or {})
            repair_payload.update(self._json_safe(payload))
            snapshot["repair_plugin_metadata"] = repair_payload
            record.params_snapshot = snapshot
            store.record_apply(record)
            payload["export_sync_record"] = self._json_safe(record.to_dict())
        except Exception:
            return

    def _document_for_layer(self, layer_ref: Any) -> Any:
        "Resolve the live Krita document from a layer reference when available."
        document_ref = getattr(layer_ref, "document_ref", None)
        document = getattr(document_ref, "document", None)
        if document is not None:
            return document

        document = getattr(layer_ref, "document", None)
        if document is not None:
            return document

        node = getattr(layer_ref, "node", None)
        document = getattr(node, "document", None)
        if callable(document):
            try:
                return document()
            except Exception:
                return None
        if document is not None:
            return document

        return None

    def _flatten_known_fields(self, metadata: dict[str, Any]) -> dict[str, Any]:
        "Copy stable top-level keys for export-friendly readers."
        allowed = {
            schema.KEY_SCHEMA_VERSION,
            schema.KEY_LAYER_ID,
            schema.KEY_LAYER_NAME,
            schema.KEY_DETECTOR_MODE,
            schema.KEY_DETECTOR_LABEL,
            schema.KEY_DETECTOR_BBOX,
            schema.KEY_DETECTOR_BBOX_COORDINATE_SPACE,
            schema.KEY_DETECTOR_SCORE,
            schema.KEY_DETECTOR_SELECTED,
            schema.KEY_DETECTOR_ACTIVE,
            schema.KEY_PROMPT_WORKFLOW,
            schema.KEY_PROMPT_TEXT,
            schema.KEY_PROMPT_EXTRACTED,
            schema.KEY_PROMPT_RAW_OUTPUT,
            schema.KEY_GENERATION_STATUS,
            schema.KEY_GENERATION_HANDOFF,
            schema.KEY_GENERATION_JOB_ID,
            schema.KEY_GENERATION_RESULT_INDEX,
            schema.KEY_GENERATION_JOB_PARAMS,
            schema.KEY_RESULT_LAYER_ID,
            schema.KEY_RESULT_LAYER_NAME,
            schema.KEY_CANDIDATE_LAYER_IDS,
            schema.KEY_ERROR_MESSAGE,
            "visible",
        }
        return {
            str(key): self._json_safe(value)
            for key, value in metadata.items()
            if key in allowed
        }

    def _write_layer_annotation(self, layer_ref: Any, payload: dict[str, Any]) -> None:
        "Best-effort write to a Krita node annotation when the API is available."
        node = getattr(layer_ref, "node", layer_ref)
        set_annotation = getattr(node, "setAnnotation", None)
        if not callable(set_annotation):
            return
        try:
            set_annotation(
                schema.REPAIR_METADATA_KEY,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )
        except Exception:
            return

    def _layer_id(self, layer_ref: Any) -> str:
        "Resolve a stable layer id from a wrapper, node, or string."
        if isinstance(layer_ref, str):
            return layer_ref
        value = getattr(layer_ref, "id_string", None)
        if value:
            return str(value)
        node = getattr(layer_ref, "node", None)
        value = getattr(node, "id", None)
        if callable(value):
            try:
                return str(value())
            except Exception:
                pass
        value = getattr(node, "uniqueId", None)
        if callable(value):
            try:
                return str(value())
            except Exception:
                pass
        return str(layer_ref)

    def _layer_name(self, layer_ref: Any) -> str:
        "Resolve a display layer name from a wrapper, node, or string."
        value = getattr(layer_ref, "name", None)
        if value:
            return str(value)
        node = getattr(layer_ref, "node", None)
        name = getattr(node, "name", None)
        if callable(name):
            try:
                return str(name())
            except Exception:
                return ""
        if name:
            return str(name)
        return ""

    def _json_safe(self, value: Any) -> Any:
        "Return a JSON-serializable value where possible."
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            if isinstance(value, dict):
                return {str(key): self._json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [self._json_safe(item) for item in value]
            return str(value)
