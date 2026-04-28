"""Native krita-ai-diffusion generation handoff service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .detection_layer_selection_model import DetectionLayerRow, DetectionLayerSelectionModel
from .repair_compat import active_ai_model


PROMPT_MODE_REPLACE = "replace"
PROMPT_MODE_APPEND = "append"
PROMPT_MODE_METADATA_ONLY = "metadata-only"
GENERATION_STATUS_QUEUED = "queued"
GENERATION_STATUS_COMPLETED = "completed"
GENERATION_STATUS_FAILED = "failed"
RESULT_METADATA_HOOK_ATTR = "_krita_repair_plugin_result_metadata_hook"


@dataclass(frozen=True, slots=True)
class GenerationHandoffContext:
    """Metadata context prepared before handing off to native generation."""

    rows: tuple[DetectionLayerRow, ...]
    prompt_mode: str
    prompt_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationTriggerService:
    """Bridge selected active candidate rows into existing ai-diffusion generation."""

    def __init__(
        self,
        selection_model: DetectionLayerSelectionModel | None = None,
        metadata_service: Any | None = None,
        model_resolver: Any | None = None,
    ) -> None:
        self.selection_model = selection_model
        self.metadata_service = metadata_service
        self.model_resolver = model_resolver or active_ai_model

    def prepare_handoff(
        self,
        rows: Iterable[DetectionLayerRow],
        prompt_mode: str = PROMPT_MODE_METADATA_ONLY,
    ) -> GenerationHandoffContext:
        """Prepare selected active row context for native generation handoff."""
        normalized_mode = self._normalize_prompt_mode(prompt_mode)
        row_tuple = tuple(row for row in rows if row.selected and row.active)
        prompt_text = self._compose_prompt_text(row_tuple)
        metadata = {
            "prompt_mode": normalized_mode,
            "prompt_text": prompt_text,
            "candidate_layer_ids": [row.layer_id for row in row_tuple],
            "candidate_layers": [row.to_metadata() for row in row_tuple],
        }
        return GenerationHandoffContext(
            rows=row_tuple,
            prompt_mode=normalized_mode,
            prompt_text=prompt_text,
            metadata=metadata,
        )

    def trigger_selected(
        self,
        filter_mode: str | None = None,
        prompt_mode: str = PROMPT_MODE_METADATA_ONLY,
        queue_mode: str = "back",
        execute: bool = True,
    ) -> GenerationHandoffContext:
        """Prepare and optionally trigger native generation for selected active rows."""
        if self.selection_model is None:
            raise RuntimeError("selection_model is required")

        rows = self.selection_model.selected_active_rows(filter_mode)
        return self.trigger_rows(rows, prompt_mode, queue_mode, execute)

    def trigger_rows(
        self,
        rows: Iterable[DetectionLayerRow],
        prompt_mode: str = PROMPT_MODE_METADATA_ONLY,
        queue_mode: str = "back",
        execute: bool = True,
    ) -> GenerationHandoffContext:
        """Prepare and optionally trigger native generation for explicit rows."""
        context = self.prepare_handoff(rows, prompt_mode)
        if not context.rows:
            raise RuntimeError("No selected active candidate rows are available")

        self._write_handoff_metadata(context)

        if execute:
            model = self._active_model()
            self._apply_candidate_context(model, context)
            self._install_result_metadata_hook(model, context)
            self._call_native_generation(model, queue_mode)

        for row in context.rows:
            row.generation_status = GENERATION_STATUS_QUEUED

        return context

    def _active_model(self) -> Any:
        """Resolve the current ai-diffusion model through the shared wrapper."""
        model = self.model_resolver()
        if model is None:
            raise RuntimeError("No active krita-ai-diffusion model is available")
        return model

    def _apply_candidate_context(self, model: Any, context: GenerationHandoffContext) -> None:
        """Apply selected candidate context before calling native generation."""
        apply_context = getattr(self.metadata_service, "apply_candidate_context", None)
        if callable(apply_context):
            apply_context(model, context.metadata)

        primary = context.rows[0] if context.rows else None
        if primary is None:
            return

        try_set_preview_layer = getattr(model, "try_set_preview_layer", None)
        if callable(try_set_preview_layer):
            try_set_preview_layer(primary.layer_id)

    def _call_native_generation(self, model: Any, queue_mode: str) -> None:
        """Call verified native generation entry points only."""
        normalized_queue = str(queue_mode or "back").strip().lower()
        if normalized_queue == "replace":
            generate_replace = getattr(model, "generate_replace", None)
            if callable(generate_replace):
                generate_replace()
                return

        generate = getattr(model, "generate", None)
        if not callable(generate):
            raise RuntimeError("Active model does not expose generate()")
        generate()


    def _install_result_metadata_hook(
        self,
        model: Any,
        context: GenerationHandoffContext,
    ) -> None:
        """Attach result-layer metadata after native ai-diffusion applies a job result.

        ai-diffusion owns job execution and result-layer creation. The repair plugin
        therefore waits until Model.apply_generated_result(...) has actually created
        or updated the result layer, then writes metadata best-effort.
        """
        if self.metadata_service is None:
            return

        apply_generated_result = getattr(model, "apply_generated_result", None)
        if not callable(apply_generated_result):
            return

        existing_state = getattr(model, RESULT_METADATA_HOOK_ATTR, None)
        if isinstance(existing_state, dict) and existing_state.get("installed"):
            contexts = existing_state.setdefault("contexts", [])
            contexts.append(context)
            return

        state: dict[str, Any] = {
            "installed": True,
            "contexts": [context],
            "original_apply_generated_result": apply_generated_result,
        }

        def wrapped_apply_generated_result(job_id: str, index: int, *args: Any, **kwargs: Any) -> Any:
            result = apply_generated_result(job_id, index, *args, **kwargs)
            contexts = list(state.get("contexts") or [])
            active_context = contexts[-1] if contexts else context
            try:
                self._attach_result_metadata_from_job(model, job_id, index, active_context)
            except Exception:
                # Metadata attachment is best-effort and must never break native apply.
                pass
            return result

        setattr(model, "apply_generated_result", wrapped_apply_generated_result)
        setattr(model, RESULT_METADATA_HOOK_ATTR, state)

    def _attach_result_metadata_from_job(
        self,
        model: Any,
        job_id: str,
        index: int,
        context: GenerationHandoffContext,
    ) -> None:
        """Attach completed-generation metadata to layers created by a finished job."""
        service = self.metadata_service
        if service is None:
            return

        attach = getattr(service, "attach_result_metadata", None)
        if not callable(attach):
            attach = getattr(service, "attach_generation_metadata", None)
        if not callable(attach):
            return

        job = self._find_job(model, job_id)
        metadata = {
            "generation_status": GENERATION_STATUS_COMPLETED,
            "generation_job_id": str(job_id or ""),
            "generation_result_index": int(index),
            "generation_handoff": context.metadata,
            "candidate_layer_ids": [row.layer_id for row in context.rows],
            "generation_job_params": self._job_params_snapshot(job),
        }

        for layer in self._candidate_result_layers(model, job):
            layer_metadata = dict(metadata)
            layer_metadata["result_layer_id"] = self._layer_id(layer)
            layer_metadata["result_layer_name"] = self._layer_name(layer)
            attach(layer, layer_metadata)

    def _find_job(self, model: Any, job_id: str) -> Any | None:
        """Find an ai-diffusion job by id from the active model."""
        jobs = getattr(model, "jobs", None)
        find = getattr(jobs, "find", None)
        if callable(find):
            try:
                return find(job_id)
            except Exception:
                return None
        return None

    def _candidate_result_layers(self, model: Any, job: Any | None) -> list[Any]:
        """Return likely generated result layers after native apply_result completed."""
        layers_obj = getattr(model, "layers", None)
        if layers_obj is None:
            return []

        result: list[Any] = []
        active_layer = getattr(layers_obj, "active", None)
        if active_layer is not None:
            result.append(active_layer)

        expected_names = self._expected_result_layer_names(job)
        if expected_names:
            for layer in list(getattr(layers_obj, "images", []) or []):
                if self._layer_name(layer) in expected_names:
                    result.append(layer)

        return self._dedupe_layers(result)

    def _expected_result_layer_names(self, job: Any | None) -> set[str]:
        """Mirror common ai-diffusion result-layer names for metadata lookup."""
        if job is None:
            return set()

        params = getattr(job, "params", None)
        if params is None:
            return set()

        seed = str(getattr(params, "seed", "") or "")
        prompts = self._expected_prompts(params)
        result: set[str] = set()

        for prompt in prompts:
            if seed:
                result.add(f"[Generated] {prompt} ({seed})")
                result.add(f"{prompt} ({seed})")
                result.add(f"[Upscale] {prompt} ({seed})")
            else:
                result.add(f"[Generated] {prompt}")
                result.add(prompt)
                result.add(f"[Upscale] {prompt}")

        if getattr(params, "is_layered", False):
            base_prompt = self._trim_text(str(getattr(params, "name", "") or ""), 200)
            for layer_index in range(1, 17):
                if seed:
                    result.add(f"[Layer {layer_index}] {base_prompt} ({seed})")
                else:
                    result.add(f"[Layer {layer_index}] {base_prompt}")

        return result

    def _expected_prompts(self, params: Any) -> set[str]:
        """Return prompt text used by normal and region result-layer creation."""
        prompts: set[str] = set()
        prompt_name = self._trim_text(str(getattr(params, "name", "") or ""), 200)
        if prompt_name:
            prompts.add(prompt_name)

        for region in list(getattr(params, "regions", []) or []):
            region_prompt = str(getattr(region, "prompt", "") or "").strip()
            if region_prompt:
                prompts.add(region_prompt)

        return prompts

    def _job_params_snapshot(self, job: Any | None) -> dict[str, Any]:
        """Build a JSON-friendly snapshot from an ai-diffusion job."""
        if job is None:
            return {}

        params = getattr(job, "params", None)
        if params is None:
            return {}

        snapshot: dict[str, Any] = {
            "name": str(getattr(params, "name", "") or ""),
            "seed": getattr(params, "seed", 0),
            "metadata": dict(getattr(params, "metadata", {}) or {}),
            "has_mask": bool(getattr(params, "has_mask", False)),
            "is_layered": bool(getattr(params, "is_layered", False)),
        }

        bounds = getattr(params, "bounds", None)
        if bounds is not None:
            offset = getattr(bounds, "offset", None)
            extent = getattr(bounds, "extent", None)
            snapshot["bounds"] = {
                "offset": self._json_safe(offset),
                "extent": self._json_safe(extent),
            }

        regions = []
        for region in list(getattr(params, "regions", []) or []):
            regions.append(
                {
                    "layer_id": str(getattr(region, "layer_id", "") or ""),
                    "prompt": str(getattr(region, "prompt", "") or ""),
                    "bounds": self._json_safe(getattr(region, "bounds", None)),
                    "is_background": bool(getattr(region, "is_background", False)),
                }
            )
        snapshot["regions"] = regions
        return snapshot

    def _dedupe_layers(self, layers: Iterable[Any]) -> list[Any]:
        """Preserve order while removing duplicate layer references."""
        result: list[Any] = []
        seen: set[str] = set()
        for layer in layers:
            key = self._layer_id(layer) or str(id(layer))
            if key in seen:
                continue
            result.append(layer)
            seen.add(key)
        return result

    def _layer_id(self, layer: Any) -> str:
        """Return a best-effort layer id for ai-diffusion or Krita layer wrappers."""
        value = getattr(layer, "id_string", None)
        if value:
            return str(value)
        value = getattr(layer, "id", None)
        if value:
            return str(value)
        node = getattr(layer, "node", None)
        node_id = getattr(node, "id", None)
        if callable(node_id):
            try:
                return str(node_id())
            except Exception:
                return ""
        return str(id(layer))

    def _layer_name(self, layer: Any) -> str:
        """Return a best-effort layer display name."""
        value = getattr(layer, "name", None)
        if value:
            return str(value)
        node = getattr(layer, "node", None)
        node_name = getattr(node, "name", None)
        if callable(node_name):
            try:
                return str(node_name())
            except Exception:
                return ""
        if node_name:
            return str(node_name)
        return ""

    def _trim_text(self, text: str, max_length: int) -> str:
        """Small local equivalent for matching ai-diffusion's result-layer names."""
        value = str(text or "").strip()
        if len(value) <= max_length:
            return value
        return value[: max(0, max_length - 1)].rstrip() + "…"

    def _json_safe(self, value: Any) -> Any:
        """Return simple JSON-friendly values for metadata snapshots."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _write_handoff_metadata(self, context: GenerationHandoffContext) -> None:
        """Write generation handoff metadata through an optional metadata service."""
        service = self.metadata_service
        if service is None:
            return

        attach = getattr(service, "attach_generation_metadata", None)
        if not callable(attach):
            return

        for row in context.rows:
            attach(
                row.layer_id,
                {
                    "generation_status": GENERATION_STATUS_QUEUED,
                    "generation_handoff": context.metadata,
                    "row": row.to_metadata(),
                },
            )

    def _compose_prompt_text(self, rows: Iterable[DetectionLayerRow]) -> str:
        """Compose a stable prompt text from row prompt fields."""
        parts: list[str] = []
        seen: set[str] = set()
        for row in rows:
            text = str(row.prompt_text or "").strip()
            if not text or text in seen:
                continue
            parts.append(text)
            seen.add(text)
        return ", ".join(parts)

    def _normalize_prompt_mode(self, prompt_mode: str) -> str:
        """Normalize prompt mode and reject unsupported values."""
        value = str(prompt_mode or PROMPT_MODE_METADATA_ONLY).strip().lower()
        if value in {"metadata", "metadata_only"}:
            value = PROMPT_MODE_METADATA_ONLY
        if value not in {PROMPT_MODE_REPLACE, PROMPT_MODE_APPEND, PROMPT_MODE_METADATA_ONLY}:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
        return value