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
GENERATION_STATUS_FAILED = "failed"


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