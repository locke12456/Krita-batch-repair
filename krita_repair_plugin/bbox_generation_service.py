from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from krita_ai_metadata.sync_map_store import SyncRecord

from .repair_compat import add_repair_result_layer_to_group


@dataclass(slots=True)
class RepairGenerationTask:
    record: SyncRecord
    group_layer: Any
    source_layer: Any
    bbox: dict[str, int]
    crop_png_bytes: bytes
    prompt_text: str
    detector_mode: str
    detector_label: str
    base_positive: str = ""
    user_positive: str = ""
    base_negative: str = ""
    user_negative: str = ""


@dataclass(slots=True)
class RepairGenerationResult:
    task: RepairGenerationTask
    success: bool = False
    output_png_bytes: bytes | None = None
    created_layer_id: str = ""
    created_layer_name: str = ""
    error: str = ""


class BBoxGenerationService:
    def __init__(self, metadata_service: Any | None = None) -> None:
        self.metadata_service = metadata_service

    def build_generation_prompt(
        self,
        result_row: Any,
        base_positive: str = "",
        user_positive: str = "",
        base_negative: str = "",
        user_negative: str = "",
    ) -> tuple[str, str]:
        positive_parts = [
            str(getattr(result_row, "prompt_text", "") or "").strip(),
            str(user_positive or "").strip(),
            str(base_positive or "").strip(),
        ]
        negative_parts = [
            str(user_negative or "").strip(),
            str(base_negative or "").strip(),
        ]
        positive = ", ".join(part for part in positive_parts if part)
        negative = ", ".join(part for part in negative_parts if part)
        return positive, negative

    def task_from_result_row(
        self,
        row: Any,
        base_positive: str = "",
        user_positive: str = "",
        base_negative: str = "",
        user_negative: str = "",
    ) -> RepairGenerationTask:
        positive, negative = self.build_generation_prompt(
            row,
            base_positive=base_positive,
            user_positive=user_positive,
            base_negative=base_negative,
            user_negative=user_negative,
        )
        return RepairGenerationTask(
            record=row.record,
            group_layer=row.group_layer,
            source_layer=row.source_layer,
            bbox=dict(row.crop_bbox),
            crop_png_bytes=bytes(row.crop_png_bytes),
            prompt_text=positive,
            detector_mode=str(row.detector_mode),
            detector_label=str(row.detector_label),
            base_positive=base_positive,
            user_positive=user_positive,
            base_negative=negative,
            user_negative=user_negative,
        )

    def generate_result_row(self, row: Any) -> RepairGenerationResult:
        task = self.task_from_result_row(row)
        row.mark_generation_running()
        try:
            result = self.generate_bbox_task(task)
            if result.success:
                row.mark_generation_done(
                    result.created_layer_id,
                    result.created_layer_name,
                )
            else:
                row.mark_generation_failed(result.error)
            return result
        except Exception as exc:
            row.mark_generation_failed(str(exc))
            raise

    def generate_bbox_task(self, task: RepairGenerationTask) -> RepairGenerationResult:
        self.validate_task(task)

        # The explicit WorkflowInput path must be connected here when the target
        # Comfy workflow contract is finalized. Until then, do not call
        # Model.generate() or any native full-canvas handoff.
        raise RuntimeError(
            "BBox-only workflow enqueue is not wired yet; refusing full-canvas redraw."
        )

    def apply_result_to_group(
        self,
        task: RepairGenerationTask,
        output_png_bytes: bytes,
        layer_name: str = "",
    ) -> RepairGenerationResult:
        self.validate_task(task)
        if not output_png_bytes:
            return RepairGenerationResult(
                task=task,
                success=False,
                error="Output PNG bytes are empty.",
            )

        bbox = task.bbox
        name = layer_name or self._generated_layer_name(task)
        created_layer = add_repair_result_layer_to_group(
            document_ref=task.source_layer.document_ref,
            group_layer=task.group_layer,
            source_layer=task.source_layer,
            name=name,
            png_bytes=output_png_bytes,
            x=int(bbox.get("x", 0) or 0),
            y=int(bbox.get("y", 0) or 0),
        )
        result = RepairGenerationResult(
            task=task,
            success=True,
            output_png_bytes=output_png_bytes,
            created_layer_id=str(created_layer.id_string),
            created_layer_name=str(created_layer.name),
        )
        self._attach_generation_metadata(created_layer, result)
        return result

    def validate_task(self, task: RepairGenerationTask) -> None:
        if task.group_layer is None:
            raise RuntimeError("Group layer is required for bbox generation.")
        if task.source_layer is None:
            raise RuntimeError("Source layer is required for bbox generation.")
        if not task.crop_png_bytes:
            raise RuntimeError("BBox crop PNG bytes are required for bbox generation.")

        width = int(task.bbox.get("width", 0) or 0)
        height = int(task.bbox.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("BBox width and height must be positive.")

    def _generated_layer_name(self, task: RepairGenerationTask) -> str:
        mode = str(task.detector_mode or "repair").lower()
        source_name = str(getattr(task.source_layer, "name", "") or "source")
        if "head" in mode:
            prefix = "[Generated repair head]"
        elif "censor" in mode:
            prefix = "[Generated repair censor]"
        else:
            prefix = f"[Generated repair {mode}]"
        return f"{prefix} {source_name}"

    def _attach_generation_metadata(
        self,
        layer_ref: Any,
        result: RepairGenerationResult,
    ) -> None:
        service = self.metadata_service
        if service is None:
            return
        attach = getattr(service, "attach_group_batch_result_metadata", None)
        if callable(attach):
            task = result.task
            attach(
                layer_ref,
                {
                    "repair_plugin.generation_status": "done",
                    "repair_plugin.generation_result_layer_id": result.created_layer_id,
                    "repair_plugin.generation_result_layer_name": result.created_layer_name,
                    "repair_plugin.detector_bbox": task.bbox,
                    "repair_plugin.crop_bbox": task.bbox,
                    "repair_plugin.source_group_id": task.record.group_id,
                    "repair_plugin.source_group_name": task.record.group_name,
                    "repair_plugin.export_key": task.record.export_key,
                    "repair_plugin.prompt_text": task.prompt_text,
                },
            )
