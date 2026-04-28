from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from krita_ai_metadata.sync_map_store import SyncRecord

from .repair_compat import active_ai_model, add_repair_result_layer_to_group


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
    job_id: str = ""
    error: str = ""


class BBoxGenerationService:
    """Explicit bbox-only generation service.

    This service intentionally does not call Model.generate() or Model._prepare_workflow().
    It builds a bbox-local WorkflowInput directly, queues it through the active model's
    client/job queue, then applies the returned image into the original group layer.
    """

    def __init__(
        self,
        metadata_service: Any | None = None,
        model_resolver: Callable[[], Any] | None = None,
        on_row_finished: Callable[[Any, RepairGenerationResult], None] | None = None,
    ) -> None:
        self.metadata_service = metadata_service
        self.model_resolver = model_resolver or active_ai_model
        self.on_row_finished = on_row_finished

    def build_generation_prompt(
        self,
        result_row: Any,
        base_positive: str = "",
        user_positive: str = "",
        base_negative: str = "",
        user_negative: str = "",
    ) -> tuple[str, str]:
        """Insert bbox prompt before user/base positive prompt and preserve negative prompt."""
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
        """Queue bbox generation for one result row and return immediately.

        Completion is handled asynchronously by _enqueue_and_watch(...), which updates
        the row and applies the generated layer inside the original group.
        """
        task = self.task_from_result_row(row)
        row.mark_generation_running()
        try:
            return self.enqueue_task(task, row)
        except Exception as exc:
            row.mark_generation_failed(str(exc))
            raise

    def enqueue_task(self, task: RepairGenerationTask, row: Any | None = None) -> RepairGenerationResult:
        self.validate_task(task)
        model = self.resolve_model()
        workflow_input, job_params = self.build_workflow_input(task, model)
        self._run_async(self._enqueue_and_watch(model, task, workflow_input, job_params, row))
        return RepairGenerationResult(task=task, success=True)

    def resolve_model(self) -> Any:
        """Resolve an active connected krita-ai-diffusion model."""
        model = self.model_resolver()
        if model is None:
            raise RuntimeError("No active krita-ai-diffusion model is available.")

        connection = getattr(model, "_connection", None)
        if connection is None:
            raise RuntimeError("Active ai-diffusion model has no connection.")

        try:
            from ai_diffusion.connection import ConnectionState
        except Exception as exc:
            raise RuntimeError(f"ai-diffusion connection API unavailable: {exc}") from exc

        if getattr(connection, "state", None) is not ConnectionState.connected:
            raise RuntimeError("Krita AI Diffusion is not connected to ComfyUI.")
        if getattr(connection, "client_if_connected", None) is None:
            raise RuntimeError("Connected ComfyUI client is unavailable.")
        if not callable(getattr(model, "_enqueue_job", None)):
            raise RuntimeError("Active ai-diffusion model does not expose _enqueue_job().")
        return model

    def build_workflow_input(self, task: RepairGenerationTask, model: Any) -> tuple[Any, Any]:
        """Build explicit bbox-local WorkflowInput and document-space JobParams."""
        self.validate_task(task)

        try:
            from copy import copy
            from ai_diffusion import workflow
            from ai_diffusion.api import (
                ConditioningInput,
                ExtentInput,
                FillMode,
                ImageInput,
                InpaintMode,
                InpaintParams,
                WorkflowInput,
                WorkflowKind,
            )
            from ai_diffusion.client import resolve_arch
            from ai_diffusion.files import FileLibrary
            from ai_diffusion.image import Bounds, Extent, Image
            from ai_diffusion.jobs import JobParams
            from ai_diffusion.settings import settings
            from ai_diffusion.util import unique
        except Exception as exc:
            raise RuntimeError(f"ai-diffusion generation API unavailable: {exc}") from exc

        bbox = self._normalized_bbox(task.bbox)
        crop_image = Image.from_bytes(task.crop_png_bytes, "PNG")
        expected_extent = Extent(bbox["width"], bbox["height"])
        if crop_image.extent != expected_extent:
            raise RuntimeError(
                "BBox crop PNG extent does not match crop_bbox: "
                f"png={crop_image.width}x{crop_image.height}, "
                f"bbox={bbox['width']}x{bbox['height']}."
            )

        local_bounds = Bounds(0, 0, bbox["width"], bbox["height"])
        doc_bounds = Bounds(bbox["x"], bbox["y"], bbox["width"], bbox["height"])
        mask = self.build_bbox_local_mask(local_bounds)

        connection = getattr(model, "_connection")
        client = connection.client
        style = getattr(model, "active_style", None) or getattr(model, "style", None)
        if style is None:
            raise RuntimeError("Active ai-diffusion style is unavailable.")

        seed = int(getattr(model, "seed", 0) or 0)
        if not bool(getattr(model, "fixed_seed", False)):
            seed = workflow.generate_seed()

        strength = float(getattr(model, "strength", 1.0) or 1.0)
        conditioning = ConditioningInput(task.prompt_text, task.base_negative)
        conditioning.language = str(getattr(model, "prompt_translation_language", "") or "")

        checkpoint = copy(style.get_models(client.models.checkpoints))
        checkpoint.version = resolve_arch(style, client)

        prepared = workflow.prepare_prompts(
            conditioning,
            style,
            seed,
            checkpoint.version,
            InpaintMode.fill,
            FileLibrary.instance(),
        )

        inpaint = InpaintParams(
            InpaintMode.fill,
            local_bounds,
            fill=FillMode.neutral,
            grow=0,
            feather=0,
            blend=0,
        )

        native_inpaint = getattr(model, "inpaint", None)
        inpaint.use_inpaint_model = bool(getattr(native_inpaint, "use_inpaint", False))
        inpaint.use_condition_mask = bool(getattr(native_inpaint, "use_prompt_focus", False))
        inpaint.use_reference = False

        perf = model._performance_settings(client)
        workflow_input = workflow.prepare(
            WorkflowKind.refine_region,
            crop_image,
            prepared.conditioning,
            style,
            seed,
            client.models,
            FileLibrary.instance(),
            perf,
            mask=mask,
            strength=strength,
            loras=prepared.loras,
            inpaint=inpaint,
            layer_count=1,
        )

        job_name = self._generated_layer_name(task)
        job_params = JobParams(
            doc_bounds,
            job_name,
            seed=seed,
            has_mask=True,
            inpaint_mode=InpaintMode.fill,
            metadata={
                "prompt": task.prompt_text,
                "negative_prompt": task.base_negative,
                "repair_plugin.result_id": str(getattr(task, "result_id", "") or ""),
                "repair_plugin.detector_mode": task.detector_mode,
                "repair_plugin.detector_label": task.detector_label,
                "repair_plugin.crop_bbox": dict(bbox),
                "repair_plugin.source_group_id": task.record.group_id,
                "repair_plugin.source_group_name": task.record.group_name,
                "repair_plugin.export_key": task.record.export_key,
                **prepared.metadata,
            },
        )
        if getattr(workflow_input, "models", None) is not None:
            job_params.set_style(style, workflow_input.models.checkpoint)

        return workflow_input, job_params

    def build_bbox_local_mask(self, local_bounds: Any) -> Any:
        """Build a full-white bbox-local mask."""
        from ai_diffusion.image import Mask

        return Mask.rectangle(local_bounds, local_bounds)

    async def _enqueue_and_watch(
        self,
        model: Any,
        task: RepairGenerationTask,
        workflow_input: Any,
        job_params: Any,
        row: Any | None,
    ) -> None:
        from ai_diffusion.jobs import JobKind, JobState

        result: RepairGenerationResult | None = None
        job = None
        job_id = ""

        try:
            # Use live_preview to avoid native preview/apply side effects in Model._finish_job().
            job = model.jobs.add(JobKind.live_preview, job_params)
            await model._enqueue_job(job, workflow_input, front=False)
            job_id = str(getattr(job, "id", "") or "")
            if row is not None:
                row.mark_generation_running(job_id)

            while getattr(job, "state", None) not in {JobState.finished, JobState.cancelled}:
                await asyncio.sleep(0.1)

            if getattr(job, "state", None) is not JobState.finished:
                raise RuntimeError("BBox generation job was cancelled or interrupted.")

            if len(job.results) <= 0:
                raise RuntimeError("BBox generation job finished without output image.")

            result = self.apply_image_result_to_group(task, job.results[0], job_id=job_id)
            if row is not None:
                row.mark_generation_done(
                    result.created_layer_id,
                    result.created_layer_name,
                    job_id,
                )

        except Exception as exc:
            result = RepairGenerationResult(
                task=task,
                success=False,
                job_id=job_id,
                error=str(exc),
            )
            if row is not None:
                row.mark_generation_failed(str(exc), job_id)
        finally:
            if row is not None and result is not None:
                self._notify_row_finished(row, result)

    def apply_image_result_to_group(
        self,
        task: RepairGenerationTask,
        output_image: Any,
        job_id: str = "",
    ) -> RepairGenerationResult:
        self.validate_output_extent(task, output_image)
        output_png_bytes = bytes(output_image.to_bytes())
        result = self.apply_result_to_group(task, output_png_bytes)
        result.job_id = str(job_id or "")
        return result

    def generate_bbox_task(self, task: RepairGenerationTask) -> RepairGenerationResult:
        """Compatibility entry point: queue the task without falling back to full canvas."""
        return self.enqueue_task(task)

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

        bbox = self._normalized_bbox(task.bbox)
        name = layer_name or self._generated_layer_name(task)
        document_ref = getattr(task.source_layer, "document_ref", None)
        if document_ref is None:
            raise RuntimeError("Source layer document_ref is required for bbox generation apply.")

        created_layer = add_repair_result_layer_to_group(
            document_ref=document_ref,
            group_layer=task.group_layer,
            source_layer=task.source_layer,
            name=name,
            png_bytes=output_png_bytes,
            x=bbox["x"],
            y=bbox["y"],
        )
        result = RepairGenerationResult(
            task=task,
            success=True,
            output_png_bytes=output_png_bytes,
            created_layer_id=str(getattr(created_layer, "id_string", "") or ""),
            created_layer_name=str(getattr(created_layer, "name", "") or ""),
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

        bbox = self._normalized_bbox(task.bbox)
        if bbox["width"] <= 0 or bbox["height"] <= 0:
            raise RuntimeError("BBox width and height must be positive.")

    def validate_output_extent(self, task: RepairGenerationTask, output_image: Any) -> None:
        bbox = self._normalized_bbox(task.bbox)
        extent = getattr(output_image, "extent", None)
        width = int(getattr(extent, "width", 0) or (extent[0] if extent else 0))
        height = int(getattr(extent, "height", 0) or (extent[1] if extent else 0))
        if width != bbox["width"] or height != bbox["height"]:
            raise RuntimeError(
                "Generated image extent does not match crop_bbox: "
                f"output={width}x{height}, bbox={bbox['width']}x{bbox['height']}."
            )

    def _normalized_bbox(self, bbox: dict[str, Any]) -> dict[str, int]:
        x = int(bbox.get("x", bbox.get("x1", 0)) or 0)
        y = int(bbox.get("y", bbox.get("y1", 0)) or 0)
        width = int(bbox.get("width", 0) or 0)
        height = int(bbox.get("height", 0) or 0)
        if width <= 0 and "x2" in bbox:
            width = max(0, int(bbox.get("x2", 0) or 0) - x)
        if height <= 0 and "y2" in bbox:
            height = max(0, int(bbox.get("y2", 0) or 0) - y)
        return {"x": x, "y": y, "width": width, "height": height}

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
                    "repair_plugin.generation_job_id": result.job_id,
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

    def _run_async(self, future: Any) -> None:
        try:
            from ai_diffusion import eventloop

            eventloop.run(future)
        except Exception:
            asyncio.create_task(future)

    def _notify_row_finished(self, row: Any, result: RepairGenerationResult) -> None:
        callback = self.on_row_finished
        if callable(callback):
            callback(row, result)
