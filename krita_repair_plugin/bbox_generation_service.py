from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from krita_ai_metadata.sync_map_store import SyncRecord

from .repair_compat import (
    QtCore,
    QtGui,
    active_ai_model,
    add_repair_result_layer_to_group,
    move_layer_immediately_above,
    render_node_projection,
)


@dataclass(slots=True)
class RepairGenerationTask:
    record: SyncRecord
    group_layer: Any
    source_layer: Any
    bbox: dict[str, int]
    detector_bbox: dict[str, int]
    crop_png_bytes: bytes
    prompt_text: str
    detector_mode: str
    detector_label: str
    prompt_type_prompt: str = ""
    base_positive: str = ""
    user_positive: str = ""
    base_negative: str = ""
    user_negative: str = ""
    attach_transparency_mask: bool = True


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

    _DEBUG = False

    def __init__(
        self,
        metadata_service: Any | None = None,
        model_resolver: Callable[[], Any] | None = None,
        on_row_finished: Callable[[Any, RepairGenerationResult], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.metadata_service = metadata_service
        self.model_resolver = model_resolver or active_ai_model
        self.on_row_finished = on_row_finished
        self.log_callback = log_callback

    def active_model_prompt_snapshot(self) -> tuple[str, str]:
        """Return current KAD UI positive and negative prompts without preparing a workflow."""
        try:
            model = self.model_resolver()
        except Exception:
            return "", ""
        if model is None:
            return "", ""

        regions = getattr(model, "active_regions", None) or getattr(model, "regions", None)
        if regions is None:
            return "", ""

        active = getattr(regions, "active_or_root", None)
        if active is None:
            return "", ""

        positive = str(getattr(active, "positive", "") or "").strip()
        negative = str(getattr(active, "negative", "") or "").strip()
        return positive, negative

    def build_generation_prompt(
        self,
        result_row: Any,
        base_positive: str = "",
        user_positive: str = "",
        base_negative: str = "",
        user_negative: str = "",
    ) -> tuple[str, str]:
        """Insert bbox prompt before user/base positive prompt and preserve negative prompt."""
        prompt_type_prompt = ""
        if bool(getattr(result_row, "prompt_type_applied", False)):
            prompt_type_prompt = str(getattr(result_row, "prompt_type_prompt", "") or "").strip()
        positive_parts = [
            prompt_type_prompt,
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

    def _prompt_type_prompt_for_row(self, result_row: Any) -> str:
        """Return a default prompt fragment from the effective detector row type."""
        effective_type = getattr(result_row, "effective_prompt_type", None)
        if callable(effective_type):
            prompt_type = effective_type()
        else:
            prompt_type = (
                str(getattr(result_row, "detector_label", "") or "")
                or str(getattr(result_row, "detector_mode", "") or "")
            )
        lowered = str(prompt_type or "").strip().lower()
        if "head" in lowered or "face" in lowered:
            return "detailed head repair, natural face structure"
        if "penis" in lowered:
            return "anatomically consistent penis repair"
        if "pussy" in lowered or "vagina" in lowered:
            return "anatomically consistent pussy repair"
        if "censor" in lowered or "mosaic" in lowered:
            return "remove censorship artifact, restore natural detail"
        if lowered and lowered != "all":
            return "localized repair, coherent texture and lighting"
        return ""

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
            detector_bbox=dict(getattr(row, "detector_bbox", {}) or {}),
            crop_png_bytes=bytes(row.crop_png_bytes),
            prompt_text=positive,
            prompt_type_prompt=(
                str(getattr(row, "prompt_type_prompt", "") or "").strip()
                if bool(getattr(row, "prompt_type_applied", False))
                else ""
            ),
            detector_mode=str(row.detector_mode),
            detector_label=str(row.detector_label),
            base_positive=base_positive,
            user_positive=user_positive,
            base_negative=negative,
            user_negative=user_negative,
        )

    def generate_result_row(
        self,
        row: Any,
        base_positive: str = "",
        user_positive: str = "",
        base_negative: str = "",
        user_negative: str = "",
    ) -> RepairGenerationResult:
        """Queue bbox generation for one result row and return immediately.

        Completion is handled asynchronously by _enqueue_and_watch(...), which updates
        the row and applies the generated layer inside the original group.
        """
        task = self.task_from_result_row(
            row,
            base_positive=base_positive,
            user_positive=user_positive,
            base_negative=base_negative,
            user_negative=user_negative,
        )
        row.mark_generation_running()
        try:
            return self.enqueue_task(task, row)
        except Exception as exc:
            row.mark_generation_failed(str(exc))
            raise

    def snapshot_group_crop(self, group_layer: Any, crop_bbox: dict[str, int], created_layer: Any | None = None) -> bytes | None:
        """Take a fresh snapshot of the group layer projection, cropped to bbox.

        Unlike the detection-time cache (which snapshots a single source layer),
        this captures the full group composite — including any previously applied
        generation result layers that sit at the top of the group.
        """
        try:
            rendered = render_node_projection(group_layer)
        except Exception as exc:
            print(f"[BBoxGenerationService] WARNING: group snapshot failed: {exc}")
            return None
        projection_bounds = rendered.bounds
        image_bytes = bytes(rendered.to_bytes())

        image = QtGui.QImage()
        if not image.loadFromData(image_bytes, "PNG"):
            return None

        # Resolve crop region from created_layer bounds or crop_bbox fallback.
        # created_layer was placed at force-crop position during detection,
        # so its Krita node bounds() reflects the correct crop area.
        if created_layer is not None:
            try:
                layer_node = getattr(created_layer, "node", created_layer)
                layer_rect = layer_node.bounds()
                bbox = {
                    "x": int(layer_rect.x()),
                    "y": int(layer_rect.y()),
                    "width": int(layer_rect.width()),
                    "height": int(layer_rect.height()),
                }
            except Exception as exc:
                print(f"[BBoxGenerationService] WARNING: created_layer bounds() "
                      f"failed, falling back to crop_bbox: {exc}")
                bbox = self._normalized_bbox(crop_bbox)
        else:
            bbox = self._normalized_bbox(crop_bbox)

        # Convert document-space bbox to projection-local coordinates
        offset_x = int(getattr(projection_bounds, "x", 0) or 0)
        offset_y = int(getattr(projection_bounds, "y", 0) or 0)
        x = bbox["x"] - offset_x
        y = bbox["y"] - offset_y
        width = bbox["width"]
        height = bbox["height"]

        # Clamp to image bounds (same pattern as GroupBatchDetectionService)
        if x < 0:
            width += x
            x = 0
        if y < 0:
            height += y
            y = 0
        width = min(width, int(image.width()) - x)
        height = min(height, int(image.height()) - y)
        if width <= 0 or height <= 0:
            if self._DEBUG and self.log_callback is not None:
                self.log_callback(
                    f"[snapshot] FAILED: empty after clamp ({width}x{height})"
                )
            return None

        if self._DEBUG and self.log_callback is not None:
            _cl = locals().get("created_layer")
            _cl_name = "None" if _cl is None else str(getattr(_cl, "name", "?"))
            _crop_arg = self._normalized_bbox(crop_bbox)
            self.log_callback(
                f"[snapshot] created_layer={_cl_name} | "
                f"resolved_bbox={bbox} | "
                f"crop_bbox_arg={_crop_arg} | "
                f"proj=({offset_x},{offset_y}) | "
                f"img_size={image.width()}x{image.height()} | "
                f"final=({x},{y},{width}x{height})"
            )

        crop = image.copy(x, y, width, height)
        if self._DEBUG and self.log_callback is not None:
            self.log_callback(
                f"[snapshot] QImage after copy: {crop.width()}x{crop.height()}"
            )
        data = QtCore.QByteArray()
        buffer = QtCore.QBuffer(data)
        open_mode = getattr(QtCore.QIODevice, "OpenModeFlag", None)
        if open_mode is not None and hasattr(open_mode, "WriteOnly"):
            buffer.open(open_mode.WriteOnly)
        else:
            buffer.open(QtCore.QIODevice.WriteOnly)
        crop.save(buffer, "PNG")
        buffer.close()
        return bytes(data)

    def enqueue_task(self, task: RepairGenerationTask, row: Any | None = None) -> RepairGenerationResult:
        self.validate_task(task)
        model = self.resolve_model()

        # Fresh snapshot: re-crop from group composite so the image
        # includes any previously applied generation result layers.
        # Pass created_layer for accurate force-crop bounds.
        _created_layer = getattr(row, "created_layer", None) if row is not None else None
        fresh_crop = self.snapshot_group_crop(task.group_layer, task.bbox, created_layer=_created_layer)
        if fresh_crop:
            task.crop_png_bytes = fresh_crop

        workflow_input, job_params = self.build_workflow_input(task, model)
        self._run_async(self._enqueue_and_watch(model, task, workflow_input, job_params, row))
        return RepairGenerationResult(task=task, success=True)


    def enqueue_batch_sequential(
        self,
        tasks_and_rows: list[tuple[RepairGenerationTask, Any]],
    ) -> None:
        """Queue tasks sequentially -- each waits for completion before next.

        Unlike the per-row fire-and-forget path (enqueue_task / generate_result_row),
        this coroutine processes one task at a time so that each fresh snapshot
        captures previously applied generation results in the group composite.
        """
        if not tasks_and_rows:
            return
        model = self.resolve_model()
        self._run_async(self._generate_batch_sequential(model, tasks_and_rows))

    async def _generate_batch_sequential(
        self,
        model: Any,
        tasks_and_rows: list[tuple[RepairGenerationTask, Any]],
    ) -> None:
        """Process tasks one-by-one: snapshot -> build -> enqueue -> await -> apply -> next."""
        from ai_diffusion.jobs import JobKind, JobState

        for idx, (task, row) in enumerate(tasks_and_rows):
            result: RepairGenerationResult | None = None
            job_id = ""
            try:
                # 1. Fresh snapshot (captures any previously applied layers)
                # Pass created_layer for accurate force-crop bounds.
                _created_layer = getattr(row, "created_layer", None) if row is not None else None
                fresh_crop = self.snapshot_group_crop(task.group_layer, task.bbox, created_layer=_created_layer)
                if fresh_crop:
                    task.crop_png_bytes = fresh_crop

                # 2. Build workflow input
                self.validate_task(task)
                workflow_input, job_params = self.build_workflow_input(task, model)

                # 3. Enqueue and await completion
                if row is not None:
                    row.mark_generation_running()
                job = model.jobs.add(JobKind.live_preview, job_params)
                await model._enqueue_job(job, workflow_input, front=False)
                job_id = str(getattr(job, "id", "") or "")
                if row is not None:
                    row.mark_generation_running(job_id)

                while getattr(job, "state", None) not in {
                    JobState.finished,
                    JobState.cancelled,
                }:
                    await asyncio.sleep(0.1)

                if getattr(job, "state", None) is not JobState.finished:
                    raise RuntimeError("BBox generation job was cancelled or interrupted.")
                if len(job.results) <= 0:
                    raise RuntimeError("BBox generation job finished without output image.")

                # 4. Apply result to group (includes move-to-top)
                result = self.apply_image_result_to_group(
                    task, job.results[0], job_id=job_id,
                )
                if row is not None:
                    row.mark_generation_done(
                        result.created_layer_id,
                        result.created_layer_name,
                        job_id,
                    )
                    row.generation_order = idx

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
                ImageInput,
                InpaintMode,
                InpaintParams,
                WorkflowInput,
                WorkflowKind,
            )
            from ai_diffusion.client import resolve_arch
            from ai_diffusion.files import FileLibrary
            from ai_diffusion.image import Bounds, Extent, Image, multiple_of
            from ai_diffusion.jobs import JobParams
            from ai_diffusion.settings import settings
            from ai_diffusion.util import unique
        except Exception as exc:
            raise RuntimeError(f"ai-diffusion generation API unavailable: {exc}") from exc

        bbox = self._normalized_bbox(task.bbox)
        crop_image = Image.from_bytes(task.crop_png_bytes, "PNG")
        expected_extent = Extent(bbox["width"], bbox["height"])
        if self._DEBUG and self.log_callback is not None:
            self.log_callback(
                f"[build_workflow] crop_png={crop_image.width}x{crop_image.height} | "
                f"task.bbox={bbox} | bytes_len={len(task.crop_png_bytes)}"
            )
        if crop_image.extent != expected_extent:
            raise RuntimeError(
                "BBox crop PNG extent does not match crop_bbox: "
                f"png={crop_image.width}x{crop_image.height}, "
                f"bbox={bbox['width']}x{bbox['height']}."
            )

        # Comfy / latent workflows require diffusion-safe dimensions. If the bbox is
        # 300x300, internal nodes may turn image tensors into 296x296 while the mask
        # stays 300x300. Normalize the generation canvas to a multiple of 16 so image
        # and mask share one stable size, then scale the final output back to crop_bbox.
        generation_extent = Extent(
            multiple_of(expected_extent.width, 16),
            multiple_of(expected_extent.height, 16),
        )
        if generation_extent != expected_extent:
            crop_image = Image.scale(crop_image, generation_extent)

        local_bounds = Bounds(0, 0, generation_extent.width, generation_extent.height)
        doc_bounds = Bounds(bbox["x"], bbox["y"], bbox["width"], bbox["height"])
        whole_crop_refine = self._is_whole_crop_refine_task(task)

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

        workflow_kind = WorkflowKind.refine if whole_crop_refine else WorkflowKind.refine_region
        prompt_inpaint_mode = None if whole_crop_refine else InpaintMode.fill

        prepared = workflow.prepare_prompts(
            conditioning,
            style,
            seed,
            checkpoint.version,
            prompt_inpaint_mode,
            FileLibrary.instance(),
        )

        mask = None
        inpaint = None
        job_has_mask = False
        job_inpaint_mode = None

        if not whole_crop_refine:
            mask = self.build_detector_local_mask(
                crop_bbox=bbox,
                detector_bbox=task.detector_bbox,
                generation_extent=generation_extent,
                original_extent=expected_extent,
            )
            # --- Mask feather / grow / blend via stable Method A defaults ---
            # Keep the old bbox-local behavior: compute reasonable mask process
            # parameters without switching to KAD's full native selection API.
            _diag = (generation_extent.width ** 2 + generation_extent.height ** 2) ** 0.5
            _feather = max(int(0.10 * _diag), 32)
            _grow = 4 + _feather // 2
            _blend = min(25, _grow + _feather // 2)

            inpaint = InpaintParams(
                InpaintMode.fill,
                local_bounds,
                grow=_grow,
                feather=_feather,
                blend=_blend,
            )

            native_inpaint = getattr(model, "inpaint", None)
            inpaint.use_inpaint_model = bool(getattr(native_inpaint, "use_inpaint", False))
            inpaint.use_condition_mask = bool(getattr(native_inpaint, "use_prompt_focus", False))
            inpaint.use_reference = False
            job_has_mask = True
            job_inpaint_mode = InpaintMode.fill

        perf = model._performance_settings(client)
        workflow_input = workflow.prepare(
            workflow_kind,
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
            has_mask=job_has_mask,
            inpaint_mode=job_inpaint_mode,
            metadata={
                "prompt": task.prompt_text,
                "negative_prompt": task.base_negative,
                "repair_plugin.prompt_type_prompt": task.prompt_type_prompt,
                "repair_plugin.result_id": str(getattr(task, "result_id", "") or ""),
                "repair_plugin.detector_mode": task.detector_mode,
                "repair_plugin.detector_label": task.detector_label,
                "repair_plugin.detector_bbox": dict(task.detector_bbox),
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
        """Build a full-white bbox-local mask fallback."""
        from ai_diffusion.image import Mask

        return Mask.rectangle(local_bounds, local_bounds)

    def build_detector_local_mask(
        self,
        crop_bbox: dict[str, Any],
        detector_bbox: dict[str, Any],
        generation_extent: Any,
        original_extent: Any,
    ) -> Any:
        """Build a mask for detector_bbox inside the bbox-local crop canvas.

        crop_bbox is document-space and defines the crop image.
        detector_bbox is document-space and defines the actual inpaint target.
        generation_extent may be scaled to a diffusion-safe multiple of 16, so the
        detector-local mask must be scaled by the same factor.
        """
        from ai_diffusion.image import Bounds, Mask

        context = Bounds(0, 0, int(generation_extent.width), int(generation_extent.height))
        if not detector_bbox:
            return self.build_bbox_local_mask(context)

        crop = self._normalized_bbox(crop_bbox)
        detector = self._normalized_bbox(detector_bbox)
        if detector["width"] <= 0 or detector["height"] <= 0:
            return self.build_bbox_local_mask(context)

        scale_x = float(generation_extent.width) / max(1, int(original_extent.width))
        scale_y = float(generation_extent.height) / max(1, int(original_extent.height))
        local_x = round((detector["x"] - crop["x"]) * scale_x)
        local_y = round((detector["y"] - crop["y"]) * scale_y)
        local_w = round(detector["width"] * scale_x)
        local_h = round(detector["height"] * scale_y)

        mask_bounds = Bounds(local_x, local_y, max(1, local_w), max(1, local_h))
        mask_bounds = Bounds.restrict(mask_bounds, context)
        if mask_bounds.width <= 0 or mask_bounds.height <= 0:
            return self.build_bbox_local_mask(context)
        return Mask.rectangle(mask_bounds, context)

    def _is_whole_crop_refine_task(self, task: RepairGenerationTask) -> bool:
        """Return True when the crop itself is the full local repair target.

        When crop_bbox == detector_bbox, the cropped PNG already contains exactly
        the target local image. Running refine_region would create a full-image
        mask and force the workflow down the inpaint path. That is the wrong
        semantic: there is no surrounding context area to preserve. Use ordinary
        refine for this case and only use refine_region when crop_bbox is larger
        than detector_bbox.
        """
        crop = self._normalized_bbox(task.bbox)
        detector = self._normalized_bbox(task.detector_bbox)
        if not detector:
            return True

        return (
            int(crop.get("x", 0)) == int(detector.get("x", 0))
            and int(crop.get("y", 0)) == int(detector.get("y", 0))
            and int(crop.get("width", 0)) == int(detector.get("width", 0))
            and int(crop.get("height", 0)) == int(detector.get("height", 0))
        )

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
        try:
            from ai_diffusion.image import Extent, Image
        except Exception as exc:
            raise RuntimeError(f"ai-diffusion image API unavailable: {exc}") from exc

        bbox = self._normalized_bbox(task.bbox)
        expected_extent = Extent(bbox["width"], bbox["height"])
        extent = getattr(output_image, "extent", None)
        width = int(getattr(extent, "width", 0) or (extent[0] if extent else 0))
        height = int(getattr(extent, "height", 0) or (extent[1] if extent else 0))
        if self._DEBUG and self.log_callback is not None:
            self.log_callback(
                f"[apply_result] output={width}x{height} | "
                f"expected={expected_extent.width}x{expected_extent.height}"
            )
        if width <= 0 or height <= 0:
            raise RuntimeError("Generated image extent is invalid.")
        if width != expected_extent.width or height != expected_extent.height:
            # This is an explicit safe scale back from the internal multiple-of-16
            # generation extent to the original document-space crop bbox extent.
            output_image = Image.scale(output_image, expected_extent)

        output_png_bytes = bytes(output_image.to_bytes())
        result = self.apply_result_to_group(task, output_png_bytes)
        result.job_id = str(job_id or "")
        return result

    def generate_bbox_task(self, task: RepairGenerationTask) -> RepairGenerationResult:
        """Compatibility entry point: queue the task without falling back to full canvas."""
        return self.enqueue_task(task)

    def _move_layer_to_group_top(
        self,
        document_ref: Any,
        created_layer: Any,
        group_layer: Any,
    ) -> None:
        """Move created_layer to the top of group_layer's child stack.

        Generation results must be at the top so that subsequent group
        snapshots (via render_node_projection) include previously generated
        layers in the composite.

        Strategy:
        1. Primary: use move_layer_immediately_above (krita_core_adapter wrapper)
        2. Fallback: use Krita Node API (removeChildNode + addChildNode)
        """
        try:
            group_node = getattr(group_layer, "node", group_layer)
            created_node = getattr(created_layer, "node", created_layer)
            children = group_node.childNodes()
            if not children:
                return
            # childNodes() order: bottom -> top; children[-1] is topmost
            topmost = children[-1]
            if topmost is created_node:
                return  # already at top

            moved = False

            # Strategy 1: wrapped adapter API
            try:
                move_layer_immediately_above(
                    document_ref, created_layer, topmost,
                )
                moved = True
            except Exception as move_exc:
                print(
                    f"[BBoxGenerationService] move_layer_immediately_above "
                    f"failed, trying Node API fallback: {move_exc}"
                )

            # Strategy 2: Krita Node API fallback
            if not moved:
                try:
                    group_node.removeChildNode(created_node)
                    # Re-fetch children after removal
                    children = group_node.childNodes()
                    if children:
                        # addChildNode(child, above) places child above the ref
                        group_node.addChildNode(created_node, children[-1])
                    else:
                        group_node.addChildNode(created_node, None)
                    moved = True
                except Exception as node_exc:
                    print(
                        f"[BBoxGenerationService] Node API fallback also "
                        f"failed: {node_exc}"
                    )

            if moved and callable(
                getattr(document_ref, "refresh_projection", None)
            ):
                document_ref.refresh_projection()

        except Exception as exc:
            print(
                f"[BBoxGenerationService] WARNING: could not move layer "
                f"to group top: {exc}"
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
        if self._DEBUG and self.log_callback is not None:
            _node = getattr(created_layer, "node", created_layer)
            _b = _node.bounds()
            self.log_callback(
                f"[apply] after setPixelData: "
                f"bounds=({_b.x()},{_b.y()},{_b.width()}x{_b.height()}) | "
                f"expected=({bbox['x']},{bbox['y']},{bbox['width']}x{bbox['height']})"
            )
        self._move_layer_to_group_top(document_ref, created_layer, task.group_layer)
        # Refine: never attach mask. Detection: follow task flag.
        is_refine = str(getattr(task, "detector_mode", "") or "").strip().lower() == "refine"
        if not is_refine and task.attach_transparency_mask:
            self._attach_inward_blur_transparency_mask(
                document_ref=document_ref,
                layer_ref=created_layer,
                bbox=bbox,
                blur_px=24,
            )
            if self._DEBUG and self.log_callback is not None:
                _node2 = getattr(created_layer, "node", created_layer)
                _b2 = _node2.bounds()
                self.log_callback(
                    f"[apply] after mask: "
                    f"bounds=({_b2.x()},{_b2.y()},{_b2.width()}x{_b2.height()})"
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

    def _attach_inward_blur_transparency_mask(
        self,
        document_ref: Any,
        layer_ref: Any,
        bbox: dict[str, int],
        blur_px: int = 24,
    ) -> None:
        """Attach a real Krita transparency mask child to the generated layer.

        This must visibly create a child node like:
            Generated layer
                Transparency Mask 1

        If the child mask cannot be verified, raise an error instead of silently
        continuing with an opaque rectangle.
        """
        width = max(1, int(bbox.get("width", 0) or 0))
        height = max(1, int(bbox.get("height", 0) or 0))
        x0 = int(bbox.get("x", 0) or 0)
        y0 = int(bbox.get("y", 0) or 0)
        blur = max(1, min(int(blur_px), max(1, min(width, height) // 2)))

        values: list[int] = []
        for y in range(height):
            for x in range(width):
                dx = 0
                if x < blur:
                    dx = blur - x
                elif x > width - blur - 1:
                    dx = x - (width - blur - 1)

                dy = 0
                if y < blur:
                    dy = blur - y
                elif y > height - blur - 1:
                    dy = y - (height - blur - 1)

                d = float((dx * dx + dy * dy) ** 0.5)
                alpha = 0 if d >= blur else int(255 * (1.0 - d / float(blur)))
                values.append(max(0, min(255, alpha)))

        document = getattr(document_ref, "document", None)
        create_mask = getattr(document, "createTransparencyMask", None)
        if not callable(create_mask):
            raise RuntimeError("Krita document does not expose createTransparencyMask().")

        layer_node = getattr(layer_ref, "node", layer_ref)
        add_child = getattr(layer_node, "addChildNode", None)
        child_nodes = getattr(layer_node, "childNodes", None)
        if not callable(add_child) or not callable(child_nodes):
            raise RuntimeError("Generated layer node cannot accept or report child mask nodes.")

        mask_node = create_mask("Transparency Mask 1")
        try:
            mask_node.setName("Transparency Mask 1")
        except Exception:
            pass

        set_pixel_data = getattr(mask_node, "setPixelData", None)
        if not callable(set_pixel_data):
            raise RuntimeError("Transparency mask node does not expose setPixelData().")

        # Transparency mask coordinates are document-space, matching the generated
        # layer pixel placement. The mask size itself never exceeds the generated image.
        set_pixel_data(QtCore.QByteArray(bytes(values)), x0, y0, width, height)

        add_child(mask_node, None)

        children = list(child_nodes() or [])
        mask_children = [child for child in children if str(getattr(child, "type", lambda: "")()).lower() == "transparencymask"]
        if mask_node not in children and not mask_children:
            raise RuntimeError("Transparency mask was created but is not attached under generated layer.")

        if callable(getattr(document_ref, "refresh_projection", None)):
            document_ref.refresh_projection()

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
                    "repair_plugin.detector_bbox": task.detector_bbox,
                    "repair_plugin.crop_bbox": task.bbox,
                    "repair_plugin.source_group_id": task.record.group_id,
                    "repair_plugin.source_group_name": task.record.group_name,
                    "repair_plugin.export_key": task.record.export_key,
                    "repair_plugin.prompt_text": task.prompt_text,
                    "repair_plugin.prompt_type_prompt": task.prompt_type_prompt,
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
