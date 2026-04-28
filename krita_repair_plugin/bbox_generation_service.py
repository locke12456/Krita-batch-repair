from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from krita_ai_metadata.sync_map_store import SyncRecord


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


@dataclass(slots=True)
class RepairGenerationResult:
    task: RepairGenerationTask
    success: bool = False
    output_png_bytes: bytes | None = None
    created_layer_id: str = ""
    created_layer_name: str = ""
    error: str = ""


class BBoxGenerationService:
    def generate_bbox_task(self, task: RepairGenerationTask) -> RepairGenerationResult:
        self.validate_task(task)
        raise RuntimeError(
            "BBox-only generation is not implemented; refusing full-canvas redraw."
        )

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