from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from krita_ai_metadata.sync_map_store import SyncRecord

from .detection_service import DetectionOptions, expand_bbox_to_forced_rect
from .group_selection_model import RepairGroupRow
from .repair_result_model import RepairResultRow
from .plugin_detector import BoundingBox, DetectionResult, LayerProjectionInput
from .repair_compat import (
    QtCore,
    QtGui,
    add_repair_result_layer_to_group,
    render_node_projection,
)


@dataclass(slots=True)
class GroupDetectionReport:
    record: SyncRecord
    group_name: str
    export_key: str
    source_layer_id: str
    source_layer_name: str
    detector_mode: str
    bbox: dict[str, int]
    created_layer_id: str | None = None
    created_layer_name: str | None = None
    prompt_text: str = ""
    result_row: RepairResultRow | None = None
    warnings: list[str] = field(default_factory=list)
    error: str = ""


class GroupBatchDetectionService:
    def __init__(
        self,
        detector_manager: Any,
        metadata_service: Any,
        prompt_service: Any | None = None,
        result_selection_model: Any | None = None,
    ) -> None:
        self.detector_manager = detector_manager
        self.metadata_service = metadata_service
        self.prompt_service = prompt_service
        self.result_selection_model = result_selection_model

    def detect_rows(
        self,
        rows: list[RepairGroupRow],
        mode: str,
        options: DetectionOptions,
        extract_prompts: bool = False,
    ) -> list[GroupDetectionReport]:
        reports: list[GroupDetectionReport] = []
        for row in rows:
            reports.extend(self.detect_one_row(row, mode, options, extract_prompts))
        result_rows = [report.result_row for report in reports if report.result_row is not None]
        if self.result_selection_model is not None and result_rows:
            append_rows = getattr(self.result_selection_model, "append_rows", None)
            if callable(append_rows):
                append_rows(result_rows)
        return reports

    def detect_one_row(
        self,
        row: RepairGroupRow,
        mode: str,
        options: DetectionOptions,
        extract_prompts: bool = False,
    ) -> list[GroupDetectionReport]:
        if row.record.target_type != "group":
            return []
        if row.group_layer is None:
            row.warnings.append("Group is unresolved; skipped.")
            return []
        if not row.source_layers:
            row.warnings.append("No source layers resolved from SyncRecord.layer_ids; skipped.")
            return []

        reports: list[GroupDetectionReport] = []
        for source_layer in row.source_layers:
            reports.extend(
                self._detect_source_layer(
                    row,
                    source_layer,
                    mode,
                    options,
                    extract_prompts,
                )
            )
        return reports

    def _detect_source_layer(
        self,
        row: RepairGroupRow,
        source_layer: Any,
        mode: str,
        options: DetectionOptions,
        extract_prompts: bool = False,
    ) -> list[GroupDetectionReport]:
        rendered = render_node_projection(source_layer)
        projection_bounds = rendered.bounds
        image_bytes = bytes(rendered.to_bytes())

        projection = LayerProjectionInput(
            layer_id=str(source_layer.id_string),
            layer_name=str(source_layer.name),
            bounds=projection_bounds,
            image_bytes=image_bytes,
            coordinate_space="projection",
        )

        raw_results = self.detector_manager.detect_layer(
            mode,
            projection,
            options.to_detector_options(),
        )

        reports: list[GroupDetectionReport] = []
        for index, result in enumerate(raw_results, start=1):
            try:
                report = self._create_result_layer(
                    row=row,
                    source_layer=source_layer,
                    source_image_bytes=image_bytes,
                    projection_bounds=projection_bounds,
                    result=result,
                    index=index,
                    options=options,
                    extract_prompts=extract_prompts,
                )
                reports.append(report)
            except Exception as exc:
                reports.append(
                    GroupDetectionReport(
                        record=row.record,
                        group_name=row.group_name or "",
                        export_key=row.export_key,
                        source_layer_id=str(source_layer.id_string),
                        source_layer_name=str(source_layer.name),
                        detector_mode=str(mode),
                        bbox={},
                        warnings=list(row.warnings),
                        error=str(exc),
                    )
                )
        return reports

    def _create_result_layer(
        self,
        row: RepairGroupRow,
        source_layer: Any,
        source_image_bytes: bytes,
        projection_bounds: Any,
        result: DetectionResult,
        index: int,
        options: DetectionOptions,
        extract_prompts: bool = False,
    ) -> GroupDetectionReport:
        if row.group_layer is None:
            raise RuntimeError("Group is unresolved; refusing to create root-level repair layer.")

        bbox = self._bbox_to_document_dict(
            result.bbox,
            result.coordinate_space,
            projection_bounds,
        )
        crop_bbox = self._crop_bbox_for_options(
            source_image_bytes,
            bbox,
            options,
            projection_bounds,
        )
        crop_bytes = self._crop_png_bytes_from_dict(
            source_image_bytes,
            crop_bbox,
            projection_bounds,
        )
        if not crop_bytes:
            raise RuntimeError(
                "BBox crop failed; refusing to use the full source projection as crop bytes."
            )
        layer_name = self._result_layer_name(str(source_layer.name), result, index)

        created_layer = add_repair_result_layer_to_group(
            document_ref=source_layer.document_ref,
            group_layer=row.group_layer,
            source_layer=source_layer,
            name=layer_name,
            png_bytes=crop_bytes,
            x=int(crop_bbox.get("x", 0) or 0),
            y=int(crop_bbox.get("y", 0) or 0),
        )

        prompt_text = self._extract_prompt_text(
            str(created_layer.id_string),
            crop_bytes,
            extract_prompts,
        )

        result_row = RepairResultRow(
            selected=True,
            active=True,
            record=row.record,
            group_layer=row.group_layer,
            source_layer=source_layer,
            created_layer=created_layer,
            detector_bbox=bbox,
            crop_bbox=crop_bbox,
            crop_png_bytes=crop_bytes,
            detector_mode=str(result.mode),
            detector_label=str(result.label),
            detector_score=float(result.score),
            force_rect_crop=bool(options.force_rect_crop),
            rect_width=int(options.rect_width),
            rect_height=int(options.rect_height),
            prompt_text=prompt_text,
            prompt_success=bool(prompt_text),
            prompt_status="done" if prompt_text else "not_started",
        )

        report = GroupDetectionReport(
            record=row.record,
            group_name=row.group_name or "",
            export_key=row.export_key,
            source_layer_id=str(source_layer.id_string),
            source_layer_name=str(source_layer.name),
            detector_mode=str(result.mode),
            bbox=crop_bbox,
            created_layer_id=str(created_layer.id_string),
            created_layer_name=str(created_layer.name),
            prompt_text=prompt_text,
            result_row=result_row,
            warnings=list(row.warnings),
        )

        row.created_layer_ids.append(str(created_layer.id_string))
        row.detected_count += 1

        metadata = result_row.to_metadata() | self._report_metadata(row, report, result)
        attach = getattr(self.metadata_service, "attach_group_batch_result_metadata", None)
        if not callable(attach):
            attach = getattr(self.metadata_service, "attach_detector_metadata", None)
        if callable(attach):
            attach(created_layer, metadata)

        return report

    def _extract_prompt_text(
        self,
        layer_id: str,
        image_bytes: bytes,
        enabled: bool,
    ) -> str:
        if not enabled or self.prompt_service is None:
            return ""

        extract = getattr(self.prompt_service, "extract_prompt_from_bytes", None)
        if not callable(extract):
            return ""

        result = extract(layer_id, image_bytes)
        if bool(getattr(result, "success", False)):
            return str(getattr(result, "prompt_text", "") or "")
        return ""

    def _result_layer_name(self, source_name: str, result: DetectionResult, index: int) -> str:
        mode = str(result.mode or "repair").lower()
        if "head" in mode:
            prefix = "[Repair head]"
        elif "censor" in mode:
            prefix = "[Repair censor]"
        else:
            prefix = f"[Repair {mode}]"
        return f"{prefix} {source_name} #{index:02d}"

    def _report_metadata(
        self,
        row: RepairGroupRow,
        report: GroupDetectionReport,
        result: DetectionResult,
    ) -> dict[str, Any]:
        return {
            "repair_plugin.schema_version": 1,
            "repair_plugin.source_group_id": row.group_id,
            "repair_plugin.source_group_name": row.group_name,
            "repair_plugin.export_key": row.export_key,
            "repair_plugin.source_layer_id": report.source_layer_id,
            "repair_plugin.source_layer_name": report.source_layer_name,
            "repair_plugin.detector_mode": report.detector_mode,
            "repair_plugin.detector_label": str(result.label),
            "repair_plugin.detector_bbox": report.result_row.detector_bbox if report.result_row else report.bbox,
            "repair_plugin.crop_bbox": report.bbox,
            "repair_plugin.detector_score": float(result.score),
            "repair_plugin.force_rect_crop": bool(report.result_row.force_rect_crop) if report.result_row else False,
            "repair_plugin.rect_width": int(report.result_row.rect_width) if report.result_row else 0,
            "repair_plugin.rect_height": int(report.result_row.rect_height) if report.result_row else 0,
            "repair_plugin.prompt_text": report.prompt_text,
            "repair_plugin.created_layer_id": report.created_layer_id,
            "repair_plugin.created_layer_name": report.created_layer_name,
        }

    def _bbox_to_document_dict(
        self,
        bbox: BoundingBox,
        coordinate_space: str,
        projection_bounds: Any,
    ) -> dict[str, int]:
        x = int(bbox.x)
        y = int(bbox.y)
        if str(coordinate_space or "").lower() == "projection":
            x += int(getattr(projection_bounds, "x", 0))
            y += int(getattr(projection_bounds, "y", 0))
        return {
            "x": x,
            "y": y,
            "width": int(bbox.width),
            "height": int(bbox.height),
        }

    def _crop_png_bytes(self, png_bytes: bytes, bbox: BoundingBox) -> bytes | None:
        image = QtGui.QImage()
        if not image.loadFromData(png_bytes, "PNG"):
            return None

        x = max(0, int(bbox.x))
        y = max(0, int(bbox.y))
        width = max(1, min(int(bbox.width), int(image.width()) - x))
        height = max(1, min(int(bbox.height), int(image.height()) - y))
        if width <= 0 or height <= 0:
            return None

        crop = image.copy(x, y, width, height)
        data = QtCore.QByteArray()
        buffer = QtCore.QBuffer(data)
        buffer.open(self._write_only_mode())
        crop.save(buffer, "PNG")
        buffer.close()
        return bytes(data)


    def _crop_bbox_for_options(
        self,
        png_bytes: bytes,
        detector_bbox: dict[str, int],
        options: DetectionOptions,
        projection_bounds: Any,
    ) -> dict[str, int]:
        """Return document-space crop bbox for row state and layer placement.

        The rendered PNG is projection-local, while detector rows and layer
        placement use document-space coordinates. Keep the returned crop bbox in
        document space, but do all image-size expansion/clamping in local image
        space. This makes Force rect crop on/off use the same coordinate contract.

        When force_rect_crop is off, the crop is still expanded beyond the
        detector bbox by min_margin so that crop > detector. This ensures
        refine_region (with mask + inpaint) is used instead of whole-crop
        refine (no mask), giving consistent feathered output.
        """
        image = QtGui.QImage()
        if not image.loadFromData(png_bytes, "PNG"):
            return dict(detector_bbox)

        local_detector_bbox = self._document_bbox_to_projection_local(
            detector_bbox,
            projection_bounds,
        )

        if not options.force_rect_crop:
            # Auto-expand: use detector size as target, add margin for context
            local_crop_bbox = expand_bbox_to_forced_rect(
                local_detector_bbox,
                int(image.width()),
                int(image.height()),
                int(local_detector_bbox.get("width", 1)),
                int(local_detector_bbox.get("height", 1)),
                True,
                min_margin=100,
            )
            return self._projection_local_bbox_to_document(
                local_crop_bbox,
                projection_bounds,
            )

        # Force rect crop: use exact user-specified size, no extra margin
        local_crop_bbox = expand_bbox_to_forced_rect(
            local_detector_bbox,
            int(image.width()),
            int(image.height()),
            int(options.rect_width),
            int(options.rect_height),
            bool(options.clamp_rect_to_source_bounds),
            min_margin=0,
        )
        return self._projection_local_bbox_to_document(
            local_crop_bbox,
            projection_bounds,
        )

    def _document_bbox_to_projection_local(
        self,
        bbox: dict[str, int],
        projection_bounds: Any,
    ) -> dict[str, int]:
        """Convert a document-space bbox into rendered projection PNG coordinates."""
        offset_x = int(getattr(projection_bounds, "x", 0) or 0)
        offset_y = int(getattr(projection_bounds, "y", 0) or 0)
        return {
            "x": int(bbox.get("x", 0) or 0) - offset_x,
            "y": int(bbox.get("y", 0) or 0) - offset_y,
            "width": int(bbox.get("width", 1) or 1),
            "height": int(bbox.get("height", 1) or 1),
        }

    def _projection_local_bbox_to_document(
        self,
        bbox: dict[str, int],
        projection_bounds: Any,
    ) -> dict[str, int]:
        """Convert a rendered projection PNG bbox back into document coordinates."""
        offset_x = int(getattr(projection_bounds, "x", 0) or 0)
        offset_y = int(getattr(projection_bounds, "y", 0) or 0)
        return {
            "x": int(bbox.get("x", 0) or 0) + offset_x,
            "y": int(bbox.get("y", 0) or 0) + offset_y,
            "width": int(bbox.get("width", 1) or 1),
            "height": int(bbox.get("height", 1) or 1),
        }

    def _crop_png_bytes_from_dict(
        self,
        png_bytes: bytes,
        bbox: dict[str, int],
        projection_bounds: Any,
    ) -> bytes | None:
        """Crop rendered projection PNG using bbox converted from document space."""
        image = QtGui.QImage()
        if not image.loadFromData(png_bytes, "PNG"):
            return None

        local_bbox = self._document_bbox_to_projection_local(bbox, projection_bounds)
        x = int(local_bbox.get("x", 0) or 0)
        y = int(local_bbox.get("y", 0) or 0)
        width = int(local_bbox.get("width", 1) or 1)
        height = int(local_bbox.get("height", 1) or 1)

        # Clamp in projection-local coordinates. If the detector bbox starts
        # outside the rendered image, shift the crop origin back into the image
        # and reduce the size instead of asking Qt to copy out-of-bounds pixels,
        # which can produce an all-black crop.
        if x < 0:
            width += x
            x = 0
        if y < 0:
            height += y
            y = 0

        width = min(width, int(image.width()) - x)
        height = min(height, int(image.height()) - y)
        if width <= 0 or height <= 0:
            return None

        crop = image.copy(x, y, width, height)
        data = QtCore.QByteArray()
        buffer = QtCore.QBuffer(data)
        buffer.open(self._write_only_mode())
        crop.save(buffer, "PNG")
        buffer.close()
        return bytes(data)

    def _write_only_mode(self) -> Any:
        open_mode_flag = getattr(QtCore.QIODevice, "OpenModeFlag", None)
        if open_mode_flag is not None and hasattr(open_mode_flag, "WriteOnly"):
            return open_mode_flag.WriteOnly
        return QtCore.QIODevice.WriteOnly