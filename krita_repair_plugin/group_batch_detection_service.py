from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from krita_ai_metadata.sync_map_store import SyncRecord

from .detection_service import DetectionOptions
from .group_selection_model import RepairGroupRow
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
    warnings: list[str] = field(default_factory=list)
    error: str = ""


class GroupBatchDetectionService:
    def __init__(
        self,
        detector_manager: Any,
        metadata_service: Any,
        prompt_service: Any | None = None,
    ) -> None:
        self.detector_manager = detector_manager
        self.metadata_service = metadata_service
        self.prompt_service = prompt_service

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
        extract_prompts: bool = False,
    ) -> GroupDetectionReport:
        if row.group_layer is None:
            raise RuntimeError("Group is unresolved; refusing to create root-level repair layer.")

        bbox = self._bbox_to_document_dict(
            result.bbox,
            result.coordinate_space,
            projection_bounds,
        )
        crop_bytes = self._crop_png_bytes(source_image_bytes, result.bbox) or source_image_bytes
        layer_name = self._result_layer_name(str(source_layer.name), result, index)

        created_layer = add_repair_result_layer_to_group(
            document_ref=source_layer.document_ref,
            group_layer=row.group_layer,
            source_layer=source_layer,
            name=layer_name,
            png_bytes=crop_bytes,
            x=int(bbox.get("x", 0) or 0),
            y=int(bbox.get("y", 0) or 0),
        )

        prompt_text = self._extract_prompt_text(
            str(created_layer.id_string),
            crop_bytes,
            extract_prompts,
        )

        report = GroupDetectionReport(
            record=row.record,
            group_name=row.group_name or "",
            export_key=row.export_key,
            source_layer_id=str(source_layer.id_string),
            source_layer_name=str(source_layer.name),
            detector_mode=str(result.mode),
            bbox=bbox,
            created_layer_id=str(created_layer.id_string),
            created_layer_name=str(created_layer.name),
            prompt_text=prompt_text,
            warnings=list(row.warnings),
        )

        row.created_layer_ids.append(str(created_layer.id_string))
        row.detected_count += 1

        metadata = self._report_metadata(row, report, result)
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
            "repair_plugin.detector_bbox": report.bbox,
            "repair_plugin.detector_score": float(result.score),
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

    def _write_only_mode(self) -> Any:
        open_mode_flag = getattr(QtCore.QIODevice, "OpenModeFlag", None)
        if open_mode_flag is not None and hasattr(open_mode_flag, "WriteOnly"):
            return open_mode_flag.WriteOnly
        return QtCore.QIODevice.WriteOnly