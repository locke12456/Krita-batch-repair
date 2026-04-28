"""Layer-only detection orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .detection_layer_selection_model import DetectionLayerRow, DetectionLayerSelectionModel
from .detector_model_manager import DetectorModelManager
from .plugin_detector import BoundingBox, DetectionResult, LayerProjectionInput
from .repair_compat import (
    QtCore,
    QtGui,
    active_krita_document,
    add_layer_only_paint_layer,
    render_node_projection,
    selected_krita_nodes,
)


@dataclass(frozen=True, slots=True)
class DetectionOptions:
    """Options for one detection run."""

    top_n: int | None = None
    filter_label: str | None = None
    score_threshold: float = 0.0
    crop_strategy: str = "bbox"
    transparent_pixel_handling: str = "preserve"
    force_rect_crop: bool = False
    rect_width: int = 260
    rect_height: int = 340
    clamp_rect_to_source_bounds: bool = True

    def to_detector_options(self) -> dict[str, Any]:
        """Return options accepted by PluginDetector."""
        result: dict[str, Any] = {
            "score_threshold": self.score_threshold,
            "crop_strategy": self.crop_strategy,
            "transparent_pixel_handling": self.transparent_pixel_handling,
        }
        if self.top_n is not None:
            result["top_n"] = self.top_n
        if self.filter_label:
            result["filter_label"] = self.filter_label
        return result


def expand_bbox_to_forced_rect(
    bbox: dict[str, int],
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
    clamp_to_bounds: bool = True,
) -> dict[str, int]:
    """Expand a detector bbox around its center to a fixed crop rectangle."""
    x = int(bbox.get("x", bbox.get("x1", 0)) or 0)
    y = int(bbox.get("y", bbox.get("y1", 0)) or 0)
    width = int(bbox.get("width", 0) or 0)
    height = int(bbox.get("height", 0) or 0)
    if width <= 0 and "x2" in bbox:
        width = int(bbox["x2"]) - x
    if height <= 0 and "y2" in bbox:
        height = int(bbox["y2"]) - y

    bbox_width = max(1, width)
    bbox_height = max(1, height)
    actual_width = max(int(target_width), bbox_width)
    actual_height = max(int(target_height), bbox_height)
    center_x = x + bbox_width // 2
    center_y = y + bbox_height // 2

    x1 = center_x - actual_width // 2
    y1 = center_y - actual_height // 2
    x2 = x1 + actual_width
    y2 = y1 + actual_height

    if clamp_to_bounds:
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > image_width:
            shift = x2 - int(image_width)
            x1 -= shift
            x2 = int(image_width)
        if y2 > image_height:
            shift = y2 - int(image_height)
            y1 -= shift
            y2 = int(image_height)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(int(image_width), x2)
        y2 = min(int(image_height), y2)

    return {
        "x": int(x1),
        "y": int(y1),
        "width": max(1, int(x2 - x1)),
        "height": max(1, int(y2 - y1)),
    }


class DetectionService:
    """Detection utility service.

    The SyncRecord group batch workflow owns source selection and result
    placement. Candidate row creation is retained only for legacy/debug use.
    """

    def __init__(
        self,
        detector_manager: DetectorModelManager,
        selection_model: DetectionLayerSelectionModel | None = None,
        metadata_service: Any | None = None,
    ) -> None:
        self.detector_manager = detector_manager
        self.selection_model = selection_model or DetectionLayerSelectionModel()
        self.metadata_service = metadata_service

    def detect_projection(
        self,
        mode: str,
        source_layer: Any,
        options: DetectionOptions | dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        """Detect objects on an explicit source layer without creating UI rows."""
        projection, _image_bytes, _projection_bounds = self.render_projection_input(source_layer)
        detection_options = self._normalize_options(options)
        return self.detector_manager.detect_layer(
            mode,
            projection,
            detection_options.to_detector_options(),
        )

    def crop_bbox(
        self,
        png_bytes: bytes,
        bbox: BoundingBox,
    ) -> bytes | None:
        """Crop PNG bytes to the detector bbox."""
        return self._crop_png_bytes(png_bytes, bbox)

    def render_projection_input(
        self,
        source_layer: Any,
    ) -> tuple[LayerProjectionInput, bytes, Any]:
        """Render one explicit layer and return detector projection input."""
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
        return projection, image_bytes, projection_bounds

    def detect_layer(
        self,
        mode: str,
        source_layer: Any | None = None,
        options: DetectionOptions | dict[str, Any] | None = None,
    ) -> list[DetectionLayerRow]:
        """Legacy/debug-only: detect one layer and create candidate rows."""
        document_ref = active_krita_document()
        if document_ref is None:
            raise RuntimeError("No active Krita document is available")

        node_ref = source_layer or self._selected_source_layer()
        projection, image_bytes, projection_bounds = self.render_projection_input(node_ref)

        detection_options = self._normalize_options(options)
        raw_results = self.detector_manager.detect_layer(
            mode,
            projection,
            detection_options.to_detector_options(),
        )

        rows: list[DetectionLayerRow] = []
        for index, result in enumerate(raw_results, start=1):
            row = self._create_candidate_row(
                document_ref=document_ref,
                source_layer=node_ref,
                source_image_bytes=image_bytes,
                projection_bounds=projection_bounds,
                result=result,
                index=index,
            )
            rows.append(row)

        self.selection_model.add_rows(rows)
        return rows

    def detect_selected_layers(
        self,
        mode: str,
        options: DetectionOptions | dict[str, Any] | None = None,
    ) -> list[DetectionLayerRow]:
        """Legacy/debug-only: run detection over currently selected Krita layers."""
        selected = selected_krita_nodes()
        if not selected:
            return self.detect_layer(mode, None, options)
        rows: list[DetectionLayerRow] = []
        for node_ref in selected:
            rows.extend(self.detect_layer(mode, node_ref, options))
        return rows

    def _selected_source_layer(self) -> Any:
        """Resolve the selected source layer."""
        selected = selected_krita_nodes()
        if selected:
            return selected[0]

        document_ref = active_krita_document()
        if document_ref is None:
            raise RuntimeError("No active Krita document is available")

        active_node = document_ref.active_node()
        if active_node is None:
            raise RuntimeError("No active Krita layer is available")

        from .repair_compat import wrap_node

        return wrap_node(active_node)

    def _create_candidate_row(
        self,
        document_ref: Any,
        source_layer: Any,
        source_image_bytes: bytes,
        projection_bounds: Any,
        result: DetectionResult,
        index: int,
    ) -> DetectionLayerRow:
        """Create one candidate layer and matching selection row."""
        bbox = self._bbox_to_document_dict(
            result.bbox,
            result.coordinate_space,
            projection_bounds,
        )
        candidate_name = self._candidate_layer_name(source_layer.name, result, index)
        crop_bytes = self._crop_png_bytes(source_image_bytes, result.bbox) or source_image_bytes

        candidate_layer = add_layer_only_paint_layer(
            document_ref=document_ref,
            name=candidate_name,
            png_bytes=crop_bytes,
            parent_node=None,
            above_node=getattr(source_layer, "node", None),
        )

        row = DetectionLayerRow(
            layer_id=str(candidate_layer.id_string),
            layer_name=str(candidate_layer.name),
            mode=str(result.mode),
            label=str(result.label),
            bbox=bbox,
            coordinate_space="document",
            score=float(result.score),
            selected=True,
            active=True,
            visible=True,
            image_bytes=crop_bytes,
        )

        self._attach_detector_metadata(candidate_layer, row)
        return row

    def _attach_detector_metadata(self, candidate_layer: Any, row: DetectionLayerRow) -> None:
        """Attach detector metadata through an optional metadata service."""
        service = self.metadata_service
        if service is None:
            return
        attach = getattr(service, "attach_detector_metadata", None)
        if callable(attach):
            attach(candidate_layer, row.to_metadata())

    def _normalize_options(
        self,
        options: DetectionOptions | dict[str, Any] | None,
    ) -> DetectionOptions:
        """Normalize dict options into DetectionOptions."""
        if isinstance(options, DetectionOptions):
            return options
        data = options or {}
        return DetectionOptions(
            top_n=data.get("top_n"),
            filter_label=data.get("filter_label"),
            score_threshold=float(data.get("score_threshold", 0.0) or 0.0),
            crop_strategy=str(data.get("crop_strategy", "bbox") or "bbox"),
            transparent_pixel_handling=str(
                data.get("transparent_pixel_handling", "preserve") or "preserve"
            ),
            force_rect_crop=bool(data.get("force_rect_crop", False)),
            rect_width=int(data.get("rect_width", 260) or 260),
            rect_height=int(data.get("rect_height", 340) or 340),
            clamp_rect_to_source_bounds=bool(data.get("clamp_rect_to_source_bounds", True)),
        )

    def _bbox_to_document_dict(
        self,
        bbox: BoundingBox,
        coordinate_space: str,
        projection_bounds: Any,
    ) -> dict[str, int]:
        """Convert detector bbox values into the row document-coordinate contract."""
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

    def _candidate_layer_name(self, source_name: str, result: DetectionResult, index: int) -> str:
        """Build a deterministic candidate layer name."""
        label = result.label or result.mode
        score = int(max(0.0, min(1.0, float(result.score))) * 100)
        return f"[Repair Candidate] {source_name} - {label} {index:02d} ({score}%)"

    def _crop_png_bytes(self, png_bytes: bytes, bbox: BoundingBox) -> bytes | None:
        """Crop PNG bytes to a detector bbox using Qt image APIs."""
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
        """Return a Qt5 and Qt6 compatible write-only open mode."""
        open_mode_flag = getattr(QtCore.QIODevice, "OpenModeFlag", None)
        if open_mode_flag is not None and hasattr(open_mode_flag, "WriteOnly"):
            return open_mode_flag.WriteOnly
        return QtCore.QIODevice.WriteOnly