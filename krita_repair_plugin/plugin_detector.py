"""Plugin-owned layer projection detector contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized detector bounding box."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class LayerProjectionInput:
    """Layer projection input accepted by the repair plugin detector."""

    layer_id: str
    layer_name: str
    bounds: Any
    image_bytes: bytes
    coordinate_space: str = "projection"


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Normalized detector result returned to detection service."""

    mode: str
    label: str
    score: float
    bbox: BoundingBox
    coordinate_space: str


class _ImgutilsDetectorBackend:
    """Small backend adapter around imgutils detector functions."""

    HEAD_MODEL = "head_detect_v2.0_x_yv11"
    CENSOR_MODEL = "censor_detect_v1.0_s"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def detect(self, image_path: str, options: dict[str, Any]) -> list[Any]:
        """Run imgutils detection for a temporary projection image path."""
        if self.mode == "head":
            from imgutils.detect import detect_heads

            model_name = options.get("model_name", self.HEAD_MODEL)
            return list(detect_heads(image_path, model_name))
        if self.mode == "censor":
            from imgutils.detect import detect_censors

            model_name = options.get("model_name", self.CENSOR_MODEL)
            return list(detect_censors(image_path, model_name=model_name))
        raise ValueError(f"Unsupported backend mode: {self.mode}")


class PluginDetector:
    """Runtime detector boundary for layer-projection detection.

    Existing detector source files are algorithm references only. This class is
    the repair plugin contract that later backend code must implement.
    """

    SUPPORTED_MODES = {"all", "head", "censor"}

    def __init__(self) -> None:
        self.loaded_modes: set[str] = set()
        self.backend_handles: dict[str, Any] = {}

    def load(self, mode: str | None = None) -> dict[str, Any]:
        """Load or warm backend handles for a mode."""
        target_mode = self._normalize_mode(mode)
        modes = ("head", "censor") if target_mode == "all" else (target_mode,)
        for backend_mode in modes:
            self.backend_handles[backend_mode] = _ImgutilsDetectorBackend(backend_mode)
            self.loaded_modes.add(backend_mode)
        return {
            "mode": target_mode,
            "message": f"Detector mode loaded: {target_mode}",
            "unload_note": "Best-effort unload clears plugin references and requests cache cleanup.",
        }

    def unload(self, mode: str | None = None) -> dict[str, Any]:
        """Unload backend handles for a mode."""
        target_mode = self._normalize_mode(mode)
        if target_mode == "all":
            self.loaded_modes.clear()
            self.backend_handles.clear()
        else:
            self.loaded_modes.discard(target_mode)
            self.backend_handles.pop(target_mode, None)
        return {
            "mode": target_mode,
            "message": f"Detector mode unloaded: {target_mode}",
            "unload_note": "Backend libraries may keep global caches after plugin references are cleared.",
        }

    def detect_layer(
        self,
        mode: str,
        projection: LayerProjectionInput | Any,
        options: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        """Detect candidate regions in a layer projection."""
        target_mode = self._normalize_mode(mode)
        run_modes = ("head", "censor") if target_mode == "all" else (target_mode,)
        options = options or {}

        results: list[DetectionResult] = []
        image_bytes = self._projection_image_bytes(projection)
        coordinate_space = str(getattr(projection, "coordinate_space", "projection"))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            temp.write(image_bytes)
            temp_path = temp.name

        try:
            for backend_mode in run_modes:
                backend = self.backend_handles.get(backend_mode)
                if backend is None:
                    raise RuntimeError(f"Detector mode is not loaded: {backend_mode}")
                raw_results = backend.detect(temp_path, options)
                results.extend(
                    self._normalize_results(
                        backend_mode,
                        raw_results,
                        coordinate_space,
                        options,
                    )
                )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        score_threshold = float(options.get("score_threshold", 0.0) or 0.0)
        if score_threshold > 0:
            results = [result for result in results if result.score >= score_threshold]
        top_n = options.get("top_n")
        if top_n is not None:
            results = sorted(results, key=lambda item: item.score, reverse=True)[: int(top_n)]
        return results

    def _normalize_mode(self, mode: str | None) -> str:
        """Normalize detector mode and reject unsupported values."""
        if mode is None:
            return "all"
        text = str(mode).strip().lower()
        if not text:
            return "all"
        if text not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported detector mode: {text}")
        return text

    def _projection_image_bytes(self, projection: LayerProjectionInput | Any) -> bytes:
        """Extract PNG bytes from a LayerProjectionInput-like object."""
        image_bytes = getattr(projection, "image_bytes", None)
        if image_bytes is None and isinstance(projection, dict):
            image_bytes = projection.get("image_bytes")
        if image_bytes is None:
            raise ValueError("projection.image_bytes is required")
        return bytes(image_bytes)

    def _normalize_results(
        self,
        mode: str,
        raw_results: list[Any],
        coordinate_space: str,
        options: dict[str, Any],
    ) -> list[DetectionResult]:
        """Normalize imgutils result tuples or dicts into DetectionResult records."""
        filter_label = options.get("filter_label")
        normalized: list[DetectionResult] = []
        for item in raw_results:
            bbox, label, score = self._split_raw_result(mode, item)
            if filter_label and label != filter_label:
                continue
            normalized.append(
                DetectionResult(
                    mode=mode,
                    label=label,
                    score=float(score),
                    bbox=self._normalize_bbox(bbox),
                    coordinate_space=coordinate_space,
                )
            )
        return sorted(normalized, key=lambda result: result.score, reverse=True)

    def _split_raw_result(self, mode: str, item: Any) -> tuple[Any, str, float]:
        """Split common imgutils tuple and dict result shapes."""
        if isinstance(item, dict):
            bbox = item.get("bbox") or item.get("box")
            label = str(item.get("label") or mode)
            score = float(item.get("score", item.get("confidence", 1.0)))
            return bbox, label, score
        if isinstance(item, (tuple, list)):
            if len(item) >= 3:
                return item[0], str(item[1]), float(item[2])
            if len(item) == 2:
                return item[0], mode, float(item[1])
            if len(item) == 1:
                return item[0], mode, 1.0
        return item, mode, 1.0

    def _normalize_bbox(self, bbox: Any) -> BoundingBox:
        """Normalize tuple, dict, or object bbox values into BoundingBox."""
        if isinstance(bbox, BoundingBox):
            return bbox
        if isinstance(bbox, dict):
            return BoundingBox(
                x=int(bbox.get("x", 0)),
                y=int(bbox.get("y", 0)),
                width=int(bbox.get("width", bbox.get("w", 0))),
                height=int(bbox.get("height", bbox.get("h", 0))),
            )
        if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            return BoundingBox(
                x=x1,
                y=y1,
                width=max(0, x2 - x1),
                height=max(0, y2 - y1),
            )
        return BoundingBox(
            x=int(getattr(bbox, "x", 0)),
            y=int(getattr(bbox, "y", 0)),
            width=int(getattr(bbox, "width", 0)),
            height=int(getattr(bbox, "height", 0)),
        )