"""Explicit detector model residency manager."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

from .plugin_detector import PluginDetector


@dataclass(frozen=True, slots=True)
class DetectorStatus:
    """User-visible detector residency status."""

    state: str
    loaded_modes: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""
    unload_note: str = ""


class DetectorModelManager:
    """Own explicit detector Load/Unload lifecycle for the repair plugin."""

    STATE_UNLOADED = "unloaded"
    STATE_LOADING = "loading"
    STATE_LOADED = "loaded"
    STATE_UNLOADING = "unloading"
    STATE_ERROR = "error"

    def __init__(self, detector: PluginDetector | None = None) -> None:
        self.detector = detector or PluginDetector()
        self.state = self.STATE_UNLOADED
        self.loaded_modes: set[str] = set()
        self.message = ""
        self.unload_note = ""

    def load(self, mode: str | None = None) -> DetectorStatus:
        """Load or warm detector backend handles for a mode."""
        self.state = self.STATE_LOADING
        target_mode = self._normalize_mode(mode)
        try:
            result = self.detector.load(target_mode)
            loaded_mode = str(result.get("mode") or target_mode)
            if loaded_mode == "all":
                self.loaded_modes.update({"head", "censor"})
            else:
                self.loaded_modes.add(loaded_mode)
            self.state = self.STATE_LOADED
            self.message = str(result.get("message") or "Detector loaded")
            self.unload_note = str(result.get("unload_note") or "")
            return self.status()
        except Exception as exc:
            self.state = self.STATE_ERROR
            self.message = str(exc)
            raise

    def unload(self, mode: str | None = None) -> DetectorStatus:
        """Unload detector references and attempt best-effort memory cleanup."""
        self.state = self.STATE_UNLOADING
        target_mode = self._normalize_mode(mode)
        try:
            result = self.detector.unload(target_mode)
            if mode is None or target_mode == "all":
                self.loaded_modes.clear()
            else:
                self.loaded_modes.discard(target_mode)
            self._collect_memory()
            self.state = self.STATE_LOADED if self.loaded_modes else self.STATE_UNLOADED
            self.message = str(result.get("message") or "Detector unloaded")
            self.unload_note = str(
                result.get("unload_note")
                or "Best-effort unload completed; backend caches may retain memory."
            )
            return self.status()
        except Exception as exc:
            self.state = self.STATE_ERROR
            self.message = str(exc)
            raise

    def detect_layer(
        self,
        mode: str,
        projection: Any,
        options: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Run detection through the resident plugin detector."""
        target_mode = self._normalize_mode(mode)
        if target_mode == "all":
            missing = {"head", "censor"} - self.loaded_modes
            if missing:
                raise RuntimeError(f"Detector modes are not loaded: {', '.join(sorted(missing))}")
        elif target_mode not in self.loaded_modes:
            raise RuntimeError(f"Detector mode is not loaded: {target_mode}")
        return self.detector.detect_layer(target_mode, projection, options or {})

    def status(self) -> DetectorStatus:
        """Return current detector residency status."""
        return DetectorStatus(
            state=self.state,
            loaded_modes=tuple(sorted(self.loaded_modes)),
            message=self.message,
            unload_note=self.unload_note,
        )

    def _normalize_mode(self, mode: str | None) -> str:
        """Normalize empty mode values to the architecture's default mode."""
        if mode is None:
            return "all"
        text = str(mode).strip().lower()
        return text or "all"

    def _collect_memory(self) -> None:
        """Run best-effort Python and CUDA cache cleanup."""
        gc.collect()
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            return