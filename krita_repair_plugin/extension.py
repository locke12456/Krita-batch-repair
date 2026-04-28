"""Krita extension entry point for the repair plugin."""

from __future__ import annotations

from typing import Any

from . import PLUGIN_NAME
from .repair_compat import QMessageBox

try:
    from krita import Extension, Krita
except Exception:
    Extension = object
    Krita = None


class RepairPluginExtension(Extension):
    """Thin Krita action registration layer.

    Detector loading, prompt extraction, generation handoff, and metadata work
    live in service modules. This class only exposes entry actions.
    """

    def __init__(self, parent: Any = None) -> None:
        try:
            super().__init__(parent)
        except TypeError:
            super().__init__()
        self._docker = None

    def setup(self) -> None:
        """Krita calls setup during plugin initialization."""
        return None

    def createActions(self, window: Any) -> None:
        """Register visible repair plugin actions for a Krita window."""
        if window is None or not hasattr(window, "createAction"):
            return

        action = window.createAction(
            "krita_repair_plugin_open_docker",
            "Open Auto Detect Repair Plugin",
            "tools/scripts",
        )
        if hasattr(action, "triggered"):
            action.triggered.connect(self._show_docker)

    def _show_docker(self) -> None:
        """Create or focus the repair docker."""
        try:
            from .docker import RepairDocker

            if self._docker is None:
                self._docker = RepairDocker()
            show = getattr(self._docker, "show", None)
            raise_ = getattr(self._docker, "raise_", None)
            if callable(show):
                show()
            if callable(raise_):
                raise_()
        except Exception as exc:
            self._show_error(str(exc))

    def _show_error(self, message: str) -> None:
        """Display a compatibility-safe error message."""
        try:
            QMessageBox.critical(None, PLUGIN_NAME, message)
        except Exception:
            print(f"{PLUGIN_NAME}: {message}")


def register_extension() -> RepairPluginExtension | None:
    """Register the extension with Krita when the Krita API is available."""
    if Krita is None:
        return None
    extension = RepairPluginExtension(Krita.instance())
    Krita.instance().addExtension(extension)
    return extension


_EXTENSION = register_extension()