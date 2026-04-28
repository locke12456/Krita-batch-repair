"""Repair plugin compatibility facade.

This module re-exports the shared KritaAI Export Plugin wrapper/core surface.
It must not contain copied wrapper implementations. Repair plugin services may
also import the shared modules directly when that is clearer.
"""

from __future__ import annotations

from typing import Any

from krita_ai_metadata import ai_diffusion_compat
from krita_ai_metadata import krita_core_adapter
from krita_ai_metadata import qt_compat

active_ai_document = ai_diffusion_compat.active_document
active_ai_document_instance = ai_diffusion_compat.active_document_instance
active_ai_model = ai_diffusion_compat.active_model
deserialize_job_params = ai_diffusion_compat.deserialize_job_params
format_img_metadata = ai_diffusion_compat.format_img_metadata
is_finished_job = ai_diffusion_compat.is_finished_job
is_group_layer = ai_diffusion_compat.is_group_layer
is_image_layer = ai_diffusion_compat.is_image_layer
make_bounds = ai_diffusion_compat.make_bounds
refresh_ai_projection = ai_diffusion_compat.refresh_projection
require_ai_diffusion_api = ai_diffusion_compat.require_api
trim_prompt = ai_diffusion_compat.trim_prompt

KritaBounds = krita_core_adapter.KritaBounds
KritaDocumentRef = krita_core_adapter.KritaDocumentRef
KritaNodeRef = krita_core_adapter.KritaNodeRef
KritaRenderedImage = krita_core_adapter.KritaRenderedImage
active_krita_document = krita_core_adapter.active_krita_document
add_layer_only_paint_layer = krita_core_adapter.add_layer_only_paint_layer
all_krita_nodes = krita_core_adapter.all_krita_nodes
render_node_projection = krita_core_adapter.render_node_projection
selected_krita_nodes = krita_core_adapter.selected_krita_nodes
wrap_node = krita_core_adapter.wrap_node

QtCore = qt_compat.QtCore
QtGui = qt_compat.QtGui
QtWidgets = qt_compat.QtWidgets
Qt = qt_compat.Qt
QAction = qt_compat.QAction
QCheckBox = qt_compat.QCheckBox
QComboBox = qt_compat.QComboBox
QHBoxLayout = qt_compat.QHBoxLayout
QLabel = qt_compat.QLabel
QLineEdit = qt_compat.QLineEdit
QMessageBox = qt_compat.QMessageBox
QPushButton = qt_compat.QPushButton
QScrollArea = qt_compat.QScrollArea
QVBoxLayout = qt_compat.QVBoxLayout
QWidget = qt_compat.QWidget
checked_state = qt_compat.checked_state
unchecked_state = qt_compat.unchecked_state


def ai_diffusion_available() -> bool:
    """Return whether the shared ai-diffusion compatibility wrapper is usable."""
    return ai_diffusion_compat.IMPORT_ERROR is None


def ai_diffusion_error() -> Exception | None:
    """Return the import error captured by the shared wrapper, if any."""
    return ai_diffusion_compat.IMPORT_ERROR


def active_document_any() -> Any:
    """Prefer ai-diffusion document access, then fall back to manual Krita access."""
    if ai_diffusion_available():
        try:
            return active_ai_document()
        except Exception:
            pass
    return active_krita_document()



__all__ = [
    "QAction",
    "QCheckBox",
    "QComboBox",
    "QHBoxLayout",
    "QLabel",
    "QLineEdit",
    "QMessageBox",
    "QPushButton",
    "QScrollArea",
    "QVBoxLayout",
    "QWidget",
    "Qt",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "KritaBounds",
    "KritaDocumentRef",
    "KritaNodeRef",
    "KritaRenderedImage",
    "active_ai_document",
    "active_ai_document_instance",
    "active_ai_model",
    "active_document_any",
    "active_krita_document",
    "ai_diffusion_available",
    "ai_diffusion_error",
    "add_layer_only_paint_layer",
    "all_krita_nodes",
    "checked_state",
    "deserialize_job_params",
    "format_img_metadata",
    "is_finished_job",
    "is_group_layer",
    "is_image_layer",
    "make_bounds",
    "refresh_ai_projection",
    "render_node_projection",
    "require_ai_diffusion_api",
    "selected_krita_nodes",
    "trim_prompt",
    "unchecked_state",
    "wrap_node",
]