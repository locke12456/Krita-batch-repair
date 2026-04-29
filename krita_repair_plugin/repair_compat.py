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
find_krita_node_by_id = krita_core_adapter.find_krita_node_by_id
merge_layer_down = krita_core_adapter.merge_layer_down
merge_layer_into_target = krita_core_adapter.merge_layer_into_target
move_layer_above = krita_core_adapter.move_layer_above
move_layer_immediately_above = krita_core_adapter.move_layer_immediately_above
render_node_projection = krita_core_adapter.render_node_projection
set_layer_visible = krita_core_adapter.set_layer_visible
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


def add_repair_result_layer_to_group(
    document_ref: Any,
    group_layer: Any,
    source_layer: Any,
    name: str,
    png_bytes: bytes,
    x: int = 0,
    y: int = 0,
) -> Any:
    """Add a repair result layer under the original group at document coordinates."""
    if document_ref is None:
        raise RuntimeError("Document reference is required.")
    if group_layer is None:
        raise RuntimeError("Group layer is required; refusing root-level repair layer.")

    parent_node = getattr(group_layer, "node", group_layer)
    if parent_node is None:
        raise RuntimeError("Group node is required; refusing root-level repair layer.")

    above_node = getattr(source_layer, "node", None)
    created_layer = add_layer_only_paint_layer(
        document_ref=document_ref,
        name=name,
        png_bytes=None,
        parent_node=parent_node,
        above_node=above_node,
    )

    if not png_bytes:
        return created_layer

    node = getattr(created_layer, "node", created_layer)
    set_pixel_data = getattr(node, "setPixelData", None)
    if not callable(set_pixel_data):
        raise RuntimeError("Created layer does not expose setPixelData.")

    image = QtGui.QImage()
    if not image.loadFromData(png_bytes, "PNG"):
        raise ValueError("Repair result PNG bytes could not be decoded.")

    image = image.convertToFormat(qt_compat.image_format_argb32())
    width = int(image.width())
    height = int(image.height())
    ptr = image.bits()
    try:
        ptr.setsize(image.sizeInBytes())
    except Exception:
        ptr.setsize(image.byteCount())

    set_pixel_data(qt_compat.QByteArray(bytes(ptr)), int(x), int(y), width, height)
    document_ref.refresh_projection()
    return created_layer


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
    "add_repair_result_layer_to_group",
    "all_krita_nodes",
    "checked_state",
    "deserialize_job_params",
    "find_krita_node_by_id",
    "format_img_metadata",
    "is_finished_job",
    "is_group_layer",
    "is_image_layer",
    "make_bounds",
    "merge_layer_down",
    "merge_layer_into_target",
    "move_layer_above",
    "move_layer_immediately_above",
    "refresh_ai_projection",
    "render_node_projection",
    "set_layer_visible",
    "require_ai_diffusion_api",
    "selected_krita_nodes",
    "trim_prompt",
    "unchecked_state",
    "wrap_node",
]