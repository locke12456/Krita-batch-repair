"""Plugin-owned candidate layer selection and row state model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class DetectionLayerRow:
    """One detector candidate row owned by the repair plugin UI."""

    layer_id: str
    layer_name: str
    mode: str
    label: str = ""
    bbox: dict[str, int] = field(default_factory=dict)
    coordinate_space: str = "document"
    score: float = 0.0
    selected: bool = True
    active: bool = True
    visible: bool = True
    prompt_text: str = ""
    prompt_extracted: bool = False
    generation_status: str = "not_started"
    metadata_version: int = 1
    image_bytes: bytes | None = None
    raw_prompt_output: Any | None = None
    error_message: str = ""

    def matches_filter(self, filter_mode: str | None) -> bool:
        """Return whether this row is visible for a mode filter."""
        mode = _normalize_filter(filter_mode)
        return mode == "all" or self.mode == mode

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable row metadata payload."""
        return {
            "schema_version": self.metadata_version,
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "detector_mode": self.mode,
            "detector_label": self.label,
            "detector_bbox": dict(self.bbox),
            "detector_bbox_coordinate_space": self.coordinate_space,
            "detector_score": self.score,
            "detector_selected": self.selected,
            "detector_active": self.active,
            "prompt_text": self.prompt_text,
            "prompt_extracted": self.prompt_extracted,
            "generation_status": self.generation_status,
            "visible": self.visible,
            "error_message": self.error_message,
        }


class DetectionLayerSelectionModel:
    """Own repair-plugin row selection, activation, filtering, and query state."""

    def __init__(self) -> None:
        self.rows: list[DetectionLayerRow] = []
        self.filter_mode = "all"

    def add_row(self, row: DetectionLayerRow) -> DetectionLayerRow:
        """Add one candidate row and return it."""
        self.rows.append(row)
        return row

    def add_rows(self, rows: Iterable[DetectionLayerRow]) -> list[DetectionLayerRow]:
        """Add multiple candidate rows and return the added rows."""
        added = list(rows)
        self.rows.extend(added)
        return added

    def clear(self) -> None:
        """Clear all plugin-owned candidate rows."""
        self.rows.clear()

    def set_filter_mode(self, filter_mode: str | None) -> None:
        """Set the current UI filter mode."""
        self.filter_mode = _normalize_filter(filter_mode)

    def filtered_rows(self, filter_mode: str | None = None) -> list[DetectionLayerRow]:
        """Return rows visible under a filter."""
        mode = _normalize_filter(filter_mode or self.filter_mode)
        return [row for row in self.rows if row.matches_filter(mode)]

    def select_all(self, filter_mode: str | None = None) -> None:
        """Select all rows matching the current or supplied filter."""
        for row in self.filtered_rows(filter_mode):
            row.selected = True

    def clear_selected(self, filter_mode: str | None = None) -> None:
        """Clear selection for rows matching the current or supplied filter."""
        for row in self.filtered_rows(filter_mode):
            row.selected = False

    def set_selected(self, layer_ids: Iterable[str], selected: bool) -> None:
        """Set selected state for rows by layer id."""
        ids = set(layer_ids)
        for row in self.rows:
            if row.layer_id in ids:
                row.selected = bool(selected)

    def set_active(self, layer_ids: Iterable[str], active: bool) -> None:
        """Set active state for rows by layer id."""
        ids = set(layer_ids)
        for row in self.rows:
            if row.layer_id in ids:
                row.active = bool(active)

    def selected_layers(self, filter_mode: str | None = None) -> list[DetectionLayerRow]:
        """Return selected rows under a filter."""
        return [row for row in self.filtered_rows(filter_mode) if row.selected]

    def active_layers(self, filter_mode: str | None = None) -> list[DetectionLayerRow]:
        """Return active rows under a filter."""
        return [row for row in self.filtered_rows(filter_mode) if row.active]

    def selected_active_rows(self, filter_mode: str | None = None) -> list[DetectionLayerRow]:
        """Return selected and active rows under a filter."""
        return [
            row
            for row in self.filtered_rows(filter_mode)
            if row.selected and row.active
        ]

    def find_by_layer_id(self, layer_id: str) -> DetectionLayerRow | None:
        """Find a row by layer id."""
        return next((row for row in self.rows if row.layer_id == layer_id), None)

    def update_prompt(
        self,
        layer_id: str,
        prompt_text: str,
        raw_output: Any | None = None,
        extracted: bool = True,
        error_message: str = "",
    ) -> DetectionLayerRow | None:
        """Update prompt extraction state for one row."""
        row = self.find_by_layer_id(layer_id)
        if row is None:
            return None
        row.prompt_text = prompt_text
        row.raw_prompt_output = raw_output
        row.prompt_extracted = bool(extracted)
        row.error_message = error_message
        return row

    def update_generation_status(
        self,
        layer_id: str,
        generation_status: str,
        error_message: str = "",
    ) -> DetectionLayerRow | None:
        """Update generation status for one row."""
        row = self.find_by_layer_id(layer_id)
        if row is None:
            return None
        row.generation_status = generation_status
        row.error_message = error_message
        return row

    def update_visibility(self, layer_id: str, visible: bool) -> DetectionLayerRow | None:
        """Update row visibility from a layer refresh."""
        row = self.find_by_layer_id(layer_id)
        if row is None:
            return None
        row.visible = bool(visible)
        return row

    def refresh_visibility_from_nodes(self, node_refs: Iterable[Any]) -> None:
        """Refresh row visibility from Krita node reference objects."""
        visibility_by_id: dict[str, bool] = {}
        for node_ref in node_refs:
            layer_id = str(getattr(node_ref, "id_string", ""))
            if not layer_id:
                continue
            visibility_by_id[layer_id] = bool(
                getattr(node_ref, "visible", getattr(node_ref, "is_visible", True))
            )
        for row in self.rows:
            if row.layer_id in visibility_by_id:
                row.visible = visibility_by_id[row.layer_id]

    def as_dicts(self, filter_mode: str | None = None) -> list[dict[str, Any]]:
        """Return filtered rows as JSON-serializable dictionaries."""
        return [row.to_metadata() for row in self.filtered_rows(filter_mode)]


def _normalize_filter(filter_mode: str | None) -> str:
    """Normalize empty filter values to all."""
    if filter_mode is None:
        return "all"
    text = str(filter_mode).strip().lower()
    return text or "all"