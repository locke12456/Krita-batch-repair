"""Group-based remove background service."""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
from typing import Any

from .group_selection_model import RepairGroupRow
from .prompt_extraction_service import PromptExtractionService, PromptWorkflowClient
from .repair_compat import QtCore, QtGui, active_krita_document, render_node_projection


INPUT_NODE_ID = "1"
INFERENCE_SCALE_NODE_ID = "2"
MODEL_NODE_ID = "3"
MASK_NODE_ID = "4"
MASK_IMAGE_NODE_ID = "5"
RESTORE_SCALE_NODE_ID = "6"
BASE64_OUTPUT_NODE_ID = "7"
TEXT_OUTPUT_NODE_ID = "8"
IMAGE_INPUT_KEY = "image"

REMOVE_BACKGROUND_WORKFLOW_TEMPLATE = {
    "1": {
        "inputs": {
            "image": "",
        },
        "class_type": "ETN_LoadImageBase64",
        "_meta": {
            "title": "Group Image Base64",
        },
    },
    "2": {
        "inputs": {
            "upscale_method": "bilinear",
            "width": 1024,
            "height": 1024,
            "crop": "disabled",
            "image": ["1", 0],
        },
        "class_type": "ImageScale",
        "_meta": {
            "title": "Scale Image For BiRefNet",
        },
    },
    "3": {
        "inputs": {
            "model": "General-Lite.safetensors",
            "device": "AUTO",
            "use_weight": False,
            "dtype": "float32",
        },
        "class_type": "LoadRembgByBiRefNetModel",
        "_meta": {
            "title": "Load BiRefNet Remove Background Model",
        },
    },
    "4": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "upscale_method": "bilinear",
            "mask_threshold": 0,
            "model": ["3", 0],
            "images": ["2", 0],
        },
        "class_type": "GetMaskByBiRefNet",
        "_meta": {
            "title": "Get Remove Background Mask",
        },
    },
    "5": {
        "inputs": {
            "mask": ["4", 0],
        },
        "class_type": "MaskToImage",
        "_meta": {
            "title": "Convert Mask To Image",
        },
    },
    "6": {
        "inputs": {
            "upscale_method": "bilinear",
            "width": 1,
            "height": 1,
            "crop": "disabled",
            "image": ["5", 0],
        },
        "class_type": "ImageScale",
        "_meta": {
            "title": "Restore Mask To Source Extent",
        },
    },
    "7": {
        "inputs": {
            "image": ["6", 0],
        },
        "class_type": "easy imageToBase64",
        "_meta": {
            "title": "Mask Image To Base64",
        },
    },
    "8": {
        "inputs": {
            "text": ["7", 0],
        },
        "class_type": "ShowText|pysssss",
        "_meta": {
            "title": "Mask Base64 Text Output",
        },
    },
}


@dataclass(slots=True)
class RemoveBackgroundReport:
    """Report for one remove background attempt."""

    group_name: str
    export_key: str
    status: str
    reason: str = ""
    mask_layer_id: str = ""
    error: str = ""


class RemoveBackgroundService:
    """Run remove background for selected group rows and attach transparency masks."""

    def __init__(
        self,
        prompt_service: PromptExtractionService | None = None,
        client_factory: Any | None = None,
    ) -> None:
        self.prompt_service = prompt_service or PromptExtractionService()
        self.client_factory = client_factory or PromptWorkflowClient

    def remove_for_rows(
        self,
        rows: list[RepairGroupRow],
    ) -> list[RemoveBackgroundReport]:
        """Run remove background for selected resolved group rows."""
        reports: list[RemoveBackgroundReport] = []
        for row in rows:
            if getattr(row.record, "target_type", "") != "group":
                reports.append(
                    self._report(row, "skipped", reason="not a group record")
                )
                continue
            if not row.is_resolved:
                reports.append(
                    self._report(row, "skipped", reason="group unresolved")
                )
                continue
            if row.group_layer is None:
                reports.append(
                    self._report(row, "skipped", reason="group layer missing")
                )
                continue

            try:
                rendered = render_node_projection(row.group_layer)
                projection_bounds = rendered.bounds
                projection_png = bytes(rendered.to_bytes())
                if not projection_png:
                    reports.append(
                        self._report(
                            row,
                            "failed",
                            error="Group projection rendered empty bytes.",
                        )
                    )
                    continue

                width = int(getattr(projection_bounds, "width", 0) or 0)
                height = int(getattr(projection_bounds, "height", 0) or 0)
                if width <= 0 or height <= 0:
                    image = QtGui.QImage()
                    if image.loadFromData(projection_png, "PNG"):
                        width = int(image.width())
                        height = int(image.height())
                if width <= 0 or height <= 0:
                    reports.append(
                        self._report(
                            row,
                            "failed",
                            error="Group projection bounds are empty.",
                        )
                    )
                    continue

                mask_png_bytes = self._run_workflow(projection_png, width, height)
                mask_layer_id = self._apply_mask_to_group(
                    row,
                    mask_png_bytes,
                    projection_bounds,
                )
                reports.append(
                    self._report(
                        row,
                        "success",
                        mask_layer_id=mask_layer_id,
                    )
                )
            except Exception as exc:
                reports.append(self._report(row, "failed", error=str(exc)))

        return reports

    def build_workflow(
        self,
        image_base64: str = "",
        width: int = 1,
        height: int = 1,
    ) -> dict[str, Any]:
        """Build the remove background workflow with image and restore size injected."""
        workflow = copy.deepcopy(REMOVE_BACKGROUND_WORKFLOW_TEMPLATE)
        workflow[INPUT_NODE_ID]["inputs"][IMAGE_INPUT_KEY] = image_base64
        workflow[RESTORE_SCALE_NODE_ID]["inputs"]["width"] = max(1, int(width))
        workflow[RESTORE_SCALE_NODE_ID]["inputs"]["height"] = max(1, int(height))
        return workflow

    def _run_workflow(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
    ) -> bytes:
        """Run the raw ComfyUI remove background workflow and return mask PNG bytes."""
        if not image_bytes:
            raise RuntimeError("Group projection PNG bytes are required.")

        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        workflow = self.build_workflow(image_base64, width, height)
        client = self.client_factory(self.prompt_service.connected_comfy_url())
        prompt_id = client.submit(workflow)
        raw_output = client.wait(prompt_id)
        mask_base64 = self._extract_mask_base64(raw_output, prompt_id)
        if not mask_base64:
            raise RuntimeError("Runtime output did not contain mask base64 text.")

        try:
            return base64.b64decode(mask_base64)
        except Exception as exc:
            raise RuntimeError(f"Mask base64 output could not be decoded: {exc}") from exc

    def _apply_mask_to_group(
        self,
        row: RepairGroupRow,
        mask_png_bytes: bytes,
        projection_bounds: Any,
    ) -> str:
        """Attach the decoded mask image as a Krita transparency mask under the group."""
        if not mask_png_bytes:
            raise RuntimeError("Mask PNG bytes are empty.")

        document_ref = active_krita_document()
        if document_ref is None:
            raise RuntimeError("No active Krita document.")

        document = getattr(document_ref, "document", None)
        create_mask = getattr(document, "createTransparencyMask", None)
        if not callable(create_mask):
            raise RuntimeError("Krita document does not expose createTransparencyMask().")

        target_node = getattr(row.group_layer, "node", row.group_layer)
        add_child = getattr(target_node, "addChildNode", None)
        child_nodes = getattr(target_node, "childNodes", None)
        if not callable(add_child) or not callable(child_nodes):
            raise RuntimeError("Group node cannot accept or report child mask nodes.")

        image = QtGui.QImage()
        if not image.loadFromData(mask_png_bytes, "PNG"):
            raise ValueError("Remove background mask PNG bytes could not be decoded.")

        values = self._image_to_grayscale_bytes(image)
        width = int(image.width())
        height = int(image.height())
        x0 = int(getattr(projection_bounds, "x", 0) or 0)
        y0 = int(getattr(projection_bounds, "y", 0) or 0)

        mask_node = create_mask("Remove Background Mask")
        try:
            mask_node.setName("Remove Background Mask")
        except Exception:
            pass

        set_pixel_data = getattr(mask_node, "setPixelData", None)
        if not callable(set_pixel_data):
            raise RuntimeError("Transparency mask node does not expose setPixelData().")

        set_pixel_data(QtCore.QByteArray(values), x0, y0, width, height)
        add_child(mask_node, None)

        children = list(child_nodes() or [])
        mask_children = [
            child
            for child in children
            if str(getattr(child, "type", lambda: "")()).lower() == "transparencymask"
        ]
        if mask_node not in children and not mask_children:
            raise RuntimeError("Transparency mask was created but is not attached under the group.")

        if callable(getattr(document_ref, "refresh_projection", None)):
            document_ref.refresh_projection()

        unique_id = getattr(mask_node, "uniqueId", None)
        if callable(unique_id):
            try:
                return str(unique_id().toString())
            except Exception:
                pass
        return str(getattr(mask_node, "name", lambda: "Remove Background Mask")() or "Remove Background Mask")

    def _extract_mask_base64(self, raw_output: Any, prompt_id: str) -> str:
        """Extract mask base64 text from known ComfyUI history payload shapes."""
        prompt_payload = self._prompt_payload(raw_output, prompt_id)
        outputs = prompt_payload.get("outputs") if isinstance(prompt_payload, dict) else None
        if isinstance(outputs, dict):
            for node_id in (
                TEXT_OUTPUT_NODE_ID,
                int(TEXT_OUTPUT_NODE_ID),
                BASE64_OUTPUT_NODE_ID,
                int(BASE64_OUTPUT_NODE_ID),
                "54",
                54,
                "53",
                53,
            ):
                text = self._extract_text(outputs.get(node_id))
                if text:
                    return self._normalize_base64_text(text)

        text = self._extract_text(raw_output)
        return self._normalize_base64_text(text)

    def _prompt_payload(self, raw_output: Any, prompt_id: str) -> dict[str, Any]:
        """Return the prompt-specific payload from a history response."""
        if not isinstance(raw_output, dict):
            return {}
        payload = raw_output.get(prompt_id) or raw_output.get(str(prompt_id))
        if isinstance(payload, dict):
            return payload
        return raw_output

    def _extract_text(self, payload: Any) -> str:
        """Extract text from common ComfyUI text-like payload shapes."""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts = [self._extract_text(item) for item in payload]
            return "\n".join(part for part in parts if part).strip()
        if not isinstance(payload, dict):
            return str(payload).strip()

        ui = payload.get("ui")
        if isinstance(ui, dict):
            text = self._extract_text(ui.get("text"))
            if text:
                return text

        output = payload.get("output")
        if output is not None:
            text = self._extract_text(output)
            if text:
                return text

        for key in ("text", "string", "strings", "base64", "image", "images", "preview"):
            value = payload.get(key)
            text = self._extract_text(value)
            if text:
                return text

        return ""

    def _normalize_base64_text(self, text: str) -> str:
        """Strip optional data URI prefix and surrounding whitespace."""
        value = str(text or "").strip()
        if "," in value and value.lower().startswith("data:"):
            value = value.split(",", 1)[1].strip()
        return value

    def _image_to_grayscale_bytes(self, image: Any) -> bytes:
        """Convert a QImage into one byte per pixel for a Krita transparency mask."""
        image = image.convertToFormat(QtGui.QImage.Format_Grayscale8)
        width = int(image.width())
        height = int(image.height())
        ptr = image.bits()
        try:
            ptr.setsize(image.sizeInBytes())
        except Exception:
            ptr.setsize(image.byteCount())
        raw = bytes(ptr)

        bytes_per_line = int(image.bytesPerLine())
        if bytes_per_line == width:
            return raw[: width * height]

        values = bytearray()
        for y in range(height):
            start = y * bytes_per_line
            values.extend(raw[start:start + width])
        return bytes(values)

    def _report(
        self,
        row: RepairGroupRow,
        status: str,
        *,
        reason: str = "",
        mask_layer_id: str = "",
        error: str = "",
    ) -> RemoveBackgroundReport:
        """Build a report for one group row."""
        return RemoveBackgroundReport(
            group_name=row.group_name or "",
            export_key=row.export_key,
            status=status,
            reason=reason,
            mask_layer_id=mask_layer_id,
            error=error,
        )