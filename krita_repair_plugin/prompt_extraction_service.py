"""ComfyUI image-to-text prompt extraction service."""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import json
import time
from typing import Any
from urllib import error, request
import uuid

from .detection_layer_selection_model import DetectionLayerRow, DetectionLayerSelectionModel


STANDARD_IMAGE_NODE_ID = "1"
STANDARD_TAGGER_NODE_ID = "2"
STANDARD_OUTPUT_NODE_ID = "3"
STANDARD_IMAGE_INPUT_KEY = "image"

IMAGE2TAGGER_WORKFLOW_TEMPLATE = {
    "1": {
        "inputs": {
            "image": "",
        },
        "class_type": "ETN_LoadImageBase64",
        "_meta": {
            "title": "Input Image Base64",
        },
    },
    "2": {
        "inputs": {
            "model": "wd-eva02-large-tagger-v3",
            "threshold": 0.8,
            "character_threshold": 0.85,
            "replace_underscore": False,
            "trailing_comma": False,
            "exclude_tags": "uncensored, mosaic_censoring, censored",
            "tags": "",
            "image": ["1", 0],
        },
        "class_type": "WD14Tagger|pysssss",
        "_meta": {
            "title": "WD14 Prompt Tagger",
        },
    },
    "3": {
        "inputs": {
            "preview": "",
            "previewMode": False,
            "source": ["2", 0],
        },
        "class_type": "PreviewAny",
        "_meta": {
            "title": "Prompt Text Output",
        },
    },
}


@dataclass(frozen=True, slots=True)
class PromptExtractionResult:
    """Result of one candidate prompt extraction."""

    layer_id: str
    prompt_text: str
    raw_output: Any | None
    success: bool
    error_message: str = ""


class PromptWorkflowClient:
    """Small raw ComfyUI prompt runner for image-to-text workflows."""

    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.client_id = str(uuid.uuid4())

    def submit(self, workflow: dict[str, Any]) -> str:
        """Submit a raw workflow to ComfyUI /prompt and return prompt_id."""
        prompt_id = str(uuid.uuid4())
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
            "prompt_id": prompt_id,
        }
        response = self._post_json("prompt", payload)
        return str(response.get("prompt_id") or prompt_id)

    def wait(self, prompt_id: str) -> dict[str, Any]:
        """Wait for a prompt to appear in history with outputs."""
        deadline = time.monotonic() + self.timeout_seconds
        last_payload: dict[str, Any] = {}
        while time.monotonic() < deadline:
            payload = self.history(prompt_id)
            last_payload = payload
            prompt_payload = payload.get(prompt_id) or payload.get(str(prompt_id)) or payload
            outputs = prompt_payload.get("outputs") if isinstance(prompt_payload, dict) else None
            if outputs:
                return payload
            time.sleep(0.5)
        raise TimeoutError(f"Prompt extraction timed out for prompt_id={prompt_id}: {last_payload}")

    def history(self, prompt_id: str) -> dict[str, Any]:
        """Fetch ComfyUI history for a prompt id."""
        return self._get_json(f"history/{prompt_id}")

    def _post_json(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Post JSON to ComfyUI."""
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/{op}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_json(self, op: str) -> dict[str, Any]:
        """Get JSON from ComfyUI."""
        with request.urlopen(f"{self.base_url}/{op}", timeout=10.0) as response:
            return json.loads(response.read().decode("utf-8"))


class PromptExtractionService:
    """Extract text prompts from selected active detector candidate layer images."""

    def __init__(
        self,
        selection_model: DetectionLayerSelectionModel | None = None,
        metadata_service: Any | None = None,
        client_factory: Any | None = None,
        workflow_path: str = "",
    ) -> None:
        self.selection_model = selection_model
        self.metadata_service = metadata_service
        self.client_factory = client_factory or PromptWorkflowClient

    def set_workflow_path(self, workflow_path: str) -> None:
        """Compatibility no-op; prompt extraction uses the built-in workflow."""
        return None

    def build_image2tagger_workflow(
        self, image_base64: str = "", threshold: float | None = None,
    ) -> dict[str, Any]:
        """Return the built-in image2tagger workflow with optional image injection."""
        workflow = copy.deepcopy(IMAGE2TAGGER_WORKFLOW_TEMPLATE)
        workflow[STANDARD_IMAGE_NODE_ID]["inputs"][STANDARD_IMAGE_INPUT_KEY] = image_base64
        if threshold is not None:
            workflow[STANDARD_TAGGER_NODE_ID]["inputs"]["threshold"] = float(threshold)
        return workflow

    def extract_prompt_from_bytes(
        self,
        layer_id: str,
        image_bytes: bytes,
        threshold: float | None = None,
    ) -> PromptExtractionResult:
        """Run prompt extraction directly from bbox crop PNG bytes."""
        if not image_bytes:
            return PromptExtractionResult(
                layer_id=str(layer_id),
                prompt_text="",
                raw_output=None,
                success=False,
                error_message="Image bytes are required",
            )

        try:
            workflow = self.build_image2tagger_workflow(threshold=threshold)
            client = self.client_factory(self.connected_comfy_url())
            runtime_workflow = self.inject_image(workflow, image_bytes)
            prompt_id = client.submit(runtime_workflow)
            raw_output = client.wait(prompt_id)
            prompt_text = self.parse_prompt_output(raw_output, prompt_id)
            if not prompt_text:
                raise RuntimeError("Runtime output did not contain prompt text")
            return PromptExtractionResult(
                layer_id=str(layer_id),
                prompt_text=prompt_text,
                raw_output=raw_output,
                success=True,
            )
        except Exception as exc:
            return PromptExtractionResult(
                layer_id=str(layer_id),
                prompt_text="",
                raw_output=None,
                success=False,
                error_message=str(exc),
            )

    def run_for_selected(
        self,
        workflow_path: str | None = None,
        filter_mode: str | None = None,
    ) -> list[PromptExtractionResult]:
        """Run prompt extraction for selected active rows."""
        if self.selection_model is None:
            raise RuntimeError("selection_model is required")
        rows = self.selection_model.selected_active_rows(filter_mode)
        return self.run_for_rows(rows, None)

    def run_for_rows(
        self,
        rows: list[DetectionLayerRow],
        workflow_path: str | None = None,
    ) -> list[PromptExtractionResult]:
        """Run prompt extraction for explicit rows."""
        workflow = self.build_image2tagger_workflow()
        base_url = self.connected_comfy_url()
        client = self.client_factory(base_url)

        results: list[PromptExtractionResult] = []
        for row in rows:
            result = self.extract_prompt_for_row(row, workflow, client)
            results.append(result)
        return results

    def extract_prompt_for_row(
        self,
        row: DetectionLayerRow,
        workflow: dict[str, Any],
        client: PromptWorkflowClient,
    ) -> PromptExtractionResult:
        """Run prompt extraction for one row."""
        if not row.image_bytes:
            result = PromptExtractionResult(
                layer_id=row.layer_id,
                prompt_text="",
                raw_output=None,
                success=False,
                error_message="Candidate row does not contain image bytes",
            )
            self._write_result(row, result)
            return result

        try:
            runtime_workflow = self.inject_image(workflow, row.image_bytes)
            prompt_id = client.submit(runtime_workflow)
            raw_output = client.wait(prompt_id)
            prompt_text = self.parse_prompt_output(raw_output, prompt_id)
            if not prompt_text:
                raise RuntimeError("Runtime output did not contain prompt text")
            result = PromptExtractionResult(
                layer_id=row.layer_id,
                prompt_text=prompt_text,
                raw_output=raw_output,
                success=True,
            )
        except Exception as exc:
            result = PromptExtractionResult(
                layer_id=row.layer_id,
                prompt_text="",
                raw_output=None,
                success=False,
                error_message=str(exc),
            )

        self._write_result(row, result)
        return result

    def load_workflow(self, workflow_path: str) -> dict[str, Any]:
        """Compatibility helper; runtime prompt extraction uses the built-in workflow."""
        with open(workflow_path, "r", encoding="utf-8") as handle:
            workflow = json.load(handle)
        self.validate_standard_workflow(workflow)
        return workflow

    def validate_standard_workflow(self, workflow: dict[str, Any]) -> None:
        """Validate the standard node mapping used by this service."""
        node = workflow.get(STANDARD_IMAGE_NODE_ID)
        if not isinstance(node, dict):
            raise ValueError("Workflow is missing node 1")
        if node.get("class_type") != "ETN_LoadImageBase64":
            raise ValueError("Workflow node 1 must be ETN_LoadImageBase64")
        inputs = node.setdefault("inputs", {})
        if STANDARD_IMAGE_INPUT_KEY not in inputs:
            inputs[STANDARD_IMAGE_INPUT_KEY] = ""

    def inject_image(self, workflow: dict[str, Any], image_bytes: bytes) -> dict[str, Any]:
        """Return a workflow copy with row image bytes injected as base64."""
        result = copy.deepcopy(workflow)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        result[STANDARD_IMAGE_NODE_ID]["inputs"][STANDARD_IMAGE_INPUT_KEY] = encoded
        return result

    def connected_comfy_url(self) -> str:
        """Read the active ComfyUI URL from krita-ai-diffusion connection state."""
        try:
            from ai_diffusion.connection import ConnectionState
            from ai_diffusion.root import root
        except Exception as exc:
            raise RuntimeError(f"Krita AI Diffusion connection API unavailable: {exc}") from exc

        connection = getattr(root, "connection", None)
        if connection is None:
            raise RuntimeError("Krita AI Diffusion connection is unavailable")
        if getattr(connection, "state", None) is not ConnectionState.connected:
            raise RuntimeError("Krita AI Diffusion is not connected to ComfyUI")

        client = getattr(connection, "client_if_connected", None)
        if client is None:
            raise RuntimeError("Connected ComfyUI client is unavailable")

        url = str(getattr(client, "url", "") or "").strip()
        if not url:
            raise RuntimeError("Connected ComfyUI client URL is unavailable")
        return url

    def parse_prompt_output(self, raw_output: Any, prompt_id: str) -> str:
        """Parse prompt text from runtime history or executed output."""
        prompt_payload = self._prompt_payload(raw_output, prompt_id)
        outputs = prompt_payload.get("outputs") if isinstance(prompt_payload, dict) else None
        if isinstance(outputs, dict):
            for node_id in (STANDARD_OUTPUT_NODE_ID, int(STANDARD_OUTPUT_NODE_ID)):
                text = self._extract_text(outputs.get(node_id))
                if text:
                    return text
            for node_id in (STANDARD_TAGGER_NODE_ID, int(STANDARD_TAGGER_NODE_ID)):
                text = self._extract_text(outputs.get(node_id))
                if text:
                    return text

        text = self._extract_text(raw_output)
        return text or ""

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
            return ", ".join(part for part in parts if part).strip()
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

        for key in ("text", "string", "strings", "tags", "preview"):
            value = payload.get(key)
            text = self._extract_text(value)
            if text:
                return text

        return ""

    def _write_result(self, row: DetectionLayerRow, result: PromptExtractionResult) -> None:
        """Write prompt extraction result to row and optional layer metadata."""
        row.prompt_text = result.prompt_text
        row.prompt_extracted = result.success
        row.raw_prompt_output = result.raw_output
        row.error_message = result.error_message

        service = self.metadata_service
        if service is None:
            return
        attach = getattr(service, "attach_prompt_metadata", None)
        if callable(attach):
            attach(row.layer_id, row.to_metadata())