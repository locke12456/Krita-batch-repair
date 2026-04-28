"Repair plugin metadata schema constants."

from __future__ import annotations


SCHEMA_VERSION = 1

REPAIR_METADATA_KEY = "krita_repair_plugin/metadata"

KEY_SCHEMA_VERSION = "schema_version"
KEY_LAYER_ID = "layer_id"
KEY_LAYER_NAME = "layer_name"
KEY_DETECTOR_MODE = "detector_mode"
KEY_DETECTOR_LABEL = "detector_label"
KEY_DETECTOR_BBOX = "detector_bbox"
KEY_DETECTOR_BBOX_COORDINATE_SPACE = "detector_bbox_coordinate_space"
KEY_DETECTOR_SCORE = "detector_score"
KEY_DETECTOR_SELECTED = "detector_selected"
KEY_DETECTOR_ACTIVE = "detector_active"
KEY_PROMPT_WORKFLOW = "prompt_workflow"
KEY_PROMPT_TEXT = "prompt_text"
KEY_PROMPT_EXTRACTED = "prompt_extracted"
KEY_PROMPT_RAW_OUTPUT = "prompt_raw_output"
KEY_GENERATION_STATUS = "generation_status"
KEY_GENERATION_HANDOFF = "generation_handoff"
KEY_ERROR_MESSAGE = "error_message"

STATUS_NOT_STARTED = "not_started"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


def base_payload(layer_id: str = "", layer_name: str = "") -> dict[str, object]:
    "Return the common base metadata payload."
    return {
        KEY_SCHEMA_VERSION: SCHEMA_VERSION,
        KEY_LAYER_ID: str(layer_id or ""),
        KEY_LAYER_NAME: str(layer_name or ""),
    }
