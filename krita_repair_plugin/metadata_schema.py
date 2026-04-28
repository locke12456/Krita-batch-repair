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
KEY_GENERATION_JOB_ID = "generation_job_id"
KEY_GENERATION_RESULT_INDEX = "generation_result_index"
KEY_GENERATION_JOB_PARAMS = "generation_job_params"
KEY_RESULT_LAYER_ID = "result_layer_id"
KEY_RESULT_LAYER_NAME = "result_layer_name"
KEY_CANDIDATE_LAYER_IDS = "candidate_layer_ids"
KEY_ERROR_MESSAGE = "error_message"

KEY_SOURCE_GROUP_ID = "source_group_id"
KEY_SOURCE_GROUP_NAME = "source_group_name"
KEY_EXPORT_KEY = "export_key"
KEY_SOURCE_LAYER_ID = "source_layer_id"
KEY_SOURCE_LAYER_NAME = "source_layer_name"
KEY_CREATED_LAYER_ID = "created_layer_id"
KEY_CREATED_LAYER_NAME = "created_layer_name"

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
