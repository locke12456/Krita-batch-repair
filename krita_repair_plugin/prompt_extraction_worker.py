from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .prompt_extraction_service import PromptExtractionResult, PromptExtractionService
from .repair_result_model import RepairResultRow


ProgressCallback = Callable[["PromptExtractionProgress"], None]
RowCallback = Callable[[RepairResultRow, PromptExtractionResult | None], None]


@dataclass(slots=True)
class PromptExtractionProgress:
    total: int
    completed: int = 0
    current_result_id: str = ""
    current_label: str = ""
    cancelled: bool = False


class PromptExtractionWorker:
    def __init__(
        self,
        service: PromptExtractionService,
        on_progress: ProgressCallback | None = None,
        on_row_finished: RowCallback | None = None,
        on_completed: ProgressCallback | None = None,
        threshold: float | None = None,
    ) -> None:
        self.service = service
        self.on_progress = on_progress
        self.on_row_finished = on_row_finished
        self.on_completed = on_completed
        self.threshold = threshold
        self.rows: list[RepairResultRow] = []
        self.cancelled = False
        self._task: Any | None = None

    def enqueue(self, rows: Iterable[RepairResultRow]) -> None:
        self.rows = list(rows)
        self.cancelled = False
        total = len(self.rows)
        for index, row in enumerate(self.rows, start=1):
            row.mark_prompt_queued(index=index, total=total)

    def cancel(self) -> None:
        self.cancelled = True
        total = len(self.rows)
        for index, row in enumerate(self.rows, start=1):
            if row.prompt_status in {"queued", "running"}:
                row.mark_prompt_cancelled(index=index, total=total)

    def start(self) -> Any | None:
        if not self.rows:
            return None

        try:
            from ai_diffusion import eventloop

            self._task = eventloop.run(self.run_async())
            return self._task
        except Exception:
            self._task = asyncio.create_task(self.run_async())
            return self._task

    async def run_async(self) -> list[PromptExtractionResult]:
        results: list[PromptExtractionResult] = []
        total = len(self.rows)
        cache_key = f"tag[{self.threshold}]" if self.threshold is not None else "tag[0.8]"

        for index, row in enumerate(self.rows, start=1):
            if self.cancelled:
                row.mark_prompt_cancelled(index=index, total=total)
                self._emit_progress(index - 1, total, row, cancelled=True)
                self._emit_row_finished(row, None)
                continue

            # --- Group-level tagger cache check ---
            cached_prompt = self._get_group_tagger_cache(row, cache_key)
            if cached_prompt is not None:
                row.mark_prompt_done(
                    prompt_text=cached_prompt,
                    index=index,
                    total=total,
                )
                self._emit_progress(index, total, row)
                self._emit_row_finished(row, None)
                continue

            row.mark_prompt_running(index=index, total=total)
            self._emit_progress(index - 1, total, row)

            result = await asyncio.to_thread(
                self.service.extract_prompt_from_bytes,
                row.result_id,
                row.crop_png_bytes,
                self.threshold,
            )
            results.append(result)

            if self.cancelled:
                row.mark_prompt_cancelled(index=index, total=total)
                self._emit_progress(index, total, row, cancelled=True)
                self._emit_row_finished(row, result)
                continue

            if result.success:
                row.mark_prompt_done(
                    prompt_text=result.prompt_text,
                    raw_output=result.raw_output,
                    index=index,
                    total=total,
                )
                # --- Save to group-level tagger cache & persist ---
                self._set_group_tagger_cache(row, cache_key, result.prompt_text)
            else:
                row.mark_prompt_failed(
                    error=result.error_message,
                    index=index,
                    total=total,
                )

            self._emit_progress(index, total, row)
            self._emit_row_finished(row, result)

        final = PromptExtractionProgress(
            total=total,
            completed=min(total, len(results)),
            cancelled=bool(self.cancelled),
        )
        self._emit_completed(final)
        return results

    def _emit_progress(
        self,
        completed: int,
        total: int,
        row: RepairResultRow,
        cancelled: bool = False,
    ) -> None:
        callback = self.on_progress
        if not callable(callback):
            return
        callback(
            PromptExtractionProgress(
                total=int(total),
                completed=int(completed),
                current_result_id=row.result_id,
                current_label=row.display_name,
                cancelled=bool(cancelled),
            )
        )

    def _emit_row_finished(
        self,
        row: RepairResultRow,
        result: PromptExtractionResult | None,
    ) -> None:
        callback = self.on_row_finished
        if callable(callback):
            callback(row, result)

    def _emit_completed(self, progress: PromptExtractionProgress) -> None:
        callback = self.on_completed
        if callable(callback):
            callback(progress)

    # ---- Group-level tagger cache (tag[threshold]) ----

    @staticmethod
    def _parse_threshold(cache_key: str) -> str:
        """Extract threshold string from cache_key like 'tag[0.3]' -> '0.3'."""
        m = __import__("re").match(r"^tag\[(.+)\]$", cache_key)
        return m.group(1) if m else cache_key

    def _get_group_tagger_cache(
        self, row: RepairResultRow, cache_key: str,
    ) -> str | None:
        """Check group record.params_snapshot for cached tagger prompt."""
        record = row.record
        if record is None:
            return None
        snapshot = getattr(record, "params_snapshot", None) or {}
        # New path: metadata.tag_cache[threshold]
        threshold_str = self._parse_threshold(cache_key)
        metadata = snapshot.get("metadata")
        if isinstance(metadata, dict):
            tag_cache = metadata.get("tag_cache")
            if isinstance(tag_cache, dict):
                cached = tag_cache.get(threshold_str)
                if cached and isinstance(cached, str):
                    return cached
        # Fallback: old top-level key for un-migrated KRA files
        cached = snapshot.get(cache_key)
        if cached and isinstance(cached, str):
            return cached
        return None

    def _set_group_tagger_cache(
        self, row: RepairResultRow, cache_key: str, prompt_text: str,
    ) -> None:
        """Save tagger prompt cache into metadata.tag_cache namespace."""
        record = row.record
        if record is None:
            return

        snapshot = dict(getattr(record, "params_snapshot", None) or {})
        # Ensure metadata dict exists
        metadata = snapshot.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            snapshot["metadata"] = metadata
        # Ensure tag_cache sub-dict exists
        tag_cache = metadata.get("tag_cache")
        if not isinstance(tag_cache, dict):
            tag_cache = {}
            metadata["tag_cache"] = tag_cache
        # Write to safe namespace
        threshold_str = self._parse_threshold(cache_key)
        tag_cache[threshold_str] = str(prompt_text or "")
        record.params_snapshot = snapshot

        try:
            document_ref = getattr(row.source_layer, "document_ref", None)
            if document_ref is None:
                document_ref = getattr(row.group_layer, "document_ref", None)
            if document_ref is None:
                from .repair_compat import active_krita_document
                document_ref = active_krita_document()
            if document_ref is None:
                return

            from krita_ai_metadata.sync_map_store import SyncMapStore
            store = SyncMapStore(document_ref)
            store.record_apply(record)
        except Exception as exc:
            print(
                f"[PromptWorker] WARNING: failed to persist tagger cache "
                f"for key={cache_key}: {exc}"
            )