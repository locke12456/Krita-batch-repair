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
    ) -> None:
        self.service = service
        self.on_progress = on_progress
        self.on_row_finished = on_row_finished
        self.on_completed = on_completed
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

        for index, row in enumerate(self.rows, start=1):
            if self.cancelled:
                row.mark_prompt_cancelled(index=index, total=total)
                self._emit_progress(index - 1, total, row, cancelled=True)
                self._emit_row_finished(row, None)
                continue

            row.mark_prompt_running(index=index, total=total)
            self._emit_progress(index - 1, total, row)

            result = await asyncio.to_thread(
                self.service.extract_prompt_from_bytes,
                row.result_id,
                row.crop_png_bytes,
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