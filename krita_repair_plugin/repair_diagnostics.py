"""Failure diagnostics shared by repair plugin services.

Every failure path in this plugin must produce a non-empty, traceable
message. ``str(exc)`` alone is not enough: several exception types render
as an empty string, which surfaced in the log docker as a bare "failed"
with nothing to investigate.
"""

from __future__ import annotations

import os
import traceback
from typing import Callable

TRACEBACK_FRAME_LIMIT = 8
CAUSE_CHAIN_LIMIT = 3


def exception_summary(exc: BaseException) -> str:
    """Return a one-line ``TypeName: message`` that is never empty."""
    name = type(exc).__name__
    try:
        message = str(exc).strip()
    except Exception:
        message = ""
    if not message:
        args = getattr(exc, "args", ())
        message = repr(args) if args else "<no message>"
    return f"{name}: {message}"


def exception_detail(exc: BaseException, frame_limit: int = TRACEBACK_FRAME_LIMIT) -> str:
    """Return the summary plus a compact traceback tail and cause chain."""
    lines = [exception_summary(exc)]

    tb = getattr(exc, "__traceback__", None)
    if tb is not None:
        try:
            frames = traceback.extract_tb(tb)[-frame_limit:]
        except Exception:
            frames = []
        for frame in frames:
            source = str(frame.line or "").strip()
            location = f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}()"
            lines.append(f"    at {location}" + (f" -> {source}" if source else ""))

    seen = {id(exc)}
    cause = exc.__cause__ or exc.__context__
    depth = 0
    while cause is not None and id(cause) not in seen and depth < CAUSE_CHAIN_LIMIT:
        seen.add(id(cause))
        lines.append(f"    caused by {exception_summary(cause)}")
        depth += 1
        cause = cause.__cause__ or cause.__context__

    return "\n".join(lines)


def emit_log(log_callback: Callable[[str], None] | None, text: str) -> None:
    """Send text to a log callback, falling back to stdout. Never raises."""
    message = str(text or "")
    if callable(log_callback):
        try:
            log_callback(message)
            return
        except Exception as exc:
            print(f"[repair_diagnostics] log callback failed: {exception_summary(exc)}")
    print(message)


def log_exception(
    log_callback: Callable[[str], None] | None,
    context: str,
    exc: BaseException,
) -> str:
    """Log a failure with full detail and return the detail string."""
    detail = exception_detail(exc)
    emit_log(log_callback, f"{context}\n{detail}")
    return detail
