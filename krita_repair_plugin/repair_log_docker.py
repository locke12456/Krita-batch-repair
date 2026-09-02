"""Independent log/report Docker panel for the repair plugin."""

from __future__ import annotations

import time
from typing import Any

from .repair_compat import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from .repair_compat import QPlainTextEdit
except (ImportError, AttributeError):
    from .repair_compat import QtWidgets
    QPlainTextEdit = QtWidgets.QPlainTextEdit

try:
    from krita import DockWidget
except Exception:
    DockWidget = QWidget


class RepairLogDocker(DockWidget):
    """Independent log and report display panel.

    Created and managed by RepairDocker. Lifecycle is synchronized
    with the main docker: show together, close together.
    """

    def __init__(self, parent: Any = None) -> None:
        try:
            super().__init__(parent)
        except TypeError:
            super().__init__()

        if hasattr(self, "setWindowTitle"):
            self.setWindowTitle("Repair Log")

        self._report_text = QPlainTextEdit()
        self._report_text.setReadOnly(True)
        self._report_text.setMaximumHeight(120)

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)

        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self.clear)

        self._copy_button = QPushButton("Copy All")
        self._copy_button.clicked.connect(self.copy_all)

        button_row = QHBoxLayout()
        button_row.addWidget(self._copy_button)
        button_row.addWidget(self._clear_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Report"))
        layout.addWidget(self._report_text)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self._log_text)
        layout.addLayout(button_row)

        root = QWidget()
        root.setLayout(layout)

        if hasattr(self, "setWidget"):
            self.setWidget(root)
        elif hasattr(self, "setLayout"):
            self.setLayout(layout)

    def canvasChanged(self, canvas: Any) -> None:
        """Krita docker callback kept intentionally lightweight."""
        return None

    def append_log(self, text: str) -> None:
        """Append a timestamped log entry and auto-scroll to the bottom.

        Multi-line entries keep the timestamp on the first line only, so a
        traceback stays readable as one block.
        """
        self.show()
        stamp = time.strftime("%H:%M:%S")
        lines = str(text or "").splitlines() or [""]
        entry = "\n".join(
            [f"{stamp} {lines[0]}"] + [f"         {line}" for line in lines[1:]]
        )
        self._log_text.appendPlainText(entry)
        scrollbar = self._log_text.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def copy_all(self) -> None:
        """Copy the report and full log to the clipboard for bug reports."""
        text = (
            "=== Report ===\n"
            f"{self._report_text.toPlainText()}\n"
            "=== Log ===\n"
            f"{self._log_text.toPlainText()}"
        )
        try:
            from .repair_compat import QtWidgets

            clipboard = QtWidgets.QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
                return
        except Exception:
            pass
        print(text)

    def set_report(self, text: str) -> None:
        """Set the report section, replacing previous content."""
        self.show()
        self._report_text.setPlainText(str(text or ""))

    def clear(self) -> None:
        """Clear all log and report content."""
        self._report_text.clear()
        self._log_text.clear()