"""Independent log/report Docker panel for the repair plugin."""

from __future__ import annotations

from typing import Any

from .repair_compat import (
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

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Report"))
        layout.addWidget(self._report_text)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self._log_text)
        layout.addWidget(self._clear_button)

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
        """Append a log line and auto-scroll to the bottom."""
        self.show()
        self._log_text.appendPlainText(str(text or ""))
        scrollbar = self._log_text.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def set_report(self, text: str) -> None:
        """Set the report section, replacing previous content."""
        self.show()
        self._report_text.setPlainText(str(text or ""))

    def clear(self) -> None:
        """Clear all log and report content."""
        self._report_text.clear()
        self._log_text.clear()