"""Embeds a matplotlib confusion-matrix plot in a Qt widget - separate from
wolfsentry_downloader's map_view.py approach (QWebEngineView/Leaflet), which
is map-specific and irrelevant here. matplotlib's own Qt backend
(FigureCanvasQTAgg) is the simpler fit for a static plot like this.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ConfusionMatrixWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)

    def plot(self, cm: np.ndarray, labels: list[str]) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        im = ax.imshow(cm, cmap="Blues")
        self._figure.colorbar(im, ax=ax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        max_val = cm.max() if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > max_val / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

        self._figure.tight_layout()
        self._canvas.draw()
