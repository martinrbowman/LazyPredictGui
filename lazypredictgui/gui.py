"""PySide6 GUI: import/label wolfSentry mic sessions (or load a
pre-featurized CSV directly), combine same-class sessions, split train/
validation/test, run LazyPredict across ~30 scikit-learn models, show a
ranked leaderboard, and display a confusion matrix for a chosen model.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import sounddevice as sd
from PySide6.QtCore import QSettings, Signal, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

ABOUT_TEXT = """\
<h3>LazyPredictGUI</h3>
<p>Originally designed for the wolfSentry project, but also supports CSV files.</p>
<p>Free for non-commercial use.</p>
<p>Request features - leave an issue/request on GitHub.</p>
<p>If you found this useful, <a href="https://buymeacoffee.com/redwolfelectronics">buy me a coffee</a>.</p>
"""

from .data import load_csv, resolve_label_column, three_way_split
from .dataset import DISCARD_LABEL, SessionStore
from .mic_session_reader import decode_session
from .workers import LazyPredictWorker, ModelEvalWorker
from .confusion_matrix_widget import ConfusionMatrixWidget

# No fixed class list - the app supports any number of classes (2 or more;
# LazyPredict/sklearn need at least 2 distinct labels present to train
# anything, but there's no upper bound here). "background"/"dog"/"speech"
# is just this project's own first use case, not a hardcoded limit - the
# label combo below is editable and remembers whatever class names get
# typed in, so a different deployment (or more classes added later, e.g.
# "footsteps"/"vehicle" per LoRa.md's SENSOR_EVENT types) works the same
# way with no code change.
DEFAULT_SEED_LABELS = ["background", "dog", "speech"]

GOERTZEL_PREFIX_COLS = ("rms", "zcr", "peak")
MFCC_PC_PREFIX_COLS = ("pc_rms", "pc_zcr", "pc_peak")


def _goertzel_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("goertzel_") or c in GOERTZEL_PREFIX_COLS]


def _mfcc_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("mfcc") or c in MFCC_PC_PREFIX_COLS]


class ImportLabelTab(QWidget):
    """Import & Label tab - the audio-specific path (import wolfSentry
    session files, play back, label, combine) plus a plain "Load CSV..."
    picker for the fully generic path."""

    datasetReady = Signal(object)  # pandas.DataFrame, emitted by "Continue" / "Load CSV"
    switchToSplitTab = Signal()  # ask MainWindow to flip to the next tab

    SESSION_COL = 0
    LABEL_COL = 1
    EXAMPLES_COL = 2

    def __init__(self, settings: QSettings):
        super().__init__()
        self.settings = settings
        self.store = SessionStore()

        layout = QVBoxLayout(self)

        # --- audio import/label ---
        import_row = QHBoxLayout()
        import_btn = QPushButton("Import session...")
        import_btn.clicked.connect(self._import_session)
        import_row.addWidget(import_btn)
        self.sub_clip_checkbox = QCheckBox("Split into 1-second sub-clips")
        self.sub_clip_checkbox.setToolTip(
            "Off (default): this session = 1 example, using the on-device Goertzel "
            "summary + whole-session MFCC. Needs many session recordings per class "
            "before a split means anything.\n"
            "On: this session = up to 30 examples (one per second), MFCC + PC RMS/ZCR/"
            "peak only (no on-device Goertzel per-second) - gets usable data out of "
            "just a couple of recordings. Applies to the NEXT import, not retroactively."
        )
        import_row.addWidget(self.sub_clip_checkbox)
        import_row.addStretch()
        layout.addLayout(import_row)

        self.session_table = QTableWidget(0, 3)
        self.session_table.setHorizontalHeaderLabels(["Session", "Label", "Examples"])
        self.session_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.session_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.session_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.session_table)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label selected session:"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        # Seeded with this project's own first use case, purely as a
        # convenience default - type any other class name and it's
        # remembered (see _assign_label) for the rest of this session, no
        # fixed limit on how many.
        self.label_combo.addItems(DEFAULT_SEED_LABELS + ["discard"])
        self.label_combo.setCurrentIndex(-1)
        label_row.addWidget(self.label_combo)
        assign_btn = QPushButton("Assign")
        assign_btn.clicked.connect(self._assign_label)
        label_row.addWidget(assign_btn)
        play_btn = QPushButton("Play")
        play_btn.clicked.connect(self._play_selected)
        label_row.addWidget(play_btn)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self._stop_playback)
        label_row.addWidget(stop_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)
        label_row.addWidget(remove_btn)
        label_row.addStretch()
        layout.addLayout(label_row)

        self.count_label = QLabel("No sessions labeled yet.")
        layout.addWidget(self.count_label)

        combine_row = QHBoxLayout()
        combine_btn = QPushButton("Combine & Continue")
        combine_btn.clicked.connect(self._combine_and_continue)
        combine_row.addWidget(combine_btn)
        combine_row.addStretch()
        layout.addLayout(combine_row)

        # --- generic CSV path ---
        csv_row = QHBoxLayout()
        csv_btn = QPushButton("Load CSV...")
        csv_btn.clicked.connect(self._load_csv)
        csv_row.addWidget(csv_btn)
        csv_row.addWidget(QLabel("Label column:"))
        self.label_col_combo = QComboBox()
        self.label_col_combo.setEditable(True)
        csv_row.addWidget(self.label_col_combo)
        apply_col_btn = QPushButton("Use this column")
        apply_col_btn.clicked.connect(self._apply_csv_label_column)
        csv_row.addWidget(apply_col_btn)
        csv_row.addStretch()
        layout.addLayout(csv_row)

        self._sessions: dict[str, object] = {}  # name -> MicSession, for playback
        self._raw_csv_df: pd.DataFrame | None = None

    def _import_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import mic session", "", "Bin files (*.bin)")
        if not path:
            return
        with open(path, "rb") as f:
            data = f.read()
        session = decode_session(data)
        name = path.rsplit("/", 1)[-1]
        sub_clip_mode = self.sub_clip_checkbox.isChecked()
        self.store.add(name, session, sub_clip_mode=sub_clip_mode)
        self._sessions[name] = session

        row = self.session_table.rowCount()
        self.session_table.insertRow(row)
        self.session_table.setItem(row, self.SESSION_COL, QTableWidgetItem(name))
        self.session_table.setItem(row, self.LABEL_COL, QTableWidgetItem(""))
        example_count = self.store.get(name).example_count()
        self.session_table.setItem(row, self.EXAMPLES_COL, QTableWidgetItem(str(example_count)))
        # Select the row we just added rather than leaving the previous
        # selection (or none) in place - Play/Assign act on whatever row is
        # actually selected, and a freshly-imported session is the obvious
        # next thing to listen to and label.
        self.session_table.setCurrentCell(row, self.SESSION_COL)

        if session.summary is None:
            QMessageBox.warning(
                self, "Incomplete session",
                f"{name} has no SESSION_SUMMARY record (aborted/truncated capture) - "
                "on-device Goertzel/RMS/ZCR/peak columns will be missing for this "
                "session, but it can still be labeled and used for MFCC."
            )

    def _current_row(self) -> int:
        return self.session_table.currentRow()

    def _current_name(self) -> str | None:
        row = self._current_row()
        if row < 0:
            return None
        item = self.session_table.item(row, self.SESSION_COL)
        return item.text() if item else None

    def _assign_label(self) -> None:
        row = self._current_row()
        name = self._current_name()
        if name is None:
            return
        label = self.label_combo.currentText().strip()
        if not label:
            return
        # Remember any newly-typed class name for reuse on the next
        # session - no fixed limit on how many distinct classes this can
        # grow to.
        if self.label_combo.findText(label) < 0:
            self.label_combo.addItem(label)
        self.store.set_label(name, DISCARD_LABEL if label == "discard" else label)
        self.session_table.setItem(row, self.LABEL_COL, QTableWidgetItem(label))
        self._update_counts()

    def _play_selected(self) -> None:
        name = self._current_name()
        if name is None:
            return
        session = self._sessions[name]
        if len(session.audio) == 0:
            return
        y = session.audio.astype(np.float32) / 32768.0
        sd.play(y, samplerate=8000)

    def _stop_playback(self) -> None:
        sd.stop()

    def _remove_selected(self) -> None:
        row = self._current_row()
        name = self._current_name()
        if name is None:
            return
        self.store.remove(name)
        self._sessions.pop(name, None)
        self.session_table.removeRow(row)
        self._update_counts()

    def _update_counts(self) -> None:
        counts = self.store.labeled_counts()
        if not counts:
            self.count_label.setText("No sessions labeled yet.")
        else:
            text = ", ".join(f"{label}: {n}" for label, n in sorted(counts.items()))
            self.count_label.setText(text)

    def _combine_and_continue(self) -> None:
        df = self.store.to_dataframe()
        if df.empty:
            QMessageBox.warning(self, "No data", "No labeled sessions to combine yet.")
            return

        counts = self.store.labeled_counts()
        summary = ", ".join(f"{label}: {n}" for label, n in sorted(counts.items()))
        QMessageBox.information(
            self, "Combined",
            f"Combined {len(df)} session(s) across {len(counts)} class(es):\n{summary}\n\n"
            "Moving to the Split & Leaderboard tab.",
        )
        self.datasetReady.emit(df)
        self.switchToSplitTab.emit()

    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV files (*.csv)")
        if not path:
            return
        self._raw_csv_df = load_csv(path)

        # Auto-detect BEFORE touching the combo box - QComboBox.addItems()
        # selects the first item by default, which would otherwise look
        # like an explicit user override on every fresh load (a real bug
        # this exact sequencing caused: renaming an unrelated column to
        # "label" while the CSV's own real "label" column was still
        # present, producing two same-named columns and crashing the
        # split). blockSignals so populating the combo doesn't itself
        # trigger anything.
        auto_label_col = resolve_label_column(self._raw_csv_df, override=None)
        self.label_col_combo.blockSignals(True)
        self.label_col_combo.clear()
        self.label_col_combo.addItems(list(self._raw_csv_df.columns))
        self.label_col_combo.setCurrentText(auto_label_col)
        self.label_col_combo.blockSignals(False)

        self._apply_csv_label_column()

    def _apply_csv_label_column(self) -> None:
        """Applies whatever column is currently selected in the combo as
        the label column - called after a fresh load (auto-detected) and
        again if the user manually changes the selection and clicks "Use
        this column"."""
        if self._raw_csv_df is None:
            return
        label_col = self.label_col_combo.currentText()
        if label_col not in self._raw_csv_df.columns:
            QMessageBox.warning(self, "Unknown column", f"{label_col!r} is not a column in "
                                 "the loaded CSV.")
            return

        df = self._raw_csv_df
        if label_col != "label":
            # Drop any pre-existing "label" column first - renaming a
            # different column to "label" must never leave two columns
            # with that name (that's the bug fixed above).
            df = df.drop(columns=["label"], errors="ignore").rename(columns={label_col: "label"})
        self.datasetReady.emit(df)
        self.switchToSplitTab.emit()


class SplitLeaderboardTab(QWidget):
    """Split & Leaderboard tab - feature-set picker + train/val/test split +
    LazyPredict leaderboard."""

    leaderboardUpdated = Signal()

    def __init__(self, settings: QSettings):
        super().__init__()
        self.settings = settings
        self._df: pd.DataFrame | None = None
        self._worker: LazyPredictWorker | None = None

        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.leaderboard: pd.DataFrame | None = None

        layout = QVBoxLayout(self)

        feature_row = QHBoxLayout()
        feature_row.addWidget(QLabel("Feature set:"))
        self.goertzel_cb = QCheckBox("Goertzel (on-device)")
        self.goertzel_cb.setChecked(True)
        self.mfcc_cb = QCheckBox("MFCC (PC-computed)")
        self.mfcc_cb.setChecked(True)
        feature_row.addWidget(self.goertzel_cb)
        feature_row.addWidget(self.mfcc_cb)
        feature_row.addStretch()
        layout.addLayout(feature_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Train %:"))
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        split_row.addWidget(self.train_spin)
        split_row.addWidget(QLabel("Val %:"))
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(15)
        split_row.addWidget(self.val_spin)
        split_row.addWidget(QLabel("Test %:"))
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(15)
        split_row.addWidget(self.test_spin)
        self.split_total_label = QLabel("Total: 100%")
        split_row.addWidget(self.split_total_label)
        split_row.addStretch()
        layout.addLayout(split_row)

        for spin in (self.train_spin, self.val_spin, self.test_spin):
            spin.valueChanged.connect(self._update_split_total)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run LazyPredict")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.run_btn)
        self.status_label = QLabel("Waiting for a dataset (see Import & Label tab).")
        run_row.addWidget(self.status_label)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self._update_split_total()

    def _update_split_total(self) -> None:
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        self.split_total_label.setText(f"Total: {total:.0f}%")
        ok = abs(total - 100.0) < 0.01
        self.split_total_label.setStyleSheet("" if ok else "color: red;")
        self.run_btn.setEnabled(ok and self._df is not None)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df
        self.status_label.setText(f"Dataset ready: {len(df)} examples.")
        self._update_split_total()

    def _selected_feature_columns(self, df: pd.DataFrame) -> list[str]:
        cols: list[str] = []
        if self.goertzel_cb.isChecked():
            cols += _goertzel_columns(df)
        if self.mfcc_cb.isChecked():
            cols += _mfcc_columns(df)
        return cols

    def _run(self) -> None:
        if self._df is None:
            return
        cols = self._selected_feature_columns(self._df)
        if not cols:
            if not self.goertzel_cb.isChecked() and not self.mfcc_cb.isChecked():
                message = "Pick at least one feature set (Goertzel and/or MFCC)."
            else:
                # A checkbox IS checked but the dataset has no matching
                # columns - the real cause is almost always that every
                # imported session used the *other* granularity mode (e.g.
                # Goertzel checked, but every session was imported with
                # "Split into 1-second sub-clips" on, which never produces
                # goertzel_* columns - the device only computes Goertzel
                # over a whole session, not per second).
                missing = []
                if self.goertzel_cb.isChecked() and not _goertzel_columns(self._df):
                    missing.append("Goertzel")
                if self.mfcc_cb.isChecked() and not _mfcc_columns(self._df):
                    missing.append("MFCC")
                message = (
                    f"No {' or '.join(missing)} columns found in this dataset. If this "
                    "came from the audio importer: Goertzel columns only exist for "
                    "sessions imported in whole-session mode (not \"Split into 1-second "
                    "sub-clips\"), and MFCC/PC columns only exist if the audio decoded "
                    "successfully. Check the other feature-set box, or re-import "
                    "sessions with the matching granularity."
                )
            QMessageBox.warning(self, "No matching features", message)
            return

        if len(self._df) < 2:
            QMessageBox.warning(
                self, "Not enough data",
                f"Only {len(self._df)} example(s) in the combined dataset - need at "
                "least 2 to split at all, and realistically many more per class for a "
                "meaningful leaderboard. Label more sessions first.",
            )
            return

        try:
            train_df, val_df, test_df = three_way_split(
                self._df, "label",
                self.train_spin.value() / 100.0,
                self.val_spin.value() / 100.0,
                self.test_spin.value() / 100.0,
            )
        except ValueError as exc:
            QMessageBox.critical(
                self, "Split failed",
                f"Couldn't split the dataset with these settings: {exc}\n\n"
                "This usually means there isn't enough data for the requested split - "
                "label more sessions, or adjust the train/val/test percentages.",
            )
            return

        self.X_train, self.y_train = train_df[cols], train_df["label"]
        self.X_val, self.y_val = val_df[cols], val_df["label"]
        self.X_test, self.y_test = test_df[cols], test_df["label"]

        self.run_btn.setEnabled(False)
        self.status_label.setText("Running LazyPredict across all models...")
        self._worker = LazyPredictWorker(self.X_train, self.X_val, self.y_train, self.y_val)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_success(self, leaderboard: pd.DataFrame) -> None:
        self.leaderboard = leaderboard
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Done - {len(leaderboard)} models ranked.")

        df = leaderboard.reset_index()
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        self.table.setRowCount(len(df))
        for row in range(len(df)):
            for col, name in enumerate(df.columns):
                self.table.setItem(row, col, QTableWidgetItem(str(df.iloc[row][name])))
        self.table.setSortingEnabled(True)

        self.leaderboardUpdated.emit()

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "LazyPredict failed", message)


class ConfusionMatrixTab(QWidget):
    def __init__(self, split_tab: SplitLeaderboardTab):
        super().__init__()
        self.split_tab = split_tab
        self._worker: ModelEvalWorker | None = None

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        row.addWidget(self.model_combo)
        eval_btn = QPushButton("Evaluate on Test Set")
        eval_btn.clicked.connect(self._evaluate)
        row.addWidget(eval_btn)
        row.addStretch()
        layout.addLayout(row)

        self.metrics_label = QLabel("")
        layout.addWidget(self.metrics_label)

        self.cm_widget = ConfusionMatrixWidget()
        layout.addWidget(self.cm_widget)

    def refresh_models(self) -> None:
        self.model_combo.clear()
        if self.split_tab.leaderboard is not None:
            self.model_combo.addItems([str(i) for i in self.split_tab.leaderboard.index])

    def _evaluate(self) -> None:
        model_name = self.model_combo.currentText()
        if not model_name or self.split_tab.X_train is None:
            return

        self.metrics_label.setText("Evaluating...")
        self._worker = ModelEvalWorker(
            model_name,
            self.split_tab.X_train, self.split_tab.X_val,
            self.split_tab.y_train, self.split_tab.y_val,
            self.split_tab.X_test, self.split_tab.y_test,
        )
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_success(self, result: dict) -> None:
        self.metrics_label.setText(
            f"Accuracy: {result['accuracy']:.3f}   F1 (weighted): {result['f1']:.3f}"
        )
        self.cm_widget.plot(np.asarray(result["confusion_matrix"]), result["labels"])

    def _on_failed(self, message: str) -> None:
        self.metrics_label.setText("Failed.")
        QMessageBox.critical(self, "Evaluation failed", message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LazyPredictGUI")
        self.resize(1200, 800)

        self.settings = QSettings("LazyPredictGUI", "LazyPredictGUI")

        import_tab = ImportLabelTab(self.settings)
        split_tab = SplitLeaderboardTab(self.settings)
        confusion_tab = ConfusionMatrixTab(split_tab)

        import_tab.datasetReady.connect(split_tab.set_dataframe)
        split_tab.leaderboardUpdated.connect(confusion_tab.refresh_models)

        tabs = QTabWidget()
        tabs.addTab(import_tab, "Import & Label")
        tabs.addTab(split_tab, "Split & Leaderboard")
        tabs.addTab(confusion_tab, "Confusion Matrix")
        self.setCentralWidget(tabs)

        import_tab.switchToSplitTab.connect(lambda: tabs.setCurrentWidget(split_tab))

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction("&About LazyPredictGUI")
        about_action.triggered.connect(self._show_about)

    def _show_about(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("About LazyPredictGUI")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(ABOUT_TEXT)
        box.exec()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
