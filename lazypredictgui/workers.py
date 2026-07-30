"""Background QThread workers - mirrors wolfsentry_downloader/gui.py's
SightingDownloadWorker/RankingDownloadWorker shape (succeeded/failed
Signals, run() does the blocking work) so the GUI doesn't block while
LazyClassifier fits ~30 models or a final model retrains.
"""

from __future__ import annotations

import pandas as pd
from lazypredict.Supervised import LazyClassifier
from PySide6.QtCore import QThread, Signal
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import all_estimators


class LazyPredictWorker(QThread):
    succeeded = Signal(object)  # pandas.DataFrame leaderboard
    failed = Signal(str)

    def __init__(self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series,
                 y_val: pd.Series):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def run(self) -> None:
        try:
            clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False)
            leaderboard, _predictions = clf.fit(self.X_train, self.X_val, self.y_train,
                                                 self.y_val)
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the GUI
            self.failed.emit(f"LazyPredict failed: {exc}")
            return

        self.succeeded.emit(leaderboard)


def _resolve_estimator_class(model_name: str):
    """Maps a lazypredict leaderboard row name (the sklearn class name,
    documented lazypredict behavior) back to a real sklearn class - avoids
    depending on lazypredict's own private internals, which differ across
    versions."""
    for name, cls in all_estimators(type_filter="classifier"):
        if name == model_name:
            return cls
    raise ValueError(f"could not resolve {model_name!r} to a known sklearn classifier")


class ModelEvalWorker(QThread):
    succeeded = Signal(dict)  # {"confusion_matrix", "labels", "accuracy", "f1"}
    failed = Signal(str)

    def __init__(self, model_name: str, X_train: pd.DataFrame, X_val: pd.DataFrame,
                 y_train: pd.Series, y_val: pd.Series, X_test: pd.DataFrame,
                 y_test: pd.Series):
        super().__init__()
        self.model_name = model_name
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def run(self) -> None:
        try:
            estimator_cls = _resolve_estimator_class(self.model_name)

            # Faithful retrain of what the leaderboard actually ranked -
            # LazyClassifier fits each model inside its own preprocessing
            # pipeline (imputation + scaling), so mirror that here rather
            # than fitting the raw estimator directly.
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", estimator_cls()),
            ])

            X_train_val = pd.concat([self.X_train, self.X_val])
            y_train_val = pd.concat([self.y_train, self.y_val])

            encoder = LabelEncoder()
            encoder.fit(pd.concat([y_train_val, self.y_test]))

            y_train_val_enc = encoder.transform(y_train_val)
            y_test_enc = encoder.transform(self.y_test)

            pipeline.fit(X_train_val, y_train_val_enc)
            y_pred_enc = pipeline.predict(self.X_test)

            cm = confusion_matrix(y_test_enc, y_pred_enc)
            accuracy = accuracy_score(y_test_enc, y_pred_enc)
            f1 = f1_score(y_test_enc, y_pred_enc, average="weighted")

            result = {
                "confusion_matrix": cm,
                "labels": list(encoder.classes_),
                "accuracy": accuracy,
                "f1": f1,
            }
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the GUI
            self.failed.emit(f"Model evaluation failed: {exc}")
            return

        self.succeeded.emit(result)
