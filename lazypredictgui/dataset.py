"""In-memory model for the audio import/label/combine pipeline.

Each imported wolfSentry session file becomes one named entry here. Multiple
entries sharing the same label get pooled into one combined table before any
train/val/test split happens (the explicit requirement: several dog
sessions combined together, not split individually).

Two example granularities, chosen per session at import time:
- **Whole session** (default): 1 session = 1 example, using the on-device
  Goertzel/RMS/ZCR/peak summary plus MFCC computed over the full ~30s.
  Needs many separate session recordings per class before a split means
  anything (see README's volume guidance).
- **Per-second sub-clips**: 1 session = up to 30 examples, one per 1-second
  audio chunk already stored in the session. Gets useful data out of just a
  couple of recordings, but on-device Goertzel is only computed at the
  whole-session level (not per second), so these rows only carry MFCC +
  PC-computed RMS/ZCR/peak, not goertzel_* columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .mic_features import extract_mfcc, extract_pc_simple_features
from .mic_session_reader import BAND_HZ, MicSession

DISCARD_LABEL = "__discard__"
SUB_CLIP_SAMPLES = 8000  # 1 second at 8kHz - matches the device's own per-tick chunk size


@dataclass
class ImportedSession:
    name: str
    session: MicSession
    label: str | None = None  # None = not yet labeled; DISCARD_LABEL = excluded
    sub_clip_mode: bool = False  # False = 1 row for the whole session; True = 1 row per second
    _rows_cache: list[dict[str, float]] | None = field(default=None, repr=False)

    def example_count(self) -> int:
        """How many example rows this session will contribute - for the
        per-class running counts, since sub-clip mode multiplies this."""
        if self.sub_clip_mode:
            return len(self.session.audio) // SUB_CLIP_SAMPLES
        return 1

    def rows(self) -> list[dict[str, float]]:
        """Computed once per session and cached - MFCC extraction isn't
        free, don't redo it every time to_dataframe() is called."""
        if self._rows_cache is not None:
            return self._rows_cache

        self._rows_cache = self._sub_clip_rows() if self.sub_clip_mode else [self._whole_session_row()]
        return self._rows_cache

    def _whole_session_row(self) -> dict[str, float]:
        row: dict[str, float] = {}
        summary = self.session.summary
        if summary is not None:
            row["rms"] = summary.rms
            row["zcr"] = summary.zcr
            row["peak"] = summary.peak
            for hz, energy in zip(BAND_HZ, summary.band_energy):
                row[f"goertzel_{hz}hz"] = energy
        # else: aborted/incomplete session, no on-device summary - leave
        # those columns missing (NaN once assembled into a DataFrame), the
        # split/LazyPredict pipeline's SimpleImputer handles that.

        if len(self.session.audio) > 0:
            row.update(extract_mfcc(self.session.audio))
            row.update(extract_pc_simple_features(self.session.audio))
        return row

    def _sub_clip_rows(self) -> list[dict[str, float]]:
        audio = self.session.audio
        n_chunks = len(audio) // SUB_CLIP_SAMPLES
        rows = []
        for i in range(n_chunks):
            chunk = audio[i * SUB_CLIP_SAMPLES : (i + 1) * SUB_CLIP_SAMPLES]
            row = dict(extract_mfcc(chunk))
            row.update(extract_pc_simple_features(chunk))
            rows.append(row)
        return rows


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ImportedSession] = {}

    def add(self, name: str, session: MicSession, sub_clip_mode: bool = False) -> None:
        self._sessions[name] = ImportedSession(name=name, session=session,
                                                 sub_clip_mode=sub_clip_mode)

    def set_label(self, name: str, label: str | None) -> None:
        self._sessions[name].label = label

    def remove(self, name: str) -> None:
        self._sessions.pop(name, None)

    def names(self) -> list[str]:
        return list(self._sessions.keys())

    def get(self, name: str) -> ImportedSession:
        return self._sessions[name]

    def labeled_counts(self) -> dict[str, int]:
        """Per-class count of actual example rows (not sessions) - a
        sub-clip-mode session contributes ~30, a whole-session-mode one
        contributes 1, so this reflects what the split will actually see."""
        counts: dict[str, int] = {}
        for entry in self._sessions.values():
            if entry.label and entry.label != DISCARD_LABEL:
                counts[entry.label] = counts.get(entry.label, 0) + entry.example_count()
        return counts

    def to_dataframe(self) -> pd.DataFrame:
        """Pools every labeled (non-discarded) session into one combined
        table, `label` column last. A session in sub-clip mode contributes
        multiple rows (all sharing that session's label); mixing sub-clip
        and whole-session rows for the same label is fine - missing columns
        (e.g. sub-clip rows have no goertzel_* columns) just come out NaN,
        same as an incomplete/aborted whole-session import already does."""
        rows = []
        for entry in self._sessions.values():
            if not entry.label or entry.label == DISCARD_LABEL:
                continue
            for feature_row in entry.rows():
                row = dict(feature_row)
                row["source"] = entry.name
                row["label"] = entry.label
                rows.append(row)

        return pd.DataFrame(rows)
