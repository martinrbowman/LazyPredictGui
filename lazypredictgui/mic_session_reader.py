"""Decodes a wolfSentry mic dataset session file.

Mirrors /home/martin/development/wolfSentry/src/mic_dataset.h's wire format
exactly - keep the two in sync by hand (no shared build between the repos).
A session file is 30 AUDIO records (one per second) followed by exactly one
SESSION_SUMMARY record, each starting with a record_type byte. Reads
sequentially and stops cleanly on truncated trailing data rather than
raising, same defensive pattern as wolfSentry's own host-side decoders
(record.py::decode_records).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

REC_AUDIO = 0
REC_SESSION_SUMMARY = 1

# Must match MIC_DATASET_BAND_HZ in src/mic_dataset.h exactly - only used
# here for column naming (e.g. "goertzel_300hz"), not for any computation.
BAND_HZ = [300, 400, 500, 650, 850, 1100, 1400, 1800, 2300, 2900, 3600, 4000]

_AUDIO_HEADER_FMT = "<BBIH"  # record_type, reserved, uptime_ms, sample_count
_AUDIO_HEADER_SIZE = struct.calcsize(_AUDIO_HEADER_FMT)
assert _AUDIO_HEADER_SIZE == 8

_SUMMARY_FMT = "<BBIHIfIhH" + "f" * len(BAND_HZ)
_SUMMARY_SIZE = struct.calcsize(_SUMMARY_FMT)
assert _SUMMARY_SIZE == 72


@dataclass
class MicSessionSummary:
    uptime_ms: int
    total_samples: int
    rms: float
    zcr: int
    peak: int
    band_energy: list[float]  # same order as BAND_HZ


@dataclass
class MicSession:
    audio: np.ndarray  # int16, concatenated in capture order
    summary: MicSessionSummary | None  # None if the session file is
    # truncated/aborted (mic_dataset_stop was used, or the transfer was cut
    # short) - the reconstructed audio is still usable for playback/labeling
    # and PC-side MFCC, just missing the on-device Goertzel/RMS/ZCR/peak
    # values.

    @property
    def duration_s(self) -> float:
        return len(self.audio) / 8000.0


def decode_session(data: bytes) -> MicSession:
    chunks: list[np.ndarray] = []
    summary: MicSessionSummary | None = None
    offset = 0

    while offset < len(data):
        if offset + 1 > len(data):
            break
        record_type = data[offset]

        if record_type == REC_AUDIO:
            if offset + _AUDIO_HEADER_SIZE > len(data):
                break
            _rtype, _reserved, _uptime_ms, sample_count = struct.unpack_from(
                _AUDIO_HEADER_FMT, data, offset
            )
            body_start = offset + _AUDIO_HEADER_SIZE
            body_end = body_start + sample_count * 2
            if body_end > len(data):
                break
            samples = np.frombuffer(data, dtype="<i2", count=sample_count, offset=body_start)
            chunks.append(samples)
            offset = body_end

        elif record_type == REC_SESSION_SUMMARY:
            if offset + _SUMMARY_SIZE > len(data):
                break
            fields = struct.unpack_from(_SUMMARY_FMT, data, offset)
            (
                _rtype,
                _reserved,
                uptime_ms,
                _reserved2,
                total_samples,
                rms,
                zcr,
                peak,
                _reserved3,
                *band_energy,
            ) = fields
            summary = MicSessionSummary(
                uptime_ms=uptime_ms,
                total_samples=total_samples,
                rms=rms,
                zcr=zcr,
                peak=peak,
                band_energy=list(band_energy),
            )
            offset += _SUMMARY_SIZE

        else:
            # Unrecognized record type - stop rather than misinterpret
            # subsequent bytes (same "truncated data" defensive stance).
            break

    audio = np.concatenate(chunks) if chunks else np.array([], dtype="<i2")
    return MicSession(audio=audio, summary=summary)
