"""MFCC feature extraction for a reconstructed mic dataset session.

This is the reference spec for these parameters - the same way
wolfSentry/LoRa.md is a cross-repo wire contract, a later on-chip inference
port needs one clear source of truth to match, not "whatever the Python
happened to do":

  - sample rate: 8000 Hz (matches the device's capture rate exactly, no
    resampling)
  - n_mfcc: 13 coefficients
  - n_fft: 256 samples (32ms at 8kHz)
  - hop_length: 80 samples (10ms at 8kHz)
  - pooling: mean + std across all frames in the session -> 26 fixed-length
    values per session (needed since sklearn/lazypredict wants one
    fixed-length row per example, not a variable-length frame sequence)

Also computes RMS/zero-crossing-rate/peak the same way the firmware does
(see src/mic_dataset.c), as a cross-check that the two pipelines agree -
not meant to replace the on-device values already decoded by
mic_session_reader.py.
"""

from __future__ import annotations

import librosa
import numpy as np

SAMPLE_RATE_HZ = 8000
N_MFCC = 13
N_FFT = 256
HOP_LENGTH = 80


def extract_mfcc(audio_int16: np.ndarray) -> dict[str, float]:
    """audio_int16: 1-D int16 array (native capture format, no normalization
    applied by the caller). Returns a flat dict of mfcc{i}_mean/mfcc{i}_std."""
    y = audio_int16.astype(np.float32) / 32768.0
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE_HZ, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)

    features: dict[str, float] = {}
    for i in range(N_MFCC):
        features[f"mfcc{i}_mean"] = float(mean[i])
        features[f"mfcc{i}_std"] = float(std[i])
    return features


def extract_pc_simple_features(audio_int16: np.ndarray) -> dict[str, float]:
    """RMS/zero-crossing-rate/peak computed the same way the firmware does,
    prefixed pc_ to distinguish from the on-device-computed columns decoded
    by mic_session_reader.py - a cross-check, not a replacement."""
    if len(audio_int16) == 0:
        return {"pc_rms": 0.0, "pc_zcr": 0, "pc_peak": 0}

    x = audio_int16.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    signs = np.signbit(audio_int16)
    zcr = int(np.count_nonzero(signs[1:] != signs[:-1]))
    peak = int(np.max(np.abs(audio_int16)))

    return {"pc_rms": rms, "pc_zcr": zcr, "pc_peak": peak}
