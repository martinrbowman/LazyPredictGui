"""Generates synthetic per-class feature CSVs for testing LazyPredictGUI's
pipeline without needing real hardware captures yet - background/dog/speech,
matching the exact column shape lazypredictgui/dataset.py's
ImportedSession.features() produces (goertzel_*hz, rms/zcr/peak,
mfcc{i}_mean/std, pc_rms/zcr/peak), plus a `label` column already set per
file so they can be loaded individually or concatenated.

Not physically accurate audio features - just distinct-enough per-class
distributions to give LazyPredict something real to differentiate, so the
leaderboard/confusion-matrix pipeline can be exercised end to end.

Run: python generate_synthetic_data.py
Writes quiet.csv, dog.csv, speech.csv into this same directory.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

BAND_HZ = [300, 400, 500, 650, 850, 1100, 1400, 1800, 2300, 2900, 3600, 4000]
N_MFCC = 13
N_ROWS_PER_CLASS = 60
RNG_SEED = 42


def _goertzel_profile(band_means: list[float], band_std_frac: float, rng: np.random.Generator,
                       n: int) -> dict[str, np.ndarray]:
    cols = {}
    for hz, mean in zip(BAND_HZ, band_means):
        cols[f"goertzel_{hz}hz"] = rng.normal(mean, mean * band_std_frac, n).clip(min=0)
    return cols


def _mfcc_profile(mean_pattern: list[float], rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = {}
    for i in range(N_MFCC):
        base = mean_pattern[i % len(mean_pattern)]
        cols[f"mfcc{i}_mean"] = rng.normal(base, abs(base) * 0.2 + 0.5, n)
        cols[f"mfcc{i}_std"] = rng.normal(abs(base) * 0.3 + 1.0, 0.5, n).clip(min=0)
    return cols


def _make_class(label: str, rms_mean: float, rms_std: float, peak_mean: float, zcr_mean: float,
                 zcr_std: float, band_means: list[float], mfcc_pattern: list[float],
                 rng: np.random.Generator) -> pd.DataFrame:
    n = N_ROWS_PER_CLASS
    data: dict[str, np.ndarray] = {}

    data["rms"] = rng.normal(rms_mean, rms_std, n).clip(min=1)
    data["zcr"] = rng.normal(zcr_mean, zcr_std, n).clip(min=0).astype(int)
    data["peak"] = rng.normal(peak_mean, peak_mean * 0.15, n).clip(min=1, max=32767).astype(int)

    data.update(_goertzel_profile(band_means, band_std_frac=0.25, rng=rng, n=n))
    data.update(_mfcc_profile(mfcc_pattern, rng=rng, n=n))

    # PC-computed cross-check columns - same rough scale as the on-device
    # ones with a little independent noise, matching how they're a
    # cross-check on real data rather than identical duplicates.
    data["pc_rms"] = data["rms"] * rng.normal(1.0, 0.05, n)
    data["pc_zcr"] = (data["zcr"] * rng.normal(1.0, 0.05, n)).astype(int)
    data["pc_peak"] = (data["peak"] * rng.normal(1.0, 0.05, n)).astype(int)

    df = pd.DataFrame(data)
    df["label"] = label
    return df


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    out_dir = pathlib.Path(__file__).parent

    # quiet/background: low energy overall, roughly flat spectrum, low ZCR
    quiet = _make_class(
        "quiet", rms_mean=80, rms_std=30, peak_mean=400, zcr_mean=300, zcr_std=80,
        band_means=[500] * len(BAND_HZ),
        mfcc_pattern=[-5, -2, 0, 1, -1, 0, 2, -1, 0, 1, -2, 0, 1],
        rng=rng,
    )

    # dog: high energy, concentrated in low-mid bands (bark fundamental +
    # harmonics), moderate-high ZCR (broadband noise component in barks)
    dog = _make_class(
        "dog", rms_mean=3200, rms_std=900, peak_mean=18000, zcr_mean=1800, zcr_std=400,
        band_means=[9000, 12000, 10000, 7000, 5000, 3000, 1500, 1000, 800, 700, 600, 500],
        mfcc_pattern=[10, 6, -4, 3, -6, 2, -3, 1, -2, 4, -1, 2, -1],
        rng=rng,
    )

    # speech: moderate energy, concentrated in mid bands (formants F1/F2/F3),
    # high ZCR (fricatives/sibilants)
    speech = _make_class(
        "speech", rms_mean=1500, rms_std=500, peak_mean=9000, zcr_mean=2600, zcr_std=500,
        band_means=[2000, 2500, 3000, 4500, 6000, 7000, 6500, 5000, 3500, 2000, 1200, 900],
        mfcc_pattern=[-2, 8, 5, -5, 3, -4, 2, -3, 4, -2, 3, -1, 2],
        rng=rng,
    )

    for name, df in (("quiet", quiet), ("dog", dog), ("speech", speech)):
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
