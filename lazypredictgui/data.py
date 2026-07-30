"""Generic CSV loading + label-column resolution + train/val/test splitting.

Used by both import paths (the plain "Load CSV..." picker, and the
wolfSentry session importer's pooled DataFrame from dataset.py) - this
module doesn't know or care which one produced the DataFrame.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_label_column(df: pd.DataFrame, override: str | None = None) -> str:
    """Picks the label column: an explicit override if given, else a column
    literally named 'label' if present, else the last column."""
    if override is not None:
        if override not in df.columns:
            raise ValueError(f"column {override!r} not found in CSV")
        return override
    if "label" in df.columns:
        return "label"
    return df.columns[-1]


def three_way_split(
    df: pd.DataFrame,
    label_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits df into (train, val, test) DataFrames, stratified by label_col
    where possible (falls back to a plain split if any class has too few
    examples to stratify). Fractions must sum to ~1.0 (checked by the GUI's
    own live validation, not re-checked here)."""
    y = df[label_col]

    def _safe_stratify(frame: pd.DataFrame, labels: pd.Series):
        counts = labels.value_counts()
        return labels if counts.min() >= 2 else None

    train_df, rest_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=random_state,
        stratify=_safe_stratify(df, y),
    )

    # val_frac/test_frac are fractions of the *original* whole, so within
    # `rest_df` the val share is val_frac/(val_frac+test_frac).
    if len(rest_df) < 2:
        # Can't split 0-or-1 rows into two non-empty parts (sklearn raises
        # ValueError: "the resulting train set will be empty" here) - this
        # happens with genuinely tiny datasets (a smoke test with a
        # handful of sessions), not just malformed input. Rather than
        # crash, hand the whole (possibly empty) remainder to whichever of
        # val/test has the larger requested share - LazyPredict needs a
        # non-empty val set to produce a leaderboard at all, test can
        # tolerate being empty (it only matters once you get to the
        # Confusion Matrix tab).
        if val_frac >= test_frac:
            val_df, test_df = rest_df, rest_df.iloc[0:0]
        else:
            val_df, test_df = rest_df.iloc[0:0], rest_df
        return train_df, val_df, test_df

    rest_y = rest_df[label_col]
    val_share_of_rest = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest_df,
        train_size=val_share_of_rest,
        random_state=random_state,
        stratify=_safe_stratify(rest_df, rest_y),
    )

    return train_df, val_df, test_df
