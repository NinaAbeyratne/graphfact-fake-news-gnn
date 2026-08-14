"""
Full preprocessing pipeline

Run directly to process train / valid / test splits:

    python src/preprocessing/prepare_dataset.py

The script expects the following raw TSV files in data/raw/:
    train.tsv
    valid.tsv
    test.tsv

Outputs are written to data/processed/:
    processed_train.csv
    processed_valid.csv
    processed_test.csv

Encoders are FIT on the training set only, then used to TRANSFORM
all three splits.  This prevents data leakage from validation / test sets
into the preprocessing step.
"""

from __future__ import annotations
import pathlib
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from clean_text import apply_text_cleaning, build_article_column

# Paths

# Resolve paths relative to this file so the script works from any cwd.
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent          # repo root
RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"

# Column names for the raw TSV files (no header row in the LIAR dataset).
COLUMN_NAMES = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker-title",
    "state",
    "party-affiliation",
    "barely-true",
    "false",
    "half-true",
    "mostly-true",
    "pof",
    "context",
]

# Categorical columns whose NaN values are imputed with the column mode.
CAT_COLS = [
    "subject",
    "speaker",
    "speaker-title",
    "state",
    "party-affiliation",
    "context",
]

# Numeric credit-history columns: NaN → 0, then cast to int.
NUM_COLS = ["barely-true", "false", "half-true", "mostly-true", "pof"]

# Categorical columns to be label-encoded (fit on train, transform all).
ENCODE_COLS = [
    "label",
    "subject",
    "speaker",
    "speaker-title",
    "state",
    "party-affiliation",
]


# Load a raw TSV split as a data frame
def load_raw(path: str | pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMN_NAMES)
    df = df.drop(columns=["id"])
    return df

# Download a single LIAR-dataset split from Kaggle using kagglehub
def load_from_kaggle(split_name: str) -> pd.DataFrame:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for Kaggle loading.\n"
            "Install it with:  pip install kagglehub[pandas-datasets]"
        ) from exc

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "doanquanvietnamca/liar-dataset",
        f"{split_name}.tsv",
        pandas_kwargs={"sep": "\t", "header": None, "names": COLUMN_NAMES},
    )
    df = df.drop(columns=["id"])
    return df

# Download all requested LIAR-dataset splits from Kaggle
def load_all_from_kaggle(
    split_names: tuple[str, ...] = ("train", "valid", "test"),
) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        print(f"Downloading '{split_name}.tsv' from Kaggle ...")
        dfs[split_name] = load_from_kaggle(split_name)
        print(f"  Loaded {len(dfs[split_name]):,} rows.")
    return dfs


# Structural cleaning (NaN imputation + numeric casting)

# Fill NaN in values in column with mode
def _impute_with_mode(df: pd.DataFrame, col: str, mode_value) -> pd.DataFrame:
    df[col] = df[col].fillna(mode_value)
    return df


# Impute missing values and fix dtypes
def structural_clean(
    df: pd.DataFrame,
    cat_modes: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:

    df = df.copy()

    # Categorical columns: fill with mode
    if cat_modes is None:
        cat_modes = {col: df[col].mode()[0] for col in CAT_COLS}

    for col in CAT_COLS:
        df[col] = df[col].fillna(cat_modes[col])

    # Numeric credit-history columns: fill with 0, cast to int
    df[NUM_COLS] = df[NUM_COLS].fillna(0).astype(int)

    return df, cat_modes

# Label encoding (fit on train, transform all splits)
def fit_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    for col in ENCODE_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


# Transform data frames using pre-fitted encoders
def apply_encoders(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        col_str = df[col].astype(str)
        # Map unseen labels to the first class in the encoder (safe fallback).
        known_classes = set(le.classes_)
        df[col] = col_str.apply(
            lambda v: v if v in known_classes else le.classes_[0]
        )
        df[col] = le.transform(df[col])
    return df


# Full pipeline

# Execute the end-to-end preprocessing pipeline for all data splits
def run_pipeline(
    splits: dict[str, "pathlib.Path | pd.DataFrame"],
    output_dir: pathlib.Path = PROCESSED_DIR,
    save_encoders: bool = True,
) -> None:
    
    if "train" not in splits:
        raise ValueError("splits dict must contain a 'train' key.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all splits (Path → DataFrame, or pass through if already DF)
    print("Loading dataset splits ...")
    raw: dict[str, pd.DataFrame] = {}
    for name, source in splits.items():
        if isinstance(source, pd.DataFrame):
            raw[name] = source
        else:
            raw[name] = load_raw(source)

    # Structural cleaning (fit on train)
    print("Structural cleaning ...")
    train_clean, cat_modes = structural_clean(raw["train"])
    cleaned: dict[str, pd.DataFrame] = {"train": train_clean}
    for name, df in raw.items():
        if name != "train":
            cleaned[name], _ = structural_clean(df, cat_modes=cat_modes)

    # Label encoding (fit on train)
    print("Fitting label encoders on train ...")
    encoders = fit_encoders(cleaned["train"])

    encoded: dict[str, pd.DataFrame] = {}
    for name, df in cleaned.items():
        print(f"  Encoding '{name}' split ...")
        encoded[name] = apply_encoders(df, encoders)

    # Text cleaning + article column
    processed: dict[str, pd.DataFrame] = {}
    for name, df in encoded.items():
        print(f"  Text-cleaning '{name}' split ...")
        df = apply_text_cleaning(df)
        df = build_article_column(df)
        processed[name] = df

    # Save results
    for name, df in processed.items():
        out_path = output_dir / f"processed_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}  ({len(df):,} rows)")

    if save_encoders:
        artifacts = {"encoders": encoders, "cat_modes": cat_modes}
        pkl_path = output_dir / "preprocessing_artifacts.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(artifacts, fh)
        print(f"  Saved encoders → {pkl_path}")

    print("Data preparation completed.")


# Entry point
if __name__ == "__main__":
    
    # Load directly from Kaggle via kagglehub (preferred)
    splits: dict[str, "pathlib.Path | pd.DataFrame"] = {}

    try:
        print("Attempting to load dataset from Kaggle ...")
        splits = load_all_from_kaggle()          # type: ignore[assignment]
        print("Kaggle download complete.")

    except Exception as kaggle_err:  # noqa: BLE001
        print(f"[WARNING] Kaggle loading failed: {kaggle_err}")
        print("Falling back to local TSV files in data/raw/ ...")

        # Fall back to local TSV files already in data/raw/
        for split_name in ("train", "valid", "test"):
            tsv_path = RAW_DIR / f"{split_name}.tsv"
            if tsv_path.exists():
                splits[split_name] = tsv_path
            else:
                print(
                    f"[WARNING] {tsv_path} not found – "
                    f"skipping '{split_name}' split."
                )

        if not splits:
            raise FileNotFoundError(
                f"No TSV files found in {RAW_DIR} and Kaggle download failed. "
                "Place train.tsv / valid.tsv / test.tsv in data/raw/ or ensure "
                "kagglehub is configured with valid credentials."
            ) from kaggle_err

        if "train" not in splits:
            raise FileNotFoundError(
                "train.tsv is required to fit the encoders but was not found "
                f"in {RAW_DIR} and Kaggle download failed."
            ) from kaggle_err

    run_pipeline(splits)