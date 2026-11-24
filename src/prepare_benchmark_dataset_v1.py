"""
Prepare reproducible benchmark inputs from the validated Synthea cohort.

Run from the project root:
    python src/prepare_benchmark_dataset_v1.py

Expected input:
    output/processed/patient_dataset_24m_primary.csv  (919 rows)
    output/processed/patient_dataset_24m_all.csv      (1142 rows)

Outputs are written to output/benchmark_v1/.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES_F1 = [
    "age", "systolic_bp", "diastolic_bp",
    "heart_rate", "respiratory_rate", "bmi"
]
FEATURES_F2 = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate", "bmi"
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def find_id_column(df):
    for c in ["PATIENT", "Id", "id", "patient_id", "Patient"]:
        if c in df.columns:
            return c
    raise ValueError(f"No patient ID column found. Columns: {list(df.columns)}")

def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "output" / "processed"
    out = root / "output" / "benchmark_v1"
    out.mkdir(parents=True, exist_ok=True)

    primary_path = processed / "patient_dataset_24m_primary.csv"
    all_path = processed / "patient_dataset_24m_all.csv"
    for p in [primary_path, all_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    primary = pd.read_csv(primary_path)
    all_df = pd.read_csv(all_path)
    id_col = find_id_column(primary)

    if len(primary) != 919:
        raise ValueError(f"Expected 919 primary patients, got {len(primary)}")
    if primary[id_col].astype(str).duplicated().any():
        raise ValueError("Duplicate patient IDs in primary cohort")

    missing = [f for f in FEATURES_F1 if f not in primary.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    for f in FEATURES_F1:
        primary[f] = pd.to_numeric(primary[f], errors="coerce")
    if primary[FEATURES_F1].isna().any().any():
        raise ValueError("Primary feature matrix contains missing/non-numeric values")
    if not np.isfinite(primary[FEATURES_F1].to_numpy(dtype=float)).all():
        raise ValueError("Primary feature matrix contains infinite values")

    # Keep IDs alongside the raw feature matrix for traceability.
    primary[[id_col] + FEATURES_F1].to_csv(out / "patient_dataset_v1.csv", index=False)

    age = pd.to_numeric(primary["age"], errors="coerce")
    adult = primary.loc[age >= 18].copy()
    if len(adult) == 0:
        raise ValueError("Adult cohort is empty")
    adult[[id_col] + FEATURES_F1].to_csv(out / "adult_dataset_v1.csv", index=False)

    feature_files = {}
    scaler_files = {}
    cohort_files = {
        "all_919": primary,
        "adult_only": adult,
    }
    configurations = {"F1": FEATURES_F1, "F2": FEATURES_F2}

    for cohort_name, cohort in cohort_files.items():
        for config_name, features in configurations.items():
            X = cohort[features].astype(float)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            stem = f"{cohort_name}_{config_name}"
            feature_path = out / f"features_{stem}.csv"
            scaler_path = out / f"feature_scaler_{stem}.joblib"

            scaled_df = pd.DataFrame(X_scaled, columns=features)
            scaled_df.insert(0, id_col, cohort[id_col].astype(str).to_numpy())
            scaled_df.to_csv(feature_path, index=False)
            joblib.dump(scaler, scaler_path)

            feature_files[stem] = {
                "file": feature_path.name,
                "rows": int(len(cohort)),
                "features": features,
            }
            scaler_files[stem] = scaler_path.name

    manifest = {
        "benchmark": "Synthea Patient Phenotyping Benchmark V1",
        "source_dataset": str(primary_path.relative_to(root)),
        "source_sha256": sha256(primary_path),
        "source_rows": int(len(primary)),
        "source_id_column": id_col,
        "index_date": str(primary["index_date"].iloc[0]) if "index_date" in primary else None,
        "window_months": 24,
        "cohorts": {
            "all_919": int(len(primary)),
            "adult_only": int(len(adult)),
            "under_18_in_primary": int((age < 18).sum()),
        },
        "feature_configurations": {
            "F1": FEATURES_F1,
            "F2": FEATURES_F2,
        },
        "scaled_feature_files": feature_files,
        "scaler_files": scaler_files,
        "scaling": "StandardScaler fitted separately within each cohort/configuration",
        "excluded_columns": [
            id_col, "gender", "GENDER", "index_date", "window_months",
            "missing_*", "complete_primary", "complete_extended"
        ],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "note": (
            "Patient IDs are retained only for traceability and excluded from "
            "feature matrices. F1/F2 and adult/all cohorts are separate "
            "sensitivity configurations."
        ),
    }
    with open(out / "benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 68)
    print("BENCHMARK DATA PREPARATION COMPLETE")
    print("=" * 68)
    print(f"Primary cohort: {len(primary)}")
    print(f"Adult cohort:  {len(adult)}")
    print(f"Output:        {out}")
    print("Configurations: all_919_F1, all_919_F2, adult_only_F1, adult_only_F2")

if __name__ == "__main__":
    main()
