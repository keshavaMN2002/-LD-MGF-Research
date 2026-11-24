
# from pathlib import Path
# import json
# import warnings

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # ============================================================
# # CONFIGURATION
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# RAW_DIR = PROJECT_ROOT / "output" / "csv"
# PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
# REPORT_DIR = PROCESSED_DIR / "validation"

# DATASET_FILE = PROCESSED_DIR / "patient_dataset_24m_primary.csv"
# ALL_DATASET_FILE = PROCESSED_DIR / "patient_dataset_24m_all.csv"

# PATIENTS_FILE = RAW_DIR / "patients.csv"
# OBSERVATIONS_FILE = RAW_DIR / "observations.csv"

# EXPECTED_PATIENTS = 919
# WINDOW_MONTHS = 24

# FEATURES = [
#     "age",
#     "systolic_bp",
#     "diastolic_bp",
#     "heart_rate",
#     "respiratory_rate",
#     "bmi",
# ]

# NUMERIC_FEATURES = [
#     "age",
#     "systolic_bp",
#     "diastolic_bp",
#     "heart_rate",
#     "respiratory_rate",
#     "bmi",
# ]

# OBSERVATION_MAP = {
#     "Systolic Blood Pressure": "systolic_bp",
#     "Diastolic Blood Pressure": "diastolic_bp",
#     "Heart rate": "heart_rate",
#     "Respiratory rate": "respiratory_rate",
#     "Body mass index (BMI) [Ratio]": "bmi",
#     "Body Height": "body_height",
#     "Body Weight": "body_weight",
# }

# # Broad screening bounds for detecting possible data issues.
# # These are NOT clinical diagnostic thresholds and do not
# # automatically remove observations or patients.
# SCREENING_BOUNDS = {
#     "age": (0, 120),
#     "systolic_bp": (40, 300),
#     "diastolic_bp": (20, 200),
#     "heart_rate": (20, 250),
#     "respiratory_rate": (4, 80),
#     "bmi": (8, 100),
# }

# # If more than one non-empty unit occurs for a feature,
# # the feature is flagged for unit review.
# # No automatic unit conversion is performed.


# # ============================================================
# # HELPERS
# # ============================================================

# def find_patient_id_column(df):
#     candidates = ["PATIENT", "patient_id", "Id", "id", "ID"]

#     for col in candidates:
#         if col in df.columns:
#             return col

#     raise ValueError(
#         f"Could not identify patient ID column. "
#         f"Available columns: {list(df.columns)}"
#     )


# def require_columns(df, columns, filename):
#     missing = [c for c in columns if c not in df.columns]

#     if missing:
#         raise ValueError(
#             f"{filename} is missing required columns: {missing}"
#         )


# def parse_dates(series):
#     return (
#         pd.to_datetime(series, errors="coerce", utc=True)
#         .dt.tz_localize(None)
#     )


# def save_csv(df, filename):
#     path = REPORT_DIR / filename
#     df.to_csv(path, index=False)
#     print(f"Saved: {path}")


# def add_check(checks, name, passed, details, severity="FAIL"):
#     checks.append({
#         "check": name,
#         "status": "PASS" if passed else severity,
#         "details": str(details),
#     })


# # ============================================================
# # LOAD DATA
# # ============================================================

# def load_inputs():
#     required_files = [
#         DATASET_FILE,
#         ALL_DATASET_FILE,
#         PATIENTS_FILE,
#         OBSERVATIONS_FILE,
#     ]

#     for path in required_files:
#         if not path.exists():
#             raise FileNotFoundError(f"Required file not found: {path}")

#     dataset = pd.read_csv(DATASET_FILE)
#     all_dataset = pd.read_csv(ALL_DATASET_FILE)

#     patients = pd.read_csv(
#         PATIENTS_FILE,
#         usecols=["Id", "BIRTHDATE", "GENDER"],
#     )

#     observations = pd.read_csv(
#         OBSERVATIONS_FILE,
#         usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE", "UNITS"],
#         low_memory=False,
#     )

#     require_columns(dataset, FEATURES, DATASET_FILE.name)
#     require_columns(all_dataset, FEATURES, ALL_DATASET_FILE.name)

#     return dataset, all_dataset, patients, observations


# # ============================================================
# # IDENTIFY PATIENT ID AND INDEX DATE
# # ============================================================

# def identify_index_date(dataset, observations):
#     if "index_date" in dataset.columns:
#         dates = parse_dates(dataset["index_date"]).dropna()

#         if not dates.empty:
#             return dates.max()

#     dates = parse_dates(observations["DATE"]).dropna()

#     if dates.empty:
#         raise ValueError("No valid observation dates found.")

#     return dates.max()


# # ============================================================
# # DATA INTEGRITY
# # ============================================================

# def validate_integrity(dataset, all_dataset, patients):
#     checks = []

#     patient_col = find_patient_id_column(dataset)
#     all_patient_col = find_patient_id_column(all_dataset)

#     add_check(
#         checks,
#         "Expected primary cohort size",
#         len(dataset) == EXPECTED_PATIENTS,
#         f"Observed {len(dataset)}; expected {EXPECTED_PATIENTS}.",
#     )

#     add_check(
#         checks,
#         "Unique patient IDs in primary cohort",
#         dataset[patient_col].is_unique,
#         f"Duplicate IDs: {dataset[patient_col].duplicated().sum()}.",
#     )

#     add_check(
#         checks,
#         "Primary cohort IDs exist in Synthea patients",
#         set(dataset[patient_col].astype(str)).issubset(
#             set(patients["Id"].astype(str))
#         ),
#         "Every primary cohort patient ID should exist in patients.csv.",
#     )

#     add_check(
#         checks,
#         "All-patient dataset size",
#         len(all_dataset) == len(patients),
#         f"All dataset rows={len(all_dataset)}, "
#         f"patients.csv rows={len(patients)}.",
#     )

#     add_check(
#         checks,
#         "All-patient dataset IDs are unique",
#         all_dataset[all_patient_col].is_unique,
#         f"Duplicate IDs: "
#         f"{all_dataset[all_patient_col].duplicated().sum()}.",
#     )

#     return checks, patient_col


# # ============================================================
# # MISSINGNESS AND NUMERIC VALIDATION
# # ============================================================

# def validate_features(dataset):
#     checks = []
#     rows = []

#     for feature in FEATURES:
#         values = pd.to_numeric(dataset[feature], errors="coerce")

#         missing = int(values.isna().sum())
#         non_numeric = int(
#             dataset[feature].notna().sum() - values.notna().sum()
#         )
#         infinite = int(np.isinf(values.dropna()).sum())

#         rows.append({
#             "feature": feature,
#             "rows": len(dataset),
#             "missing": missing,
#             "missing_percent": round(100 * missing / len(dataset), 2),
#             "non_numeric": non_numeric,
#             "infinite": infinite,
#         })

#         add_check(
#             checks,
#             f"{feature}: no missing values",
#             missing == 0,
#             f"Missing values: {missing}.",
#         )

#         add_check(
#             checks,
#             f"{feature}: numeric and finite",
#             non_numeric == 0 and infinite == 0,
#             f"Non-numeric={non_numeric}; infinite={infinite}.",
#         )

#     return checks, pd.DataFrame(rows)


# # ============================================================
# # DESCRIPTIVE STATISTICS AND OUTLIERS
# # ============================================================

# def calculate_statistics(dataset):
#     stats = []
#     outliers = []
#     plausibility = []

#     for feature in NUMERIC_FEATURES:
#         x = pd.to_numeric(dataset[feature], errors="coerce").dropna()

#         if x.empty:
#             continue

#         q1 = x.quantile(0.25)
#         median = x.median()
#         q3 = x.quantile(0.75)
#         iqr = q3 - q1

#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr

#         iqr_mask = (x < lower) | (x > upper)

#         stats.append({
#             "feature": feature,
#             "count": int(x.count()),
#             "mean": x.mean(),
#             "std": x.std(),
#             "min": x.min(),
#             "q1": q1,
#             "median": median,
#             "q3": q3,
#             "max": x.max(),
#             "iqr": iqr,
#             "iqr_lower_fence": lower,
#             "iqr_upper_fence": upper,
#         })

#         outliers.append({
#             "feature": feature,
#             "iqr_outlier_count": int(iqr_mask.sum()),
#             "iqr_outlier_percent": round(
#                 100 * iqr_mask.sum() / len(x), 2
#             ),
#             "lower_fence": lower,
#             "upper_fence": upper,
#         })

#         low, high = SCREENING_BOUNDS[feature]
#         outside = (x < low) | (x > high)

#         plausibility.append({
#             "feature": feature,
#             "screening_lower": low,
#             "screening_upper": high,
#             "outside_screening_bounds": int(outside.sum()),
#             "observed_min": x.min(),
#             "observed_max": x.max(),
#         })

#     return (
#         pd.DataFrame(stats),
#         pd.DataFrame(outliers),
#         pd.DataFrame(plausibility),
#     )


# # ============================================================
# # CATEGORICAL AND CORRELATION CHECKS
# # ============================================================

# def validate_gender(dataset):
#     gender_col = "gender" if "gender" in dataset.columns else None

#     if gender_col is None:
#         return pd.DataFrame(), [
#             {
#                 "check": "Gender column available",
#                 "status": "WARNING",
#                 "details": "No gender column found in primary dataset.",
#             }
#         ]

#     summary = (
#         dataset[gender_col]
#         .fillna("MISSING")
#         .astype(str)
#         .value_counts()
#         .rename_axis("gender")
#         .reset_index(name="patient_count")
#     )

#     checks = [{
#         "check": "Gender values inspected",
#         "status": "PASS",
#         "details": summary.to_dict(orient="records"),
#     }]

#     return summary, checks


# def calculate_correlations(dataset):
#     numeric = dataset[NUMERIC_FEATURES].apply(
#         pd.to_numeric, errors="coerce"
#     )

#     pearson = numeric.corr(method="pearson")
#     spearman = numeric.corr(method="spearman")

#     pearson.to_csv(REPORT_DIR / "pearson_correlation.csv")
#     spearman.to_csv(REPORT_DIR / "spearman_correlation.csv")

#     return pearson, spearman


# # ============================================================
# # UNIT AUDIT
# # ============================================================

# def audit_units(observations, index_date):
#     obs = observations.copy()

#     obs["DATE_PARSED"] = parse_dates(obs["DATE"])
#     obs["VALUE_NUMERIC"] = pd.to_numeric(
#         obs["VALUE"], errors="coerce"
#     )

#     start_date = index_date - pd.DateOffset(months=WINDOW_MONTHS)

#     obs = obs[
#         (obs["DATE_PARSED"] >= start_date)
#         & (obs["DATE_PARSED"] <= index_date)
#         & obs["DESCRIPTION"].isin(OBSERVATION_MAP.keys())
#         & obs["VALUE_NUMERIC"].notna()
#     ].copy()

#     obs["feature"] = obs["DESCRIPTION"].map(OBSERVATION_MAP)
#     obs["UNITS"] = obs["UNITS"].fillna("").astype(str).str.strip()

#     unit_summary = (
#         obs.groupby(["feature", "UNITS"], dropna=False)
#         .size()
#         .reset_index(name="observation_count")
#         .sort_values(["feature", "observation_count"], ascending=[True, False])
#     )

#     checks = []

#     for feature in OBSERVATION_MAP.values():
#         feature_rows = obs[obs["feature"] == feature]
#         nonempty_units = sorted(
#             u for u in feature_rows["UNITS"].unique() if u
#         )
#         blank_count = int((feature_rows["UNITS"] == "").sum())

#         passed = len(nonempty_units) <= 1

#         details = (
#             f"Units={nonempty_units}; blank-unit observations={blank_count}; "
#             f"observations={len(feature_rows)}."
#         )

#         severity = "WARNING" if not feature_rows.empty else "WARNING"

#         if len(nonempty_units) > 1:
#             status = "FAIL"
#             details += " Normalize units before clustering."
#         elif blank_count > 0 or len(nonempty_units) == 0:
#             status = "WARNING"
#             details += " Verify missing or unspecified units."
#         else:
#             status = "PASS"

#         checks.append({
#             "check": f"{feature}: unit consistency",
#             "status": status,
#             "details": details,
#         })

#     return unit_summary, checks, obs


# # ============================================================
# # TEMPORAL CONSISTENCY
# # ============================================================

# def audit_measurement_dates(obs, index_date, patient_col):
#     selected = obs.copy()

#     selected["PATIENT"] = selected["PATIENT"].astype(str)

#     latest = (
#         selected.sort_values("DATE_PARSED")
#         .groupby(["PATIENT", "feature"], as_index=False)
#         .tail(1)
#         [["PATIENT", "feature", "DATE_PARSED"]]
#     )

#     wide = latest.pivot(
#         index="PATIENT",
#         columns="feature",
#         values="DATE_PARSED",
#     ).reset_index()

#     date_features = [
#         "bmi",
#         "body_height",
#         "body_weight",
#     ]

#     rows = []

#     for feature in date_features:
#         if feature not in wide.columns:
#             continue

#         gap = (index_date - wide[feature]).dt.days

#         rows.append({
#             "feature": feature,
#             "patients_with_measurement": int(gap.notna().sum()),
#             "median_days_before_index": gap.median(),
#             "mean_days_before_index": gap.mean(),
#             "max_days_before_index": gap.max(),
#             "measurements_older_than_365_days": int((gap > 365).sum()),
#         })

#     # Pairwise date gaps for BMI, height, and weight.
#     pair_rows = []

#     for a, b in [
#         ("bmi", "body_height"),
#         ("bmi", "body_weight"),
#         ("body_height", "body_weight"),
#     ]:
#         if a in wide.columns and b in wide.columns:
#             days = (wide[a] - wide[b]).abs().dt.days

#             pair_rows.append({
#                 "feature_a": a,
#                 "feature_b": b,
#                 "patients_with_both": int(days.notna().sum()),
#                 "median_absolute_gap_days": days.median(),
#                 "max_absolute_gap_days": days.max(),
#                 "gap_over_180_days": int((days > 180).sum()),
#             })

#     return pd.DataFrame(rows), pd.DataFrame(pair_rows)


# # ============================================================
# # SPECIAL DISTRIBUTION CHECKS
# # ============================================================

# def distribution_checks(dataset):
#     checks = []

#     rr = pd.to_numeric(
#         dataset["respiratory_rate"], errors="coerce"
#     ).dropna()

#     if not rr.empty:
#         top_value = rr.value_counts(normalize=True).index[0]
#         top_share = rr.value_counts(normalize=True).iloc[0]

#         checks.append({
#             "check": "Respiratory-rate concentration",
#             "status": "WARNING" if top_share >= 0.50 else "PASS",
#             "details": (
#                 f"Most common value={top_value}; "
#                 f"share={top_share:.2%}. "
#                 "A highly concentrated feature may contribute little "
#                 "to distance-based clustering."
#             ),
#         })

#     age = pd.to_numeric(dataset["age"], errors="coerce")

#     checks.append({
#         "check": "Adult and pediatric population",
#         "status": "PASS",
#         "details": (
#             f"Adults (age >= 18): {(age >= 18).sum()}; "
#             f"under 18: {(age < 18).sum()}."
#         ),
#     })

#     return checks


# # ============================================================
# # PLOTS
# # ============================================================

# def create_plots(dataset, pearson):
#     for feature in NUMERIC_FEATURES:
#         x = pd.to_numeric(dataset[feature], errors="coerce").dropna()

#         if x.empty:
#             continue

#         plt.figure()
#         plt.hist(x, bins=30)
#         plt.title(f"Distribution of {feature}")
#         plt.xlabel(feature)
#         plt.ylabel("Patient count")
#         plt.tight_layout()
#         plt.savefig(REPORT_DIR / f"{feature}_histogram.png", dpi=150)
#         plt.close()

#         plt.figure()
#         plt.boxplot(x, vert=True)
#         plt.title(f"Boxplot of {feature}")
#         plt.ylabel(feature)
#         plt.tight_layout()
#         plt.savefig(REPORT_DIR / f"{feature}_boxplot.png", dpi=150)
#         plt.close()

#     plt.figure(figsize=(8, 6))
#     plt.imshow(pearson, aspect="auto")
#     plt.colorbar(label="Pearson correlation")
#     plt.xticks(range(len(pearson.columns)), pearson.columns, rotation=45)
#     plt.yticks(range(len(pearson.index)), pearson.index)
#     plt.title("Feature correlation matrix")
#     plt.tight_layout()
#     plt.savefig(REPORT_DIR / "correlation_matrix.png", dpi=150)
#     plt.close()


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     REPORT_DIR.mkdir(parents=True, exist_ok=True)

#     print("=" * 65)
#     print("PATIENT DATASET V2 VALIDATION")
#     print("=" * 65)

#     dataset, all_dataset, patients, observations = load_inputs()

#     print(f"Primary dataset shape: {dataset.shape}")
#     print(f"All-patient dataset shape: {all_dataset.shape}")
#     print(f"Synthea patients: {len(patients)}")

#     index_date = identify_index_date(dataset, observations)
#     print(f"Common index date: {index_date}")

#     checks, patient_col = validate_integrity(
#         dataset, all_dataset, patients
#     )

#     feature_checks, missingness = validate_features(dataset)
#     checks.extend(feature_checks)

#     stats, outliers, plausibility = calculate_statistics(dataset)

#     gender_summary, gender_checks = validate_gender(dataset)
#     checks.extend(gender_checks)

#     pearson, spearman = calculate_correlations(dataset)

#     unit_summary, unit_checks, selected_obs = audit_units(
#         observations, index_date
#     )
#     checks.extend(unit_checks)

#     temporal_summary, temporal_pairs = audit_measurement_dates(
#         selected_obs, index_date, patient_col
#     )

#     checks.extend(distribution_checks(dataset))

#     # Check whether the complete cohort is actually complete.
#     primary_complete = all(
#         pd.to_numeric(dataset[f], errors="coerce").notna().all()
#         for f in NUMERIC_FEATURES
#     )

#     add_check(
#         checks,
#         "Primary feature matrix is complete",
#         primary_complete,
#         "All six primary features must be present for every patient.",
#     )

#     # Save reports.
#     save_csv(pd.DataFrame(checks), "validation_checks.csv")
#     save_csv(missingness, "feature_missingness.csv")
#     save_csv(stats, "descriptive_statistics.csv")
#     save_csv(outliers, "iqr_outlier_summary.csv")
#     save_csv(plausibility, "screening_bounds_summary.csv")
#     save_csv(gender_summary, "gender_distribution.csv")
#     save_csv(unit_summary, "observation_unit_summary.csv")
#     save_csv(temporal_summary, "measurement_recency_summary.csv")
#     save_csv(temporal_pairs, "measurement_date_gap_summary.csv")

#     create_plots(dataset, pearson)

#     # A strict technical gate: any FAIL means the dataset is not
#     # ready until the issue is resolved. Warnings require review.
#     check_df = pd.DataFrame(checks)
#     fail_count = int((check_df["status"] == "FAIL").sum())
#     warning_count = int((check_df["status"] == "WARNING").sum())

#     if fail_count == 0 and primary_complete:
#         verdict = "TECHNICAL_CHECKS_PASSED_REVIEW_WARNINGS"
#     else:
#         verdict = "NOT_READY"

#     summary = {
#         "dataset": DATASET_FILE.name,
#         "expected_patients": EXPECTED_PATIENTS,
#         "actual_patients": int(len(dataset)),
#         "feature_count": len(FEATURES),
#         "features": FEATURES,
#         "index_date": str(index_date),
#         "window_months": WINDOW_MONTHS,
#         "fail_count": fail_count,
#         "warning_count": warning_count,
#         "verdict": verdict,
#         "interpretation": (
#             "Passing technical checks supports use in controlled "
#             "clustering benchmarks. It does not establish clinical "
#             "validity or representativeness."
#         ),
#     }

#     with open(REPORT_DIR / "validation_summary.json", "w") as f:
#         json.dump(summary, f, indent=4)

#     print("\n" + "=" * 65)
#     print("VALIDATION RESULT")
#     print("=" * 65)
#     print(f"Patients: {len(dataset)}")
#     print(f"Features: {len(FEATURES)}")
#     print(f"Failed checks: {fail_count}")
#     print(f"Warnings: {warning_count}")
#     print(f"Verdict: {verdict}")
#     print(f"Reports saved to: {REPORT_DIR}")

#     if warning_count:
#         print("\nReview all WARNING rows in validation_checks.csv.")
#         print("Warnings do not automatically mean the data is unusable.")


# if __name__ == "__main__":
#     main()



"""
Validate the 24-month Synthea patient dataset for clustering benchmarks.

Inputs:
    output/processed/patient_dataset_24m_primary.csv
    output/processed/patient_dataset_24m_all.csv
    output/csv/patients.csv
    output/csv/observations.csv

Outputs:
    output/validation/validation_summary.json
    output/validation/validation_checks.csv
    output/validation/descriptive_statistics.csv
    output/validation/observation_unit_summary.csv
    output/validation/measurement_date_gap_summary.csv
    output/validation/iqr_outlier_summary.csv

This script validates but never modifies the source datasets.
"""

from pathlib import Path
import json
import argparse
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "bmi",
]

OBSERVATION_FEATURES = {
    "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp",
    "Heart rate": "heart_rate",
    "Respiratory rate": "respiratory_rate",
    "Body mass index (BMI) [Ratio]": "bmi",
    "Body Height": "body_height",
    "Body Weight": "body_weight",
}

PRIMARY_REQUIRED = FEATURES

EXPECTED_PRIMARY_PATIENTS = 919
EXPECTED_ALL_PATIENTS = 1142
WINDOW_MONTHS = 24

# These are expected units for auditing only.
# The script reports observed units and does not convert values.
EXPECTED_UNITS = {
    "systolic_bp": {"mm[Hg]"},
    "diastolic_bp": {"mm[Hg]"},
    "heart_rate": {"/min"},
    "respiratory_rate": {"/min"},
    "bmi": {"kg/m2"},
    "body_height": {"cm"},
    "body_weight": {"kg"},
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def detect_id_column(df):
    """Find the patient ID column in a dataset."""
    candidates = ["PATIENT", "Id", "id", "patient_id", "Patient"]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify patient ID column. "
        f"Available columns: {list(df.columns)}"
    )


def add_check(checks, name, passed, details):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details),
    })


def numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def parse_dates(series):
    return (
        pd.to_datetime(series, errors="coerce", utc=True)
        .dt.tz_localize(None)
    )


def safe_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


# ---------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------

def validate(args):
    root = Path(args.project_dir)

    processed_dir = root / "output" / "processed"
    raw_dir = root / "output" / "csv"
    output_dir = root / "output" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_path = processed_dir / "patient_dataset_24m_primary.csv"
    all_path = processed_dir / "patient_dataset_24m_all.csv"
    patients_path = raw_dir / "patients.csv"
    observations_path = raw_dir / "observations.csv"

    for path in [
        primary_path, all_path, patients_path, observations_path
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("=" * 72)
    print("SYNTHEA PATIENT DATASET VALIDATION V2")
    print("=" * 72)

    primary = pd.read_csv(primary_path)
    all_data = pd.read_csv(all_path)
    patients = pd.read_csv(patients_path, usecols=["Id", "BIRTHDATE", "GENDER"])

    observations = pd.read_csv(
        observations_path,
        usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE", "UNITS"],
        low_memory=False,
    )

    checks = []

    # -----------------------------------------------------------------
    # 1. Identify IDs and validate cohort sizes
    # -----------------------------------------------------------------

    primary_id_col = detect_id_column(primary)
    all_id_col = detect_id_column(all_data)

    patient_ids = set(patients["Id"].astype(str))
    primary_ids = primary[primary_id_col].astype(str)
    all_ids = all_data[all_id_col].astype(str)

    add_check(
        checks,
        "Expected primary cohort size",
        len(primary) == EXPECTED_PRIMARY_PATIENTS,
        f"Observed {len(primary)}; expected {EXPECTED_PRIMARY_PATIENTS}.",
    )

    add_check(
        checks,
        "Unique patient IDs in primary cohort",
        primary_ids.nunique() == len(primary),
        f"Duplicate IDs: {primary_ids.duplicated().sum()}.",
    )

    unknown_primary_ids = set(primary_ids) - patient_ids

    add_check(
        checks,
        "Primary cohort IDs exist in Synthea patients",
        len(unknown_primary_ids) == 0,
        f"Unknown patient IDs: {len(unknown_primary_ids)}.",
    )

    add_check(
        checks,
        "All-patient dataset size",
        len(all_data) == len(patients),
        f"All dataset rows={len(all_data)}, patients.csv rows={len(patients)}.",
    )

    add_check(
        checks,
        "All-patient dataset IDs are unique",
        all_ids.nunique() == len(all_data),
        f"Duplicate IDs: {all_ids.duplicated().sum()}.",
    )

    # -----------------------------------------------------------------
    # 2. Validate primary feature matrix
    # -----------------------------------------------------------------

    missing_columns = [c for c in PRIMARY_REQUIRED if c not in primary.columns]

    add_check(
        checks,
        "Required primary feature columns",
        len(missing_columns) == 0,
        f"Missing columns: {missing_columns}.",
    )

    if missing_columns:
        raise ValueError(
            f"Primary dataset is missing required columns: {missing_columns}"
        )

    for feature in FEATURES:
        values = numeric_series(primary[feature])

        missing_count = int(values.isna().sum())
        non_numeric_count = int(
            (primary[feature].notna() & values.isna()).sum()
        )
        infinite_count = int(np.isinf(values.dropna()).sum())

        add_check(
            checks,
            f"{feature}: no missing values",
            missing_count == 0,
            f"Missing values: {missing_count}.",
        )

        add_check(
            checks,
            f"{feature}: numeric and finite",
            non_numeric_count == 0 and infinite_count == 0,
            f"Non-numeric={non_numeric_count}; infinite={infinite_count}.",
        )

        primary[feature] = values

    # -----------------------------------------------------------------
    # 3. Gender distribution
    # -----------------------------------------------------------------

    gender_col = "GENDER" if "GENDER" in primary.columns else (
        "gender" if "gender" in primary.columns else None
    )

    if gender_col:
        gender_counts = (
            primary[gender_col]
            .fillna("MISSING")
            .astype(str)
            .value_counts()
            .to_dict()
        )

        add_check(
            checks,
            "Gender values inspected",
            True,
            json.dumps(gender_counts),
        )
    else:
        gender_counts = {}
        add_check(
            checks,
            "Gender values inspected",
            False,
            "No gender column found in primary dataset.",
        )

    # -----------------------------------------------------------------
    # 4. Load and filter relevant raw observations
    # -----------------------------------------------------------------

    observations["DATE"] = parse_dates(observations["DATE"])
    observations["PATIENT"] = observations["PATIENT"].astype(str)
    observations["VALUE_NUMERIC"] = numeric_series(observations["VALUE"])

    observations = observations[
        observations["DESCRIPTION"].isin(OBSERVATION_FEATURES)
    ].copy()

    observations["feature"] = observations["DESCRIPTION"].map(
        OBSERVATION_FEATURES
    )

    index_date = observations["DATE"].max()

    if pd.isna(index_date):
        raise ValueError("No valid dates found in selected observations.")

    window_start = index_date - pd.DateOffset(months=WINDOW_MONTHS)

    window_obs = observations[
        observations["DATE"].between(window_start, index_date)
        & observations["VALUE_NUMERIC"].notna()
    ].copy()

    print(f"\nIndex date: {index_date}")
    print(f"24-month window: {window_start} to {index_date}")

    # -----------------------------------------------------------------
    # 5. Unit consistency audit
    # -----------------------------------------------------------------

    unit_rows = []

    for feature in sorted(window_obs["feature"].dropna().unique()):
        feature_obs = window_obs[window_obs["feature"] == feature]

        unit_counts = (
            feature_obs["UNITS"]
            .fillna("")
            .astype(str)
            .value_counts(dropna=False)
        )

        for unit, count in unit_counts.items():
            unit_rows.append({
                "feature": feature,
                "UNITS": unit if unit else "(blank)",
                "observation_count": int(count),
            })

        observed_units = set(
            feature_obs["UNITS"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        observed_units.discard("")
        blank_count = int(
            feature_obs["UNITS"].isna().sum()
            + (feature_obs["UNITS"].astype(str).str.strip() == "").sum()
        )

        expected = EXPECTED_UNITS.get(feature)

        unit_ok = (
            expected is not None
            and observed_units == expected
            and blank_count == 0
        )

        add_check(
            checks,
            f"{feature}: unit consistency",
            unit_ok,
            f"Units={sorted(observed_units)}; "
            f"blank-unit observations={blank_count}; "
            f"observations={len(feature_obs)}.",
        )

    unit_summary = pd.DataFrame(unit_rows)

    # -----------------------------------------------------------------
    # 6. Descriptive statistics and IQR outliers
    # -----------------------------------------------------------------

    stats_rows = []
    outlier_rows = []

    for feature in FEATURES:
        x = primary[feature].dropna()

        q1 = x.quantile(0.25)
        median = x.median()
        q3 = x.quantile(0.75)
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        outliers = x[(x < lower_fence) | (x > upper_fence)]

        stats_rows.append({
            "feature": feature,
            "count": int(x.count()),
            "mean": x.mean(),
            "std": x.std(),
            "min": x.min(),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": x.max(),
            "iqr": iqr,
            "iqr_lower_fence": lower_fence,
            "iqr_upper_fence": upper_fence,
        })

        outlier_rows.append({
            "feature": feature,
            "outlier_count": int(len(outliers)),
            "outlier_percentage": (
                len(outliers) / len(x) * 100 if len(x) else 0
            ),
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "minimum_outlier": outliers.min() if len(outliers) else np.nan,
            "maximum_outlier": outliers.max() if len(outliers) else np.nan,
            "action": "Review only; no observations removed",
        })

    stats_df = pd.DataFrame(stats_rows)
    outlier_df = pd.DataFrame(outlier_rows)

    # -----------------------------------------------------------------
    # 7. Respiratory-rate concentration
    # -----------------------------------------------------------------

    rr_counts = primary["respiratory_rate"].value_counts(normalize=True)

    most_common_rr = float(rr_counts.index[0])
    rr_share = float(rr_counts.iloc[0] * 100)

    # A concentration above 50% is flagged for sensitivity analysis.
    rr_concentrated = rr_share > 50

    add_check(
        checks,
        "Respiratory-rate concentration",
        True,
        f"Most common value={most_common_rr}; "
        f"share={rr_share:.2f}%. "
        f"{'Review for sensitivity analysis.' if rr_concentrated else 'Concentration recorded for sensitivity analysis.'}",
    )

    # -----------------------------------------------------------------
    # 8. Adult and pediatric population
    # -----------------------------------------------------------------

    adult_count = int((primary["age"] >= 18).sum())
    pediatric_count = int((primary["age"] < 18).sum())

    add_check(
        checks,
        "Adult and pediatric population",
        True,
        f"Adults (age >= 18): {adult_count}; under 18: {pediatric_count}.",
    )

    # -----------------------------------------------------------------
    # 9. Latest measurement date gaps
    # -----------------------------------------------------------------

    date_features = ["bmi", "body_height", "body_weight"]
    date_obs = window_obs[
        window_obs["feature"].isin(date_features)
    ].copy()

    latest_dates = (
        date_obs.sort_values("DATE")
        .groupby(["PATIENT", "feature"])["DATE"]
        .max()
        .unstack("feature")
    )

    gap_rows = []

    for i, feature_a in enumerate(date_features):
        for feature_b in date_features[i + 1:]:
            if feature_a not in latest_dates.columns:
                continue
            if feature_b not in latest_dates.columns:
                continue

            paired = latest_dates[[feature_a, feature_b]].dropna()

            gaps = (
                paired[feature_a] - paired[feature_b]
            ).abs().dt.total_seconds() / 86400

            gap_rows.append({
                "feature_a": feature_a,
                "feature_b": feature_b,
                "patients_with_both": int(len(gaps)),
                "median_absolute_gap_days": (
                    float(gaps.median()) if len(gaps) else np.nan
                ),
                "max_absolute_gap_days": (
                    float(gaps.max()) if len(gaps) else np.nan
                ),
                "gap_over_180_days": int((gaps > 180).sum()),
            })

    gap_df = pd.DataFrame(gap_rows)

    # -----------------------------------------------------------------
    # 10. Final checks and output
    # -----------------------------------------------------------------

    add_check(
        checks,
        "Primary feature matrix is complete",
        not primary[FEATURES].isna().any().any(),
        "All six primary features must be present for every patient.",
    )

    checks_df = pd.DataFrame(checks)

    fail_count = int((checks_df["status"] == "FAIL").sum())

    # Review notes are not failures. They identify analytical caveats.
    review_notes = [
        "The cohort includes pediatric patients; run an adult-only sensitivity experiment.",
        "Respiratory rate is concentrated around its common value; compare clustering with and without it.",
        "IQR outliers are flagged for review and must not be removed automatically.",
        "Synthea is synthetic data; these checks do not establish clinical validity or representativeness.",
    ]

    summary = {
        "dataset": primary_path.name,
        "expected_patients": EXPECTED_PRIMARY_PATIENTS,
        "actual_patients": int(len(primary)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "index_date": str(index_date),
        "window_months": WINDOW_MONTHS,
        "window_start": str(window_start),
        "fail_count": fail_count,
        "warning_count": len(review_notes),
        "verdict": (
            "TECHNICAL_CHECKS_PASSED_WITH_REVIEW_NOTES"
            if fail_count == 0
            else "NOT_READY_FOR_BENCHMARKING"
        ),
        "adult_patients": adult_count,
        "pediatric_patients": pediatric_count,
        "gender_counts": gender_counts,
        "respiratory_rate_most_common_value": most_common_rr,
        "respiratory_rate_most_common_share_percent": rr_share,
        "review_notes": review_notes,
        "interpretation": (
            "Passing technical checks supports use in controlled clustering "
            "benchmarks. It does not establish clinical validity or "
            "representativeness."
        ),
    }

    checks_df.to_csv(output_dir / "validation_checks.csv", index=False)
    stats_df.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    unit_summary.to_csv(
        output_dir / "observation_unit_summary.csv", index=False
    )
    gap_df.to_csv(
        output_dir / "measurement_date_gap_summary.csv", index=False
    )
    outlier_df.to_csv(
        output_dir / "iqr_outlier_summary.csv", index=False
    )

    with open(
        output_dir / "validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4, default=safe_json_value)

    # -----------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------

    print("\n" + "=" * 72)
    print("VALIDATION RESULTS")
    print("=" * 72)

    print(f"Primary cohort size : {len(primary)}")
    print(f"Features            : {len(FEATURES)}")
    print(f"Failed checks       : {fail_count}")
    print(f"Review notes        : {len(review_notes)}")

    print("\nDescriptive statistics:")
    print(stats_df.round(3).to_string(index=False))

    print("\nIQR outlier summary (review only):")
    print(outlier_df.round(3).to_string(index=False))

    print("\nMeasurement date gaps:")
    if len(gap_df):
        print(gap_df.round(2).to_string(index=False))
    else:
        print("No paired measurement dates available.")

    print("\nFinal verdict:")
    print(summary["verdict"])

    print("\nReview notes:")
    for note in review_notes:
        print(f"- {note}")

    print(f"\nReports saved to: {output_dir}")

    if fail_count:
        print("\nVALIDATION FAILED. Resolve failed checks before benchmarking.")
        return 1

    print("\nTECHNICAL VALIDATION PASSED.")
    return 0


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the 24-month Synthea patient dataset."
    )

    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Root directory of the research project.",
    )

    arguments = parser.parse_args()

    try:
        sys.exit(validate(arguments))
    except Exception as exc:
        print(f"\nVALIDATION ERROR: {exc}")
        sys.exit(2)
