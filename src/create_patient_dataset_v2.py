
"""
create_patient_dataset_v2.py

Purpose:
    Generate patient-level EHR datasets from Synthea observations.

Outputs:
    12-month and 24-month datasets:
        - all patients, preserving missing values
        - complete primary cohort
        - complete extended cohort

Aggregation:
    - Vital signs: median within observation window
    - BMI, height, weight: latest valid observation
    - Age: calculated at common index date
    - Gender: patient demographic record

Missingness:
    - No automatic imputation
    - Missingness indicators are generated

Validation:
    - Duplicate patient IDs
    - Missing patient identifiers
    - Invalid ages
    - Missing feature coverage
    - Complete-case cohort sizes
    - Numeric data types
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "output" / "csv"
OUTPUT_DIR = BASE_DIR / "output" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_FILE = DATA_DIR / "patients.csv"
OBSERVATION_FILE = DATA_DIR / "observations.csv"

WINDOWS = [12, 24]

# Synthea observation descriptions mapped to dataset columns.
FEATURE_MAP = {
    "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp",
    "Heart rate": "heart_rate",
    "Respiratory rate": "respiratory_rate",
    "Body mass index (BMI) [Ratio]": "bmi",
    "Body Height": "body_height",
    "Body Weight": "body_weight",
}

# Vital signs: aggregate using median.
MEDIAN_FEATURES = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
]

# Anthropometric measurements: use latest valid value.
LATEST_FEATURES = [
    "bmi",
    "body_height",
    "body_weight",
]

ALL_CLINICAL_FEATURES = (
    MEDIAN_FEATURES + LATEST_FEATURES
)

# Primary configuration: age, gender, four vital signs and BMI.
PRIMARY_FEATURES = [
    "age",
    "gender",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "bmi",
]

# Extended configuration: primary + height + weight.
EXTENDED_FEATURES = PRIMARY_FEATURES + [
    "body_height",
    "body_weight",
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading Synthea data...")

    if not PATIENT_FILE.exists():
        raise FileNotFoundError(
            f"Patients file not found: {PATIENT_FILE}"
        )

    if not OBSERVATION_FILE.exists():
        raise FileNotFoundError(
            f"Observations file not found: {OBSERVATION_FILE}"
        )

    patients = pd.read_csv(
        PATIENT_FILE,
        usecols=["Id", "BIRTHDATE", "GENDER"],
        low_memory=False
    )

    observations = pd.read_csv(
        OBSERVATION_FILE,
        usecols=[
            "DATE",
            "PATIENT",
            "DESCRIPTION",
            "VALUE"
        ],
        low_memory=False
    )

    # Normalize dates to timezone-naive UTC.
    patients["BIRTHDATE"] = (
        pd.to_datetime(
            patients["BIRTHDATE"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
    )

    observations["DATE"] = (
        pd.to_datetime(
            observations["DATE"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
    )

    # Convert measurement values to numeric.
    observations["VALUE_NUM"] = pd.to_numeric(
        observations["VALUE"],
        errors="coerce"
    )

    # Remove rows without patient IDs.
    patients = patients.dropna(subset=["Id"]).copy()
    observations = observations.dropna(
        subset=["PATIENT", "DATE"]
    ).copy()

    # Preserve original observation order for timestamp ties.
    observations["_row_order"] = np.arange(
        len(observations)
    )

    print("Patients loaded:", len(patients))
    print("Observations loaded:", len(observations))

    return patients, observations


# ============================================================
# 3. DATA VALIDATION BEFORE AGGREGATION
# ============================================================

def validate_source_data(patients, observations):

    print("\n" + "=" * 65)
    print("SOURCE DATA VALIDATION")
    print("=" * 65)

    duplicate_patients = patients["Id"].duplicated().sum()

    print("Duplicate patient IDs:", duplicate_patients)

    if duplicate_patients > 0:
        raise ValueError(
            "Duplicate patient IDs found in patients.csv"
        )

    if patients["Id"].isna().any():
        raise ValueError("Missing patient identifiers found.")

    if observations["DATE"].isna().any():
        raise ValueError("Invalid observation dates found.")

    if patients["BIRTHDATE"].isna().any():
        print(
            "WARNING: Missing or invalid birth dates:",
            patients["BIRTHDATE"].isna().sum()
        )

    unknown_patients = (
        set(observations["PATIENT"].unique())
        - set(patients["Id"].unique())
    )

    print(
        "Observation patient IDs not in patients.csv:",
        len(unknown_patients)
    )

    print(
        "Observation date range:",
        observations["DATE"].min(),
        "to",
        observations["DATE"].max()
    )

    print(
        "Birth date range:",
        patients["BIRTHDATE"].min(),
        "to",
        patients["BIRTHDATE"].max()
    )

    print(
        "Non-numeric observation values:",
        observations["VALUE_NUM"].isna().sum()
    )


# ============================================================
# 4. CALCULATE PATIENT AGE AT INDEX DATE
# ============================================================

def prepare_patient_base(patients, index_date):

    base = patients.copy()

    # Calculate age in years at the common index date.
    age_days = (
        index_date - base["BIRTHDATE"]
    ).dt.total_seconds() / 86400

    base["age"] = age_days / 365.2425

    # A birth date after the index date is invalid.
    invalid_age = (
        base["age"].notna() &
        (base["age"] < 0)
    )

    if invalid_age.any():
        print(
            "WARNING: Patients with birth dates after index date:",
            int(invalid_age.sum())
        )

        base.loc[invalid_age, "age"] = np.nan

    base = base.rename(
        columns={
            "Id": "PATIENT",
            "GENDER": "gender"
        }
    )

    # Keep age precision consistent.
    base["age"] = base["age"].round(3)

    return base[
        ["PATIENT", "age", "gender"]
    ].copy()


# ============================================================
# 5. AGGREGATE OBSERVATIONS FOR ONE WINDOW
# ============================================================

def aggregate_observations(
    observations,
    index_date,
    months
):

    start_date = (
        index_date - pd.DateOffset(months=months)
    )

    print("\nObservation window:", months, "months")
    print("Start:", start_date)
    print("End:", index_date)

    # Select only required feature observations.
    window = observations[
        observations["DESCRIPTION"].isin(FEATURE_MAP)
    ].copy()

    # Apply observation window.
    window = window[
        (window["DATE"] >= start_date) &
        (window["DATE"] <= index_date)
    ].copy()

    # Convert descriptions to standardized feature names.
    window["feature"] = window["DESCRIPTION"].map(
        FEATURE_MAP
    )

    # Exclude non-numeric measurements.
    invalid_values = window["VALUE_NUM"].isna().sum()

    print(
        "Non-numeric feature observations excluded:",
        invalid_values
    )

    window = window.dropna(
        subset=["VALUE_NUM"]
    ).copy()

    if window.empty:
        raise ValueError(
            f"No valid observations in {months}-month window."
        )

    # --------------------------------------------------------
    # A. Median aggregation for vital signs
    # --------------------------------------------------------

    vital_data = window[
        window["feature"].isin(MEDIAN_FEATURES)
    ]

    vital_aggregated = (
        vital_data
        .groupby(["PATIENT", "feature"])["VALUE_NUM"]
        .median()
        .unstack("feature")
    )

    # --------------------------------------------------------
    # B. Latest observation for BMI, height and weight
    # --------------------------------------------------------

    latest_data = window[
        window["feature"].isin(LATEST_FEATURES)
    ].copy()

    latest_data = latest_data.sort_values(
        ["DATE", "_row_order"]
    )

    latest_data = latest_data.drop_duplicates(
        subset=["PATIENT", "feature"],
        keep="last"
    )

    latest_aggregated = (
        latest_data
        .pivot(
            index="PATIENT",
            columns="feature",
            values="VALUE_NUM"
        )
    )

    # Combine aggregated measurements.
    aggregated = vital_aggregated.join(
        latest_aggregated,
        how="outer"
    )

    aggregated = aggregated.reset_index()

    # Ensure all feature columns exist.
    for feature in ALL_CLINICAL_FEATURES:
        if feature not in aggregated.columns:
            aggregated[feature] = np.nan

    aggregated = aggregated[
        ["PATIENT"] + ALL_CLINICAL_FEATURES
    ]

    return aggregated, start_date


# ============================================================
# 6. BUILD PATIENT-LEVEL DATASET
# ============================================================

def build_patient_dataset(
    patients,
    observations,
    index_date,
    months
):

    print("\nBuilding", months, "month patient dataset...")

    # Patient demographic base.
    base = prepare_patient_base(
        patients,
        index_date
    )

    # Aggregate clinical observations.
    aggregated, start_date = aggregate_observations(
        observations,
        index_date,
        months
    )

    # Merge clinical features with patient demographics.
    dataset = base.merge(
        aggregated,
        on="PATIENT",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Missingness indicators
    # --------------------------------------------------------

    for feature in ALL_CLINICAL_FEATURES:
        dataset[f"missing_{feature}"] = (
            dataset[feature].isna().astype(int)
        )

    dataset["missing_age"] = (
        dataset["age"].isna().astype(int)
    )

    dataset["missing_gender"] = (
        dataset["gender"].isna().astype(int)
    )

    # Number of missing clinical measurements.
    dataset["missing_clinical_count"] = (
        dataset[
            [f"missing_{f}" for f in ALL_CLINICAL_FEATURES]
        ].sum(axis=1)
    )

    # Metadata.
    dataset["window_months"] = months
    dataset["index_date"] = index_date

    # --------------------------------------------------------
    # Completeness indicators
    # --------------------------------------------------------

    dataset["complete_primary"] = (
        dataset[PRIMARY_FEATURES]
        .notna()
        .all(axis=1)
    )

    dataset["complete_extended"] = (
        dataset[EXTENDED_FEATURES]
        .notna()
        .all(axis=1)
    )

    # Sort for reproducibility.
    dataset = dataset.sort_values(
        "PATIENT"
    ).reset_index(drop=True)

    return dataset, start_date


# ============================================================
# 7. VALIDATE FINAL DATASET
# ============================================================

def validate_dataset(dataset, months):

    print("\n" + "=" * 65)
    print(f"{months}-MONTH DATASET VALIDATION")
    print("=" * 65)

    errors = []

    # Basic checks.
    total_patients = len(dataset)

    duplicate_ids = dataset["PATIENT"].duplicated().sum()

    print("Dataset shape:", dataset.shape)
    print("Total patients:", total_patients)
    print("Duplicate patient IDs:", duplicate_ids)

    if duplicate_ids > 0:
        errors.append("Duplicate patient IDs")

    if dataset["PATIENT"].isna().any():
        errors.append("Missing patient IDs")

    # Check required numeric columns.
    numeric_features = [
        "age"
    ] + ALL_CLINICAL_FEATURES

    for feature in numeric_features:

        if not pd.api.types.is_numeric_dtype(
            dataset[feature]
        ):
            errors.append(
                f"{feature} is not numeric"
            )

    # Check negative age.
    negative_age = (
        dataset["age"].notna() &
        (dataset["age"] < 0)
    )

    print(
        "Negative age records:",
        int(negative_age.sum())
    )

    if negative_age.any():
        errors.append("Negative age values")

    # Check missingness indicators.
    for feature in ALL_CLINICAL_FEATURES:

        flag = f"missing_{feature}"

        expected = dataset[feature].isna().astype(int)

        if not dataset[flag].equals(expected):
            errors.append(
                f"Incorrect missingness indicator: {flag}"
            )

    # Coverage report.
    print("\nFEATURE COVERAGE")

    coverage_rows = []

    for feature in ["age", "gender"] + ALL_CLINICAL_FEATURES:

        available = dataset[feature].notna().sum()

        missing = dataset[feature].isna().sum()

        coverage = (
            available / total_patients * 100
            if total_patients > 0 else 0
        )

        print(
            f"{feature:25s}: "
            f"{available:4d} available | "
            f"{missing:4d} missing | "
            f"{coverage:.2f}% coverage"
        )

        coverage_rows.append({
            "window_months": months,
            "feature": feature,
            "available_patients": available,
            "missing_patients": missing,
            "coverage_percent": round(coverage, 2)
        })

    # Cohort sizes.
    primary_count = int(
        dataset["complete_primary"].sum()
    )

    extended_count = int(
        dataset["complete_extended"].sum()
    )

    primary_coverage = (
        primary_count / total_patients * 100
    )

    extended_coverage = (
        extended_count / total_patients * 100
    )

    print("\nCOHORT SUMMARY")

    print(
        "Complete primary cohort:",
        primary_count,
        f"({primary_coverage:.2f}%)"
    )

    print(
        "Complete extended cohort:",
        extended_count,
        f"({extended_coverage:.2f}%)"
    )

    print(
        "Patients with missing clinical features:",
        int((dataset["missing_clinical_count"] > 0).sum())
    )

    # Gender distribution.
    print("\nGENDER DISTRIBUTION")

    print(
        dataset["gender"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Check age and gender availability.
    if dataset["age"].isna().any():
        warnings.warn("Some patients have missing age.")

    if dataset["gender"].isna().any():
        warnings.warn("Some patients have missing gender.")

    if errors:
        print("\nVALIDATION ERRORS")

        for error in errors:
            print("-", error)

        raise ValueError(
            f"Validation failed for {months}-month dataset."
        )

    print("\nValidation passed.")

    return {
        "window_months": months,
        "total_patients": total_patients,
        "primary_complete": primary_count,
        "primary_coverage_percent": round(
            primary_coverage, 2
        ),
        "extended_complete": extended_count,
        "extended_coverage_percent": round(
            extended_coverage, 2
        ),
        "duplicate_patient_ids": int(duplicate_ids),
        "negative_age_records": int(negative_age.sum())
    }, coverage_rows


# ============================================================
# 8. SAVE OUTPUT FILES
# ============================================================

def save_datasets(dataset, months):

    print("\nSaving", months, "month datasets...")

    # All patients, including patients with missing values.
    all_path = (
        OUTPUT_DIR /
        f"patient_dataset_{months}m_all.csv"
    )

    dataset.to_csv(
        all_path,
        index=False
    )

    # Complete primary cohort.
    primary = dataset[
        dataset["complete_primary"]
    ].copy()

    primary_path = (
        OUTPUT_DIR /
        f"patient_dataset_{months}m_primary.csv"
    )

    primary.to_csv(
        primary_path,
        index=False
    )

    # Complete extended cohort.
    extended = dataset[
        dataset["complete_extended"]
    ].copy()

    extended_path = (
        OUTPUT_DIR /
        f"patient_dataset_{months}m_extended.csv"
    )

    extended.to_csv(
        extended_path,
        index=False
    )

    print("Saved:", all_path)
    print("Saved:", primary_path)
    print("Saved:", extended_path)

    print(
        "All patients:",
        len(dataset)
    )

    print(
        "Primary complete cohort:",
        len(primary)
    )

    print(
        "Extended complete cohort:",
        len(extended)
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("SYNTHEA PATIENT DATASET GENERATION V2")
    print("=" * 70)

    patients, observations = load_data()

    validate_source_data(
        patients,
        observations
    )

    # Common index date for both datasets.
    index_date = observations["DATE"].max()

    if pd.isna(index_date):
        raise ValueError(
            "Could not determine observation index date."
        )

    print("\nCOMMON INDEX DATE:", index_date)

    validation_summaries = []
    coverage_summaries = []

    # Generate both observation windows.
    for months in WINDOWS:

        dataset, start_date = build_patient_dataset(
            patients,
            observations,
            index_date,
            months
        )

        summary, coverage = validate_dataset(
            dataset,
            months
        )

        summary["window_start"] = start_date
        summary["index_date"] = index_date

        validation_summaries.append(summary)

        coverage_summaries.extend(coverage)

        save_datasets(
            dataset,
            months
        )

    # Save combined validation summary.
    summary_df = pd.DataFrame(
        validation_summaries
    )

    coverage_df = pd.DataFrame(
        coverage_summaries
    )

    summary_path = (
        OUTPUT_DIR /
        "dataset_validation_summary.csv"
    )

    coverage_path = (
        OUTPUT_DIR /
        "feature_coverage_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    coverage_df.to_csv(
        coverage_path,
        index=False
    )

    # Final summary.
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED")
    print("=" * 70)

    print(summary_df.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*.csv")):
        print("-", file.name)


if __name__ == "__main__":
    main()
