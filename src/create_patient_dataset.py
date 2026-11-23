
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Load Synthea CSV files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "output" / "csv"

patients = pd.read_csv(CSV_DIR / "patients.csv")
observations = pd.read_csv(CSV_DIR / "observations.csv")

print("Patients:", len(patients))
print("Observations:", len(observations))

# --------------------------------------------------
# 2. Select core clinical features
# --------------------------------------------------

feature_map = {
    "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp",
    "Heart rate": "heart_rate",
    "Respiratory rate": "respiratory_rate",
    "Body mass index (BMI) [Ratio]": "bmi",
    "Body Weight": "body_weight",
    "Body Height": "body_height"
}

selected = observations[
    observations["DESCRIPTION"].isin(feature_map)
].copy()

selected["feature"] = selected["DESCRIPTION"].map(feature_map)

# Convert values to numeric
selected["VALUE"] = pd.to_numeric(
    selected["VALUE"],
    errors="coerce"
)

selected["DATE"] = pd.to_datetime(
    selected["DATE"],
    errors="coerce"
)

selected = selected.dropna(
    subset=["PATIENT", "VALUE", "DATE"]
)

# --------------------------------------------------
# 3. Aggregate observations per patient
# --------------------------------------------------

patient_features = selected.pivot_table(
    index="PATIENT",
    columns="feature",
    values="VALUE",
    aggfunc="median"
).reset_index()

# --------------------------------------------------
# 4. Add demographic features
# --------------------------------------------------

patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"],
    errors="coerce"
)

# Use a fixed reference date for reproducibility.
reference_date = pd.Timestamp("2026-01-01")

patients["age"] = (
    reference_date - patients["BIRTHDATE"]
).dt.days / 365.25

demographics = patients[
    ["Id", "age", "GENDER"]
].rename(columns={
    "Id": "PATIENT",
    "GENDER": "gender"
})

dataset = demographics.merge(
    patient_features,
    on="PATIENT",
    how="left"
)

# --------------------------------------------------
# 5. Handle BMI missing values
# --------------------------------------------------

dataset["bmi"] = dataset["bmi"].fillna(
    dataset["bmi"].median()
)

# --------------------------------------------------
# 6. Save dataset
# --------------------------------------------------

dataset.to_csv(
    CSV_DIR / "patient_core_dataset.csv",
index=False
)

print("\nDataset created successfully.")
print("Shape:", dataset.shape)
print("\nMissing values:")
print(dataset.isnull().sum())

print("\nFirst five rows:")
print(dataset.head())

print("CSV Directory:", CSV_DIR)
print("Patients file exists:", (CSV_DIR / "patients.csv").exists())
print("Observations file exists:", (CSV_DIR / "observations.csv").exists())

