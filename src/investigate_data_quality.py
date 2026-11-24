
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "output" / "csv"

df = pd.read_csv(DATA_DIR / "patient_core_dataset.csv")
patients = pd.read_csv(DATA_DIR / "patients.csv")

# Join birth dates for inspection
patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"],
    errors="coerce"
)

data = df.merge(
    patients[["Id", "BIRTHDATE"]],
    left_on="PATIENT",
    right_on="Id",
    how="left"
)

print("\n1. NEGATIVE AGE RECORDS")
print(
    data.loc[
        data["age"] < 0,
        ["PATIENT", "BIRTHDATE", "age"]
    ].to_string(index=False)
)

print("\n2. AGE DISTRIBUTION")
print(
    pd.cut(
        data["age"],
        bins=[-1, 18, 30, 45, 60, 75, 120],
        labels=[
            "0-17", "18-29", "30-44",
            "45-59", "60-74", "75+"
        ]
    ).value_counts().sort_index()
)

print("\n3. RESPIRATORY RATE DISTRIBUTION")
print(
    data["respiratory_rate"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n4. EXTREME HEIGHT RECORDS")
print(
    data.loc[
        (data["body_height"] < 100) |
        (data["body_height"] > 200),
        ["PATIENT", "age", "body_height", "body_weight", "bmi"]
    ].to_string(index=False)
)

print("\n5. EXTREME WEIGHT RECORDS")
print(
    data.loc[
        (data["body_weight"] < 20) |
        (data["body_weight"] > 150),
        ["PATIENT", "age", "body_height", "body_weight", "bmi"]
    ].to_string(index=False)
)

print("\n6. FEATURE CORRELATION")
features = [
    "age", "bmi", "body_height", "body_weight",
    "diastolic_bp", "heart_rate",
    "respiratory_rate", "systolic_bp"
]

print(data[features].corr().round(3))
