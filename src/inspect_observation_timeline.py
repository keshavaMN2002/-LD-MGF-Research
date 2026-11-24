
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "output" / "csv"

obs = pd.read_csv(
    DATA_DIR / "observations.csv",
    usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE"]
)

patients = pd.read_csv(
    DATA_DIR / "patients.csv",
    usecols=["Id", "BIRTHDATE"]
)

# Parse dates and normalize timezone
obs["DATE"] = pd.to_datetime(
    obs["DATE"], errors="coerce", utc=True
).dt.tz_localize(None)

patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"], errors="coerce", utc=True
).dt.tz_localize(None)

print("\n1. OBSERVATION DATE RANGE")
print("Earliest:", obs["DATE"].min())
print("Latest:", obs["DATE"].max())

print("\n2. PATIENT BIRTH DATE RANGE")
print("Earliest:", patients["BIRTHDATE"].min())
print("Latest:", patients["BIRTHDATE"].max())

# --------------------------------------------------
# Check anthropometric observations
# --------------------------------------------------

features = [
    "Body Height",
    "Body Weight",
    "Body mass index (BMI) [Ratio]"
]

anthro = obs[
    obs["DESCRIPTION"].isin(features)
].copy()

anthro = anthro.merge(
    patients,
    left_on="PATIENT",
    right_on="Id",
    how="left"
)

anthro["AGE_AT_OBSERVATION"] = (
    anthro["DATE"] - anthro["BIRTHDATE"]
).dt.days / 365.25

print("\n3. ANTHROPOMETRIC OBSERVATION AGE")

for feature in features:
    subset = anthro[
        anthro["DESCRIPTION"] == feature
    ]

    print(f"\n{feature}")
    print(
        subset["AGE_AT_OBSERVATION"]
        .describe()
        .round(2)
    )

# --------------------------------------------------
# Inspect selected suspicious patients
# --------------------------------------------------

suspicious_ids = [
    "6ba95ae9-97ce-7c13-4585-5d7d3b9b0298",
    "9e1f5622-285c-9159-b058-931c134295e3",
    "9e84c64c-9b2e-671a-0cfd-9d30a3e64756"
]

print("\n4. SELECTED PATIENT HISTORY")

for patient_id in suspicious_ids:

    subset = anthro[
        anthro["PATIENT"] == patient_id
    ].sort_values("DATE")

    print("\nPatient:", patient_id)

    print(
        subset[
            [
                "DATE",
                "DESCRIPTION",
                "VALUE",
                "AGE_AT_OBSERVATION"
            ]
        ].to_string(index=False)
    )

# --------------------------------------------------
# Observation coverage by recent time window
# --------------------------------------------------

print("\n5. COVERAGE IN LAST 12 MONTHS")

cutoff = obs["DATE"].max() - pd.DateOffset(months=12)

recent = obs[obs["DATE"] >= cutoff]

print("Window starts:", cutoff)
print("Window ends:", obs["DATE"].max())
print(
    "Patients with observations:",
    recent["PATIENT"].nunique()
)
print(
    "Total patients:",
    patients["Id"].nunique()
)
