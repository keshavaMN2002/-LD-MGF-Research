
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "output" / "csv"

obs = pd.read_csv(
    DATA_DIR / "observations.csv",
    usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE"]
)

patients = pd.read_csv(
    DATA_DIR / "patients.csv",
    usecols=["Id", "BIRTHDATE", "GENDER"]
)

obs["DATE"] = pd.to_datetime(
    obs["DATE"], errors="coerce", utc=True
).dt.tz_localize(None)

obs["VALUE_NUM"] = pd.to_numeric(obs["VALUE"], errors="coerce")

index_date = obs["DATE"].max()
print("Index date:", index_date)

features = [
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
    "Respiratory rate",
    "Body mass index (BMI) [Ratio]",
    "Body Weight",
    "Body Height"
]

total_patients = patients["Id"].nunique()

for months in [12, 24]:

    start_date = index_date - pd.DateOffset(months=months)

    window = obs[
        (obs["DATE"] >= start_date) &
        (obs["DATE"] <= index_date) &
        (obs["DESCRIPTION"].isin(features)) &
        (obs["VALUE_NUM"].notna())
    ].copy()

    print("\n" + "=" * 65)
    print(f"{months}-MONTH WINDOW")
    print("Start:", start_date)
    print("End:", index_date)

    feature_patients = {}

    for feature in features:
        ids = set(
            window.loc[
                window["DESCRIPTION"] == feature, "PATIENT"
            ].dropna().unique()
        )

        feature_patients[feature] = ids

        print(
            f"{feature:45s}: "
            f"{len(ids):4d} ({len(ids)/total_patients*100:.2f}%)"
        )

    complete_ids = set.intersection(
        *feature_patients.values()
    )

    # Primary feature configuration:
    # Four vital signs + BMI
    primary_features = [
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
        "Heart rate",
        "Respiratory rate",
        "Body mass index (BMI) [Ratio]"
    ]

    primary_ids = set.intersection(
        *(feature_patients[f] for f in primary_features)
    )

    print(
        "Patients complete for primary configuration:",
        len(primary_ids)
    )

    print(
        "Primary configuration coverage:",
        f"{len(primary_ids) / total_patients * 100:.2f}%"
    )

    paired_ids = (
        feature_patients["Body Height"] &
        feature_patients["Body Weight"]
    )

    any_feature_ids = set.union(*feature_patients.values())

    print("-" * 65)
    print("Patients with at least one core feature:",
          len(any_feature_ids))
    print("Patients with all seven core features:",
          len(complete_ids))
    print("Patients with both height and weight:",
          len(paired_ids))

    print(
        "Complete feature coverage:",
        f"{len(complete_ids)/total_patients*100:.2f}%"
    )
