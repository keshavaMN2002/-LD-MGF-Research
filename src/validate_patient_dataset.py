
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "output" / "csv"

df = pd.read_csv(DATA_DIR / "patient_core_dataset.csv")

# Features for statistical analysis
numeric_features = [
    "age",
    "bmi",
    "body_height",
    "body_weight",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp"
]

print("=" * 65)
print("1. DATASET INFORMATION")
print("=" * 65)

print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate patients:", df["PATIENT"].duplicated().sum())

# --------------------------------------------------
# 2. Descriptive statistics
# --------------------------------------------------

print("\n" + "=" * 65)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 65)

print(df[numeric_features].describe().round(2).T)

# --------------------------------------------------
# 3. Outlier inspection using IQR
# --------------------------------------------------

print("\n" + "=" * 65)
print("3. OUTLIER INSPECTION")
print("=" * 65)

for feature in numeric_features:

    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[feature] < lower) |
        (df[feature] > upper)
    ]

    print(
        f"{feature:20s} "
        f"Outliers: {len(outliers):4d} "
        f"Range: [{lower:.2f}, {upper:.2f}]"
    )

# --------------------------------------------------
# 4. Correlation analysis
# --------------------------------------------------

print("\n" + "=" * 65)
print("4. FEATURE CORRELATION")
print("=" * 65)

correlation = df[numeric_features].corr()

print(correlation.round(2))

correlation.to_csv(
    DATA_DIR / "feature_correlation.csv"
)

# --------------------------------------------------
# 5. Gender distribution
# --------------------------------------------------

print("\n" + "=" * 65)
print("5. GENDER DISTRIBUTION")
print("=" * 65)

print(df["gender"].value_counts())

# --------------------------------------------------
# 6. Save validation report
# --------------------------------------------------

df[numeric_features].describe().to_csv(
    DATA_DIR / "descriptive_statistics.csv"
)

print("\nValidation completed.")
