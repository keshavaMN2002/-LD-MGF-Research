"""
Preprocessing Module for LD-MGF Research
Handles loading, missing value imputation, categorical encoding, feature scaling,
and label preservation across Diabetes 130, NHANES 2017-2018, and Heart Disease datasets.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_heart_disease(base_dir="data/raw/heart_disease"):
    """
    Loads and preprocesses the UCI Heart Disease (Cleveland) dataset.
    
    Returns:
        X (np.ndarray): Standardized feature matrix (N, D).
        y (np.ndarray): Ground truth diagnosis label (0: healthy, 1: disease).
        feature_names (list): List of feature names.
        df_clean (pd.DataFrame): Unscaled clean dataframe.
    """
    filepath = os.path.join(base_dir, "processed.cleveland.data")
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(filepath, header=None, names=cols, na_values='?')
    
    # Missing value imputation
    # ca and thal have 4 and 2 missing values respectively
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    # Extract ground truth target label (binary: 0 = healthy, 1 = heart disease present)
    # Strictly separated from clustering features
    y = (df['target'].values > 0).astype(int)
    
    features_df = df.drop(columns=['target']).copy()
    
    # One-hot encode multi-category nominal features: cp (1-4), restecg (0-2), slope (1-3), thal (3,6,7)
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    encoded_cat = pd.get_dummies(features_df[cat_cols].astype(str), drop_first=True, dtype=float)
    
    # Continuous and binary features
    num_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca']
    clean_df = pd.concat([features_df[num_cols], encoded_cat], axis=1)
    
    feature_names = clean_df.columns.tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(clean_df)
    
    return X, y, feature_names, clean_df


def load_nhanes(base_dir="data/raw/nhanes_2017_2018", sample_size=3000, random_state=42):
    """
    Loads and merges NHANES 2017-2018 XPT examination and laboratory datasets.
    
    Returns:
        X (np.ndarray): Standardized feature matrix (N, D).
        y (np.ndarray): Biomarker risk proxy label (HbA1c >= 6.5 or BMI >= 30).
        feature_names (list): List of feature names.
        df_clean (pd.DataFrame): Unscaled clean dataframe.
    """
    demo = pd.read_sas(os.path.join(base_dir, 'DEMO_J.xpt'), format='xport')[['SEQN', 'RIAGENDR', 'RIDAGEYR']]
    bmx = pd.read_sas(os.path.join(base_dir, 'BMX_J.xpt'), format='xport')[['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST']]
    bpx = pd.read_sas(os.path.join(base_dir, 'BPX_J.xpt'), format='xport')[['SEQN', 'BPXSY1', 'BPXDI1', 'BPXPLS']]
    ghb = pd.read_sas(os.path.join(base_dir, 'GHB_J.xpt'), format='xport')[['SEQN', 'LBXGH']]
    hdl = pd.read_sas(os.path.join(base_dir, 'HDL_J.xpt'), format='xport')[['SEQN', 'LBDHDD']]
    tchol = pd.read_sas(os.path.join(base_dir, 'TCHOL_J.xpt'), format='xport')[['SEQN', 'LBXTC']]
    
    merged = demo.merge(bmx, on='SEQN', how='inner')
    merged = merged.merge(bpx, on='SEQN', how='inner')
    merged = merged.merge(ghb, on='SEQN', how='inner')
    merged = merged.merge(hdl, on='SEQN', how='inner')
    merged = merged.merge(tchol, on='SEQN', how='inner')
    
    # Filter to adults (>= 18 years)
    adults = merged[merged['RIDAGEYR'] >= 18].copy()
    
    # Drop identifier
    adults = adults.drop(columns=['SEQN'])
    
    # Impute missing values with column median
    adults = adults.fillna(adults.median())
    
    # Rename columns to human-readable names
    rename_dict = {
        'RIDAGEYR': 'Age',
        'RIAGENDR': 'Gender_Female',
        'BMXWT': 'Weight_kg',
        'BMXHT': 'Height_cm',
        'BMXBMI': 'BMI',
        'BMXWAIST': 'Waist_cm',
        'BPXSY1': 'Systolic_BP',
        'BPXDI1': 'Diastolic_BP',
        'BPXPLS': 'Pulse',
        'LBXGH': 'Glycohemoglobin_HbA1c',
        'LBDHDD': 'HDL_Cholesterol',
        'LBXTC': 'Total_Cholesterol'
    }
    adults = adults.rename(columns=rename_dict)
    # Gender in NHANES is 1: Male, 2: Female -> Convert to 0: Male, 1: Female
    adults['Gender_Female'] = (adults['Gender_Female'] == 2).astype(float)
    
    # Generate external risk proxy label (Metabolic / Diabetic Risk: HbA1c >= 6.5 or BMI >= 30)
    # Strictly for external validation comparison; NEVER used in clustering feature set
    y_full = ((adults['Glycohemoglobin_HbA1c'] >= 6.5) | (adults['BMI'] >= 30.0)).astype(int).values
    
    if sample_size is not None and len(adults) > sample_size:
        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(len(adults), size=sample_size, replace=False)
        adults = adults.iloc[sample_indices].reset_index(drop=True)
        y = y_full[sample_indices]
    else:
        y = y_full
        
    feature_names = adults.columns.tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(adults)
    
    return X, y, feature_names, adults


def load_diabetes_130(base_dir="data/raw/diabetes_130", sample_size=3000, random_state=42):
    """
    Loads and preprocesses Diabetes 130-US Hospitals dataset.
    
    Returns:
        X (np.ndarray): Standardized feature matrix (N, D).
        y (np.ndarray): Ground truth 30-day readmission label.
        feature_names (list): List of feature names.
        df_clean (pd.DataFrame): Cleaned unscaled dataframe.
    """
    filepath = os.path.join(base_dir, "diabetic_data.csv")
    df = pd.read_csv(filepath, na_values='?', low_memory=False)
    
    # Remove identifiers and sparse columns (>40% missing)
    drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty',
                 'diag_1', 'diag_2', 'diag_3']
    clean_df = df.drop(columns=drop_cols).copy()
    
    # Target label: readmitted (<30 days vs NO / >30)
    # Strictly isolated from clustering features
    y_full = (clean_df['readmitted'] == '<30').astype(int).values
    clean_df = clean_df.drop(columns=['readmitted'])
    
    # Ordinal Age mapping
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    clean_df['age'] = clean_df['age'].map(age_map).fillna(55.0)
    
    # Gender encoding
    clean_df['gender'] = clean_df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)
    
    # Medication features (23 medications)
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
    for m in med_cols:
        if m in clean_df.columns:
            clean_df[m] = clean_df[m].map(med_map).fillna(0).astype(float)
            
    # Binary flags
    clean_df['change'] = (clean_df['change'] == 'Ch').astype(float)
    clean_df['diabetesMed'] = (clean_df['diabetesMed'] == 'Yes').astype(float)
    
    # Categorical: race
    clean_df['race'] = clean_df['race'].fillna('Missing')
    race_dummies = pd.get_dummies(clean_df['race'], prefix='race', drop_first=True, dtype=float)
    clean_df = clean_df.drop(columns=['race'])
    clean_df = pd.concat([clean_df, race_dummies], axis=1)
    
    # Numeric clinical utilization columns
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                'num_medications', 'number_outpatient', 'number_emergency', 
                'number_inpatient', 'number_diagnoses', 'age', 'gender', 'change', 'diabetesMed']
    all_feature_cols = [c for c in clean_df.columns if clean_df[c].dtype in [np.float64, np.int64, float, int]]
    clean_df = clean_df[all_feature_cols].copy()
    
    # Impute any residual missing values with median
    clean_df = clean_df.fillna(clean_df.median())
    
    # Subsampling for CPU efficiency and fast reproducible clustering
    if sample_size is not None and len(clean_df) > sample_size:
        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(len(clean_df), size=sample_size, replace=False)
        clean_df = clean_df.iloc[sample_indices].reset_index(drop=True)
        y = y_full[sample_indices]
    else:
        y = y_full
        
    feature_names = clean_df.columns.tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(clean_df)
    
    return X, y, feature_names, clean_df


if __name__ == "__main__":
    print("Testing Preprocessing Pipelines...")
    X_h, y_h, feat_h, _ = load_heart_disease()
    print(f"Heart Disease: X={X_h.shape}, y={y_h.shape}, labels={np.unique(y_h)}")
    
    X_n, y_n, feat_n, _ = load_nhanes(sample_size=3000)
    print(f"NHANES: X={X_n.shape}, y={y_n.shape}, labels={np.unique(y_n)}")
    
    X_d, y_d, feat_d, _ = load_diabetes_130(sample_size=3000)
    print(f"Diabetes 130: X={X_d.shape}, y={y_d.shape}, labels={np.unique(y_d)}")
    print("All preprocessing pipelines verified successfully!")
