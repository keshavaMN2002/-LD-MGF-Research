"""
Data Inspection Module for LD-MGF Research
Provides utilities to inspect raw datasets without modification.
"""

import os
import glob
import pandas as pd
import numpy as np


def inspect_heart_disease(base_dir="data/raw/heart_disease"):
    """Inspects Heart Disease raw files and returns a summary dict."""
    filepath = os.path.join(base_dir, "processed.cleveland.data")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(filepath, header=None, names=cols, na_values='?')
    
    summary = {
        'dataset': 'Heart Disease (Cleveland)',
        'path': filepath,
        'shape': df.shape,
        'missing_values': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        'columns': cols,
        'target_distribution': df['target'].value_counts().to_dict(),
        'head': df.head(3)
    }
    return summary


def inspect_nhanes(base_dir="data/raw/nhanes_2017_2018"):
    """Inspects NHANES 2017-2018 XPT files."""
    files = glob.glob(os.path.join(base_dir, "*.xpt"))
    file_summaries = {}
    for f in files:
        fname = os.path.basename(f)
        df = pd.read_sas(f, format='xport')
        file_summaries[fname] = {
            'shape': df.shape,
            'columns': list(df.columns[:8])
        }
    return {
        'dataset': 'NHANES 2017-2018',
        'file_count': len(files),
        'files': file_summaries
    }


def inspect_diabetes_130(base_dir="data/raw/diabetes_130"):
    """Inspects Diabetes 130-US Hospitals raw CSV."""
    filepath = os.path.join(base_dir, "diabetic_data.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df_sample = pd.read_csv(filepath, nrows=5000, na_values='?')
    total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1
    
    missing = df_sample.isnull().mean()
    high_missing = missing[missing > 0.3].to_dict()
    
    return {
        'dataset': 'Diabetes 130-US Hospitals',
        'path': filepath,
        'total_rows': total_rows,
        'total_cols': len(df_sample.columns),
        'high_missing_cols (>30%)': high_missing,
        'readmitted_classes': df_sample['readmitted'].value_counts().to_dict()
    }


if __name__ == "__main__":
    print("--- Inspecting Heart Disease ---")
    hd = inspect_heart_disease()
    print("Shape:", hd['shape'], "Missing:", hd['missing_values'])
    
    print("\n--- Inspecting NHANES ---")
    nh = inspect_nhanes()
    print("Files found:", nh['file_count'])
    
    print("\n--- Inspecting Diabetes 130 ---")
    db = inspect_diabetes_130()
    print("Rows:", db['total_rows'], "High missing:", db['high_missing_cols (>30%)'])
