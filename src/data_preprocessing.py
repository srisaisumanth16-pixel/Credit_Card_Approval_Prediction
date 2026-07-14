import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    """Loads the application and credit records from the specified directory."""
    app_path = os.path.join(data_dir, 'application_record.csv')
    credit_path = os.path.join(data_dir, 'credit_record.csv')
    
    if not os.path.exists(app_path) or not os.path.exists(credit_path):
        raise FileNotFoundError("Raw dataset CSV files not found. Run generate_data.py first.")
        
    df_app = pd.read_csv(app_path)
    df_credit = pd.read_csv(credit_path)
    return df_app, df_credit

def process_target(df_credit):
    """
    Transforms credit history status into a binary target label.
    Business Rules:
    - Status C (paid off), X (no loan), 0 (1-29 days late), 1 (30-59 days late) are classified as Good (1 / Approved).
    - Status 2 (60-89 days late), 3 (90-119 days late), 4 (120-149 days late), 5 (150+ days late) are classified as Bad (0 / Rejected).
    If an applicant has ANY record of status >= 2, they are labeled 0 (Rejected). Otherwise 1 (Approved).
    """
    # Map STATUS to numeric risk level
    status_map = {
        'C': -1,
        'X': -1,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5
    }
    
    df_credit['STATUS_NUMERIC'] = df_credit['STATUS'].map(status_map)
    
    # For each ID, find the maximum overdue level reached
    grouped = df_credit.groupby('ID')['STATUS_NUMERIC'].max().reset_index()
    
    # Label: 0 if max status is >= 2 (overdue for 60+ days), else 1 (Approved)
    grouped['target'] = grouped['STATUS_NUMERIC'].apply(lambda x: 0 if x >= 2 else 1)
    
    return grouped[['ID', 'target']]

def preprocess_datasets(data_dir):
    """
    Full preprocessing pipeline:
    - Load data
    - Drop duplicate IDs in application records
    - Compute targets from credit records
    - Merge application and target datasets on ID
    - Handle missing values in OCCUPATION_TYPE
    - Return a clean DataFrame
    """
    df_app, df_credit = load_data(data_dir)
    
    # Drop duplicates in application records based on ID
    df_app_clean = df_app.drop_duplicates(subset='ID', keep='first')
    
    # Construct target
    df_target = process_target(df_credit)
    
    # Merge datasets on ID (inner join to keep applicants with credit history)
    df_merged = pd.merge(df_app_clean, df_target, on='ID', how='inner')
    
    # Handle missing values
    # OCCUPATION_TYPE has missing values. If DAYS_EMPLOYED == 365243, they are Pensioner/Unemployed.
    # We will fill missing values according to employment status.
    def fill_occupation(row):
        if pd.isna(row['OCCUPATION_TYPE']):
            if row['DAYS_EMPLOYED'] == 365243:
                return 'Pensioner'
            else:
                return 'Other/Unspecified'
        return row['OCCUPATION_TYPE']
        
    df_merged['OCCUPATION_TYPE'] = df_merged.apply(fill_occupation, axis=1)
    
    return df_merged

def split_dataset(df, test_size=0.2, seed=42):
    """Splits the dataset into training and testing sets."""
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'dataset')
    
    try:
        df_processed = preprocess_datasets(data_dir)
        print("Data preprocessed successfully.")
        print(f"Merged Dataset Shape: {df_processed.shape}")
        print(f"Target Distribution:\n{df_processed['target'].value_counts(normalize=True)}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
