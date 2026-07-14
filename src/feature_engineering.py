import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def engineer_features(X_df):
    """
    Applies feature engineering to the raw training or testing features:
    - Converts DAYS_BIRTH to AGE_YEARS (positive)
    - Extracts IS_UNEMPLOYED from DAYS_EMPLOYED
    - Converts DAYS_EMPLOYED to EMPLOYMENT_YEARS (positive, 0 for unemployed)
    - Drops unnecessary columns (ID, FLAG_MOBIL)
    """
    X = X_df.copy()
    
    # Calculate age in years
    X['AGE_YEARS'] = (-X['DAYS_BIRTH'] / 365.25).round(2)
    
    # Calculate employment years and handle unemployed flag
    X['IS_UNEMPLOYED'] = (X['DAYS_EMPLOYED'] == 365243).astype(int)
    X['EMPLOYMENT_YEARS'] = X['DAYS_EMPLOYED'].apply(lambda x: 0.0 if x == 365243 else (-x / 365.25))
    X['EMPLOYMENT_YEARS'] = X['EMPLOYMENT_YEARS'].round(2)
    
    # Drop raw date/time columns and columns with zero variance (FLAG_MOBIL)
    cols_to_drop = ['ID', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    return X

def fit_preprocessing_objects(X_train_engineered):
    """
    Fits StandardScaler and OneHotEncoder on the engineered training features.
    """
    # Identify column types
    numerical_cols = ['AMT_INCOME_TOTAL', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']
    categorical_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
    ]
    
    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(X_train_engineered[numerical_cols])
    
    # Fit Encoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(X_train_engineered[categorical_cols])
    
    return scaler, encoder, numerical_cols, categorical_cols

def transform_features(X_engineered, scaler, encoder, numerical_cols, categorical_cols):
    """
    Transforms engineered features using fitted scaler and encoder.
    Returns a unified numpy array or DataFrame.
    """
    # Scale numerical features
    scaled_nums = scaler.transform(X_engineered[numerical_cols])
    
    # Encode categorical features
    encoded_cats = encoder.transform(X_engineered[categorical_cols])
    encoded_cat_names = encoder.get_feature_names(categorical_cols)
    
    # Extract binary flags
    flag_cols = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'IS_UNEMPLOYED']
    flags = X_engineered[flag_cols].values
    
    # Combine all features
    X_transformed = np.hstack((scaled_nums, encoded_cats, flags))
    
    # Reconstruct columns list for debugging/feature importance
    feature_names = numerical_cols + list(encoded_cat_names) + flag_cols
    
    return X_transformed, feature_names

if __name__ == '__main__':
    from data_preprocessing import preprocess_datasets, split_dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'dataset')
    
    try:
        df_processed = preprocess_datasets(data_dir)
        X_train, X_test, y_train, y_test = split_dataset(df_processed)
        
        # Engineer
        X_train_eng = engineer_features(X_train)
        print("Engineered features successfully.")
        print(f"Engineered columns: {list(X_train_eng.columns)}")
        
        # Fit Preprocessors
        scaler, encoder, num_cols, cat_cols = fit_preprocessing_objects(X_train_eng)
        
        # Transform
        X_train_trans, feat_names = transform_features(X_train_eng, scaler, encoder, num_cols, cat_cols)
        print(f"Transformed features shape: {X_train_trans.shape}")
        print(f"Number of generated features: {len(feat_names)}")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
