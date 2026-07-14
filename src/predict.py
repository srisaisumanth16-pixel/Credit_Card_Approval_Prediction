import os
import joblib
import pandas as pd
import numpy as np

class CreditPredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            # Default path relative to project structure
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, 'models')
            
        self.scaler_path = os.path.join(models_dir, 'scaler.pkl')
        self.encoder_path = os.path.join(models_dir, 'encoder.pkl')
        self.model_path = os.path.join(models_dir, 'best_model.pkl')
        
        # Loaded assets
        self.scaler = None
        self.encoder = None
        self.model = None
        
        # Column names needed for mapping and transforming
        self.num_cols = ['AMT_INCOME_TOTAL', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']
        self.cat_cols = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
        ]
        self.flag_cols = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'IS_UNEMPLOYED']
        
        self.load_resources()
        
    def load_resources(self):
        """Loads model and preprocessing files if they exist."""
        if not (os.path.exists(self.scaler_path) and os.path.exists(self.encoder_path) and os.path.exists(self.model_path)):
            raise FileNotFoundError("Model files not found. Please train models first using train_models.py.")
            
        self.scaler = joblib.load(self.scaler_path)
        self.encoder = joblib.load(self.encoder_path)
        self.model = joblib.load(self.model_path)
        
    def predict(self, raw_input):
        """
        Takes raw_input dict, processes it, and predicts credit approval.
        Expected raw_input dict format:
        {
            'CODE_GENDER': 'M' or 'F',
            'FLAG_OWN_CAR': 'Y' or 'N',
            'FLAG_OWN_REALTY': 'Y' or 'N',
            'CNT_CHILDREN': int,
            'AMT_INCOME_TOTAL': float,
            'NAME_INCOME_TYPE': str,
            'NAME_EDUCATION_TYPE': str,
            'NAME_FAMILY_STATUS': str,
            'NAME_HOUSING_TYPE': str,
            'AGE_YEARS': float,
            'EMPLOYMENT_YEARS': float (0 if unemployed),
            'IS_UNEMPLOYED': 0 or 1,
            'FLAG_WORK_PHONE': 0 or 1,
            'FLAG_PHONE': 0 or 1,
            'FLAG_EMAIL': 0 or 1,
            'OCCUPATION_TYPE': str,
            'CNT_FAM_MEMBERS': int
        }
        """
        # Convert to DataFrame
        df_input = pd.DataFrame([raw_input])
        
        # Scale numerical features
        scaled_nums = self.scaler.transform(df_input[self.num_cols])
        
        # Encode categorical features
        encoded_cats = self.encoder.transform(df_input[self.cat_cols])
        
        # Extract binary flags
        flags = df_input[self.flag_cols].values
        
        # Combine all features
        X_trans = np.hstack((scaled_nums, encoded_cats, flags))
        
        # Predict class and probabilities
        pred_class = int(self.model.predict(X_trans)[0])
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)[0]
            # Approved probability (class 1)
            approval_prob = float(probs[1])
        else:
            # Fallback for models without predict_proba
            approval_prob = 1.0 if pred_class == 1 else 0.0
            
        # Determine risk category based on approval probability
        # Since Approved = 1 and Rejected = 0, high approval_prob means low risk.
        if approval_prob >= 0.85:
            risk_category = 'Low Risk'
        elif approval_prob >= 0.50:
            risk_category = 'Medium Risk'
        else:
            risk_category = 'High Risk'
            
        # Map prediction class to user-friendly status
        status = "Approved" if pred_class == 1 else "Rejected"
        
        return {
            'Prediction': status,
            'Confidence': round(approval_prob * 100, 2),
            'RiskCategory': risk_category,
            'ApprovedClass': pred_class
        }

if __name__ == '__main__':
    # Test predictor
    try:
        predictor = CreditPredictor()
        test_applicant = {
            'CODE_GENDER': 'F',
            'FLAG_OWN_CAR': 'Y',
            'FLAG_OWN_REALTY': 'Y',
            'CNT_CHILDREN': 0,
            'AMT_INCOME_TOTAL': 180000.0,
            'NAME_INCOME_TYPE': 'Working',
            'NAME_EDUCATION_TYPE': 'Higher education',
            'NAME_FAMILY_STATUS': 'Married',
            'NAME_HOUSING_TYPE': 'House / apartment',
            'AGE_YEARS': 32.5,
            'EMPLOYMENT_YEARS': 5.5,
            'IS_UNEMPLOYED': 0,
            'FLAG_WORK_PHONE': 0,
            'FLAG_PHONE': 1,
            'FLAG_EMAIL': 0,
            'OCCUPATION_TYPE': 'Core staff',
            'CNT_FAM_MEMBERS': 2
        }
        res = predictor.predict(test_applicant)
        print("Inference results on test applicant:")
        print(res)
    except Exception as e:
        print(f"Error during test inference: {e}")
