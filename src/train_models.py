import os
import joblib
import pandas as pd
import numpy as np

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Project Modules
from data_preprocessing import preprocess_datasets, split_dataset
from feature_engineering import engineer_features, fit_preprocessing_objects, transform_features
from evaluate_models import (
    calculate_metrics, 
    plot_confusion_matrix, 
    plot_roc_curves, 
    plot_feature_importance, 
    plot_model_comparison
)

def train_and_evaluate_all(data_dir, models_dir, static_images_dir):
    """
    Main training pipeline:
    - Preprocess & Split
    - Feature Engineering
    - Model Training (Logistic Regression, Decision Tree, Random Forest, XGBoost)
    - Evaluation & Chart Generation
    - Model Selection & Saving
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_images_dir, exist_ok=True)
    
    print("Step 1: Preprocessing and Merging Raw Datasets...")
    df_merged = preprocess_datasets(data_dir)
    
    print("Step 2: Splitting Data into Train & Test sets...")
    X_train, X_test, y_train, y_test = split_dataset(df_merged)
    
    print("Step 3: Feature Engineering (Calculating Ages, Employment, Unemployed flags)...")
    X_train_eng = engineer_features(X_train)
    X_test_eng = engineer_features(X_test)
    
    print("Step 4: Fitting Scaler and OneHotEncoder on Train set...")
    scaler, encoder, num_cols, cat_cols = fit_preprocessing_objects(X_train_eng)
    
    print("Step 5: Transforming Train and Test datasets...")
    X_train_trans, feat_names = transform_features(X_train_eng, scaler, encoder, num_cols, cat_cols)
    X_test_trans, _ = transform_features(X_test_eng, scaler, encoder, num_cols, cat_cols)
    
    # Save the preprocessing objects
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(models_dir, 'encoder.pkl'))
    print("Saved scaler.pkl and encoder.pkl.")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = []
    trained_models = {}
    
    print("Step 6: Training and Evaluating Models...")
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        model.fit(X_train_trans, y_train)
        trained_models[model_name] = model
        
        # Predict
        y_pred = model.predict(X_test_trans)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        metrics['Model'] = model_name
        results.append(metrics)
        
        # Plot Confusion Matrix
        cm_path = os.path.join(static_images_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
        print(f"    Saved confusion matrix: {cm_path}")
        
    # Create Comparison Table
    comparison_df = pd.DataFrame(results)
    # Reorder columns
    comparison_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
    print("\n--- Model Performance Comparison ---")
    print(comparison_df.to_string(index=False))
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(models_dir, 'model_comparison.csv'), index=False)
    
    # Plot combined charts
    print("\nGenerating charts...")
    plot_roc_curves(trained_models, X_test_trans, y_test, os.path.join(static_images_dir, 'roc_curve.png'))
    plot_model_comparison(comparison_df, os.path.join(static_images_dir, 'model_comparison.png'))
    
    # Select the best model based on F1 Score
    # Wait, in credit card risk evaluation, F1 is extremely critical due to imbalance.
    best_model_row = comparison_df.sort_values(by='F1 Score', ascending=False).iloc[0]
    best_model_name = best_model_row['Model']
    best_model = trained_models[best_model_name]
    
    print(f"\n---> Best Model Selected: {best_model_name} (F1 Score: {best_model_row['F1 Score']})")
    
    # Save Best Model
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    print("Saved best_model.pkl successfully.")
    
    # Plot feature importance for best model if supported, otherwise fallback to XGBoost
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, feat_names, os.path.join(static_images_dir, 'feature_importance.png'))
    elif 'XGBoost' in trained_models:
        print("Best model does not support feature_importances_. Plotting XGBoost feature importance as representative.")
        plot_feature_importance(trained_models['XGBoost'], feat_names, os.path.join(static_images_dir, 'feature_importance.png'))
    
    return best_model_name, comparison_df

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'dataset')
    models_dir = os.path.join(project_root, 'models')
    static_images_dir = os.path.join(project_root, 'static', 'images')
    
    best_name, comp_table = train_and_evaluate_all(data_dir, models_dir, static_images_dir)
