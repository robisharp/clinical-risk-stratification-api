import joblib
import pandas as pd
from xgboost import XGBClassifier
from feature_engineering import feature_engineering_pipeline
from modelling_evaluation import run_phase2_clean_inline

def build_and_serialize_production_pipeline(raw_data_path: str):
    print("📦 Extracting and building production asset...")
    # Load and clean training data
    df = run_phase2_clean_inline(raw_data_path)
    X = df.drop(columns=['Has_Disease', 'Patient_ID', 'Name', 'Last_Visit_Date'], errors='ignore')
    y = df['Has_Disease']
    
    # Calculate imbalance profile natively
    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    
    # Fit the structural transformations 
    print("⚙️ Calibrating structural pipeline steps...")
    X_transformed = feature_engineering_pipeline.fit_transform(X, y)
    
    # Initialize the globally trained production estimator instance
    production_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=imbalance_ratio, missing=float('nan'),
        random_state=42, eval_metric="logloss"
    )
    production_model.fit(X_transformed, y)
    
    # Bundle components tightly into an export payload dict
    deployment_package = {
        "transformation_pipeline": feature_engineering_pipeline,
        "model": production_model,
        "optimal_threshold": 0.38  # Discovered from your Phase 4 tuning optimization step
    }
    
    joblib.dump(deployment_package, "healthcare_production_pipeline.joblib")
    print("💾 Production artifact successfully saved to 'healthcare_production_pipeline.joblib'")

if __name__ == "__main__":
    build_and_serialize_production_pipeline("Patient_Health_Records_Raw.csv")