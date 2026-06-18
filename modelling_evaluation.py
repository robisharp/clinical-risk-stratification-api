import re
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score

# Import the fixed pipeline definitions from File 1
from feature_engineering import feature_engineering_pipeline

warnings.filterwarnings("ignore")

def run_phase2_clean_inline(input_path: str):
    """Executes cleaning operations while explicitly keeping columns for Phase 3 parsing."""
    print("🧹 [Phase 2] Cleaning and sanitizing raw medical records...")
    df = pd.read_csv(input_path)
    cleaned_df = pd.DataFrame()
    
    # Track essential categorical structural columns
    cleaned_df['Patient_ID'] = df['Patient_ID'].fillna('UNKNOWN_ID').astype(str).str.strip()
    cleaned_df['Name'] = df['Name'].fillna('Unknown').astype(str).str.strip().str.title()
    cleaned_df['Medications'] = df['Medications']  # Maintained intact for the feature engineer!
    cleaned_df['City'] = df['City'].fillna('Unknown').astype(str).str.strip().str.title()
    cleaned_df['Diagnosis_Code'] = df['Diagnosis_Code'].fillna('UNKNOWN').astype(str).str.upper().str.strip()

    # Handle Numeric Vitals & Clinical Boundaries
    word_to_num = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}
    
    cleaned_df['Age'] = df['Age'].apply(lambda x: word_to_num.get(str(x).strip().lower(), pd.to_numeric(x, errors='coerce')))
    cleaned_df.loc[(cleaned_df['Age'] <= 0) | (cleaned_df['Age'] > 120), 'Age'] = np.nan
    
    cleaned_df['BMI'] = df['BMI'].apply(lambda x: pd.to_numeric(re.sub(r"[^\d.]", "", str(x)), errors='coerce') if pd.notna(x) else np.nan)
    cleaned_df.loc[(cleaned_df['BMI'] < 10) | (cleaned_df['BMI'] > 70), 'BMI'] = np.nan
    
    cleaned_df['Heart_Rate'] = df['Heart_Rate'].apply(lambda x: word_to_num.get(str(x).strip().lower(), pd.to_numeric(x, errors='coerce')))
    cleaned_df.loc[(cleaned_df['Heart_Rate'] <= 30) | (cleaned_df['Heart_Rate'] > 250), 'Heart_Rate'] = np.nan

    # Blood Pressure Regular Expression Separator
    def parse_bp(val):
        if pd.isna(val): return np.nan, np.nan
        match = re.search(r"(\d{2,3})\s*(?:/|-|over)\s*(\d{2,3})", str(val).lower().strip())
        if match:
            s, d = float(match.group(1)), float(match.group(2))
            if 50 <= s <= 250 and 30 <= d <= 150: return s, d
        return np.nan, np.nan

    bp_tuples = df['Blood_Pressure'].apply(parse_bp)
    cleaned_df['Systolic_BP'] = [t[0] for t in bp_tuples]
    cleaned_df['Diastolic_BP'] = [t[1] for t in bp_tuples]
    
    # Cholesterol Proxy Mapping & Gender Standardizing
    cleaned_df['Cholesterol'] = df['Cholesterol_Level'].apply(lambda x: 180.0 if str(x).lower().strip() == 'normal' else (260.0 if str(x).lower().strip() == 'high' else pd.to_numeric(x, errors='coerce')))
    cleaned_df['Gender'] = df['Gender'].apply(lambda x: "Female" if str(x).strip().lower() in ("f", "female") else ("Male" if str(x).strip().lower() in ("m", "male") else "Unknown"))

    # Align Ordinal Classes
    cleaned_df['Diabetic_Score'] = df['Diabetic'].apply(lambda x: 1.0 if str(x).strip().lower() in ("yes", "y", "1") else (0.0 if str(x).strip().lower() in ("no", "n", "0") else np.nan))
    cleaned_df['Smoker_Score'] = df['Smoker'].apply(lambda x: 1.0 if str(x).strip().lower() in ("yes", "y", "1") else (0.5 if str(x).strip().lower() in ("former", "ex-smoker") else (0.0 if str(x).strip().lower() in ("no", "n", "0") else np.nan)))

    # Temporal Feature Processing
    def parse_date(val):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%b %d %Y", "%B %d %Y", "%Y%m%d"):
            try: return pd.to_datetime(str(val).strip(), format=fmt)
            except: continue
        return pd.to_datetime(val, errors='coerce')

    cleaned_df['Last_Visit_Date'] = df['Last_Visit_Date'].apply(parse_date)
    baseline_date = datetime(2026, 6, 18)
    cleaned_df['Days_Since_Last_Visit'] = (baseline_date - cleaned_df['Last_Visit_Date']).dt.days
    
    word_to_days = {"one week": 7, "two weeks": 14, "three weeks": 21, "four weeks": 28, "a month": 30}
    cleaned_df['Follow_Up_Days'] = df['Follow_Up'].apply(lambda x: word_to_days.get(str(x).strip().lower(), pd.to_numeric(x, errors='coerce')))

    # Target Label Cleansing
    cleaned_df['Has_Disease'] = df['Has_Disease'].apply(lambda x: 1 if str(x).strip().lower() in ('1', '1.0', 'yes', 'true') else (0 if str(x).strip().lower() in ('0', '0.0', 'no', 'false') else np.nan))
    
    # Drop rows where target validation labels are missing entirely
    supervised_df = cleaned_df.dropna(subset=['Has_Disease']).copy()
    supervised_df['Has_Disease'] = supervised_df['Has_Disease'].astype(int)
    return supervised_df

def train_xgboost_pipeline(raw_data_path: str):
    # Initialize cleaning routine
    df = run_phase2_clean_inline(raw_data_path)
    
    # Separate design matrices, ignoring missing non-numeric columns safely
    X = df.drop(columns=['Has_Disease', 'Patient_ID', 'Name', 'Last_Visit_Date'], errors='ignore')
    y = df['Has_Disease']
    
    neg_count, pos_count = (y == 0).sum(), (y == 1).sum()
    imbalance_ratio = neg_count / pos_count
    print(f"📊 Class Balance Setup: No Disease={neg_count}, Has Disease={pos_count} (Ratio: {imbalance_ratio:.2f})")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_fold_f2_scores = []
    
    print("\n🏋️‍♂️ Running Stratified 5-Fold Cross Validation via XGBoost...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit-transform slices sequentially to eliminate data leakage risks
        X_train_transformed = feature_engineering_pipeline.fit_transform(X_train, y_train)
        X_val_transformed = feature_engineering_pipeline.transform(X_val)
        
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=imbalance_ratio, missing=np.nan,
            random_state=42, eval_metric="logloss"
        )
        
        model.fit(X_train_transformed, y_train, verbose=False)
        val_probs = model.predict_proba(X_val_transformed)[:, 1]
        oof_preds[val_idx] = val_probs
        
        fold_f2 = fbeta_score(y_val, (val_probs >= 0.5).astype(int), beta=2)
        test_fold_f2_scores.append(fold_f2)
        print(f"   🔹 Fold {fold+1} Baseline F2-Score: {fold_f2:.4f}")
        
    print(f"\n📈 Mean CV F2-Score (0.5 Baseline): {np.mean(test_fold_f2_scores):.4f}")
    print(f"🎯 Out-of-Fold ROC-AUC Score: {roc_auc_score(y, oof_preds):.4f}")
    
    # Threshold Optimization Routine
    print("\n⚙️ Tuning decision threshold to maximize F2-Score...")
    best_f2, best_threshold = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        score = fbeta_score(y, (oof_preds >= t).astype(int), beta=2)
        if score > best_f2:
            best_f2 = score
            best_threshold = t
            
    print(f"🚀 Optimal Threshold Discovered at: {best_threshold:.2f}")
    print(f"🔥 Optimized Global F2-Score: {best_f2:.4f}")
    
    final_preds = (oof_preds >= best_threshold).astype(int)
    print("\n📋 Final Classification Report (Optimized Threshold Scaling):")
    print(classification_report(y, final_preds, target_names=["No Disease", "Has Disease"]))

if __name__ == "__main__":
    train_xgboost_pipeline("Patient_Health_Records_Raw.csv")