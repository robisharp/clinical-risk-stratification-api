import re
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

def clean_patient_data(input_path: str, output_path: str = "Patient_Health_Records_Clean.csv"):
    print("🚀 Starting End-to-End Data Cleaning Pipeline...")
    
    # 1. Load Raw Data
    df = pd.read_csv(input_path)
    print(f"📋 Raw Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Create a copy to prevent SettingWithCopyWarnings
    cleaned_df = pd.DataFrame()
    
    # ─── ID & Name ────────────────────────────────────────────────────────────
    cleaned_df['Patient_ID'] = df['Patient_ID'].fillna('UNKNOWN_ID').astype(str).str.strip()
    cleaned_df['Name'] = df['Name'].fillna('Unknown').astype(str).str.strip().str.title()
    
    # ─── Age Cleaning ─────────────────────────────────────────────────────────
    # Map text words, handle extreme outliers (<0 or >120)
    word_to_num = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}
    
    def parse_age(val):
        if pd.isna(val):
            return np.nan
        v_str = str(val).strip().lower()
        if v_str in word_to_num:
            return float(word_to_num[v_str])
        try:
            v = float(v_str)
            return v if 0 <= v <= 120 else np.nan
        except ValueError:
            return np.nan

    cleaned_df['Age'] = df['Age'].apply(parse_age)
    
    # ─── BMI Cleaning ─────────────────────────────────────────────────────────
    # Strip unit strings like 'kg/m2' and remove physiological anomalies
    def parse_bmi(val):
        if pd.isna(val):
            return np.nan
        try:
            # Strip anything that isn't a digit or a decimal point
            v = float(re.sub(r"[^\d.]", "", str(val)))
            return v if 10 <= v <= 70 else np.nan
        except ValueError:
            return np.nan

    cleaned_df['BMI'] = df['BMI'].apply(parse_bmi)
    
    # ─── Heart Rate Cleaning ──────────────────────────────────────────────────
    # Map text strings and treat extreme values (0, 500) as NaN
    def parse_heart_rate(val):
        hr_words = {"sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
        if pd.isna(val):
            return np.nan
        v_str = str(val).strip().lower()
        if v_str in hr_words:
            return float(hr_words[v_str])
        try:
            v = float(v_str)
            return v if 30 <= v <= 250 else np.nan
        except ValueError:
            return np.nan

    cleaned_df['Heart_Rate'] = df['Heart_Rate'].apply(parse_heart_rate)
    
    # ─── Blood Pressure Parsing ───────────────────────────────────────────────
    # Split '120 over 80', '120 - 80', and '120/80' formats into individual features
    def parse_bp(val):
        if pd.isna(val):
            return np.nan, np.nan
        # Regex looks for digits, arbitrary spacers (over, -, /), and trailing digits
        match = re.search(r"(\d{2,3})\s*(?:/|-|over)\s*(\d{2,3})", str(val).lower().strip())
        if match:
            sys = float(match.group(1))
            dia = float(match.group(2))
            # Basic medical boundary verification
            if 50 <= sys <= 250 and 30 <= dia <= 150:
                return sys, dia
        return np.nan, np.nan

    bp_tuples = df['Blood_Pressure'].apply(parse_bp)
    cleaned_df['Systolic_BP'] = [t[0] for t in bp_tuples]
    cleaned_df['Diastolic_BP'] = [t[1] for t in bp_tuples]
    
    # ─── Cholesterol Level ────────────────────────────────────────────────────
    # Resolve strings like 'normal'/'high' to numerical proxies while maintaining continuous metrics
    def parse_cholesterol(val):
        chol_map = {"normal": 180, "high": 260}
        if pd.isna(val):
            return np.nan
        v_str = str(val).strip().lower()
        if v_str in chol_map:
            return float(chol_map[v_str])
        try:
            return float(v_str)
        except ValueError:
            return np.nan

    cleaned_df['Cholesterol'] = df['Cholesterol_Level'].apply(parse_cholesterol)
    
    # ─── Gender Unification ───────────────────────────────────────────────────
    def normalize_gender(val):
        if pd.isna(val): return "Unknown"
        v = str(val).strip().lower()
        if v in ("f", "female"): return "Female"
        if v in ("m", "male"):   return "Male"
        return "Unknown"

    cleaned_df['Gender'] = df['Gender'].apply(normalize_gender)
    
    # ─── Diabetic & Smoker (Ordinal Mapping) ──────────────────────────────────
    # Map fields systematically to categorical scales
    def parse_diabetic(val):
        if pd.isna(val): return np.nan
        v = str(val).strip().lower()
        if v in ("yes", "y", "1", "true"): return 1.0
        if v in ("no", "n", "0", "false"): return 0.0
        return np.nan  # "Unknown" maps to NaN for imputation downstream

    def parse_smoker(val):
        if pd.isna(val): return np.nan
        v = str(val).strip().lower()
        if v in ("yes", "y", "1"): return 1.0
        if v in ("former", "ex-smoker"): return 0.5
        if v in ("no", "n", "0"): return 0.0
        return np.nan

    cleaned_df['Diabetic_Score'] = df['Diabetic'].apply(parse_diabetic)
    cleaned_df['Smoker_Score'] = df['Smoker'].apply(parse_smoker)
    
    # ─── Temporal Parsing & Feature Engineering ───────────────────────────────
    # Standardize string formats into pandas Datetime objects
    def parse_date(val):
        if pd.isna(val):
            return pd.NaT
        v_str = str(val).strip()
        # Attempt standard date pattern evaluations
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%b %d %Y", "%B %d %Y"):
            try:
                return pd.to_datetime(v_str, format=fmt)
            except (ValueError, TypeError):
                continue
        # Fallback to general purpose date parser if structured parsing misses
        try:
            return pd.to_datetime(v_str, errors='coerce')
        except:
            return pd.NaT

    cleaned_df['Last_Visit_Date'] = df['Last_Visit_Date'].apply(parse_date)
    
    # Engineer days_since_visit feature (using a fixed historical baseline date point)
    baseline_date = datetime(2026, 6, 18)
    cleaned_df['Days_Since_Last_Visit'] = (baseline_date - cleaned_df['Last_Visit_Date']).dt.days
    
    # Cast Follow_Up safely to numeric values
    def parse_follow_up(val):
        word_map = {"one week": 7, "two weeks": 14, "three weeks": 21, "four weeks": 28, "a month": 30}
        if pd.isna(val): return np.nan
        v_str = str(val).strip().lower()
        if v_str in word_map: return float(word_map[v_str])
        try:
            return float(v_str)
        except ValueError:
            return np.nan

    cleaned_df['Follow_Up_Days'] = df['Follow_Up'].apply(parse_follow_up)
    
    # ─── Text Mining Unstructured Notes ───────────────────────────────────────
    # Strip emojis and pull explicit flags directly out of raw texts
    def process_notes(val):
        if pd.isna(val):
            return "", 0
        v_str = str(val).strip()
        # Regex strips standard unicode emoji ranges
        clean_str = re.sub(r'[\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F1E0-\u1F1FF]', '', v_str)
        
        # High BP Flag Extraction
        bp_flag = 1 if re.search(r'\bbp\s+high\b|\bhypertension\b', clean_str.lower()) else 0
        return clean_str.strip(), bp_flag

    notes_processed = df['Notes'].apply(process_notes)
    cleaned_df['Cleaned_Notes'] = [n[0] for n in notes_processed]
    cleaned_df['Flag_High_BP_From_Notes'] = [n[1] for n in notes_processed]
    
    # Keep standard diagnosis codes
    cleaned_df['Diagnosis_Code'] = df['Diagnosis_Code'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
    
    # ─── Target Resolution (Has_Disease) ──────────────────────────────────────
    # Treat 'unknown' or text as NaN, preserving indices for split verification
    def parse_target(val):
        if pd.isna(val): return np.nan
        v = str(val).strip().lower()
        if v in ('1', '1.0', 'yes', 'true'): return 1
        if v in ('0', '0.0', 'no', 'false'): return 0
        return np.nan

    cleaned_df['Has_Disease'] = df['Has_Disease'].apply(parse_target)
    
    # ─── Pipeline Output Strategy ─────────────────────────────────────────────
    # Drop records where the label is missing for our baseline supervised estimator
    supervised_df = cleaned_df.dropna(subset=['Has_Disease'])
    supervised_df['Has_Disease'] = supervised_df['Has_Disease'].astype(int)
    
    # Export clean data
    supervised_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("✨ SANITIZATION COMPLETE")
    print("="*50)
    print(f"📦 Original Data Rows:   {df.shape[0]:,}")
    print(f"✅ Clean Supervised Rows: {supervised_df.shape[0]:,} (Dropped unlabelled/unknown target rows)")
    print(f"💾 Clean file saved to:  '{output_path}'")
    print("="*50 + "\n")
    
    return supervised_df

if __name__ == "__main__":
    # Test execution
    cleaned_data = clean_patient_data("Patient_Health_Records_Raw.csv")