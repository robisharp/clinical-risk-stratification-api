import re
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ==============================================================================
# PIPELINE STRUCTURE DEFINITIONS
# ==============================================================================

class HealthcareFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['Pulse_Pressure'] = X['Systolic_BP'] - X['Diastolic_BP']
        X['Medication_Count'] = X['Medications'].fillna('').astype(str).apply(
            lambda x: len(re.split(r'[,;]', x)) if x.strip() else 0
        )
        age_bins = [-1, 18, 35, 55, 75, 150]
        X['Age_Group'] = pd.cut(X['Age'], bins=age_bins, labels=[0, 1, 2, 3, 4]).astype(float)
        bmi_bins = [-1, 18.5, 24.9, 29.9, 150]
        X['BMI_Category'] = pd.cut(X['BMI'], bins=bmi_bins, labels=[0, 1, 2, 3]).astype(float)
        return X

class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, m=5.0):
        self.m = m  
        self.mapping_ = {}
    def fit(self, X, y):
        X_col = pd.Series(X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X).reset_index(drop=True)
        y_col = pd.Series(y).reset_index(drop=True)
        stats = y_col.groupby(X_col).agg(['count', 'mean'])
        self.global_mean_ = y_col.mean()
        smooth = (stats['count'] * stats['mean'] + self.m * self.global_mean_) / (stats['count'] + self.m)
        self.mapping_ = smooth.to_dict()
        return self
    def transform(self, X):
        X_col = pd.Series(X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X)
        return X_col.map(self.mapping_).fillna(self.global_mean_).to_frame()

# ==============================================================================
# FASTAPI LIFESPAN ENGINE
# ==============================================================================

pipeline_package = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_package
    try:
        pipeline_package = joblib.load("healthcare_production_pipeline.joblib")
        print("🚀 Serialized pipeline components loaded successfully into memory.")
    except Exception as e:
        print(f"⚠️ Warning: Model package could not be resolved. Trace: {e}")
    yield
    print("🛑 Shutting down web service instance.")

app = FastAPI(
    title="Clinical Risk API Backend",
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Requests (CORS) so your UI can communicate with the API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DATA LAYOUT SCHEMAS ──────────────────────────────────────────────────────
class PatientVitalsPayload(BaseModel):
    Age: str = Field(..., json_schema_extra={"example": "54"})
    Gender: str = Field(..., json_schema_extra={"example": "Female"})
    City: str = Field(..., json_schema_extra={"example": "Boston"})
    BMI: str = Field(..., json_schema_extra={"example": "28.4 kg/m²"})
    Blood_Pressure: str = Field(..., json_schema_extra={"example": "134/82"})
    Heart_Rate: str = Field(..., json_schema_extra={"example": "70"})
    Cholesterol_Level: str = Field(..., json_schema_extra={"example": "high"})
    Diabetic: str = Field(..., json_schema_extra={"example": "yes"})
    Smoker: str = Field(..., json_schema_extra={"example": "Former"})
    Medications: Optional[str] = Field(default="None", json_schema_extra={"example": "Aspirin; Metformin"})
    Last_Visit_Date: str = Field(..., json_schema_extra={"example": "2024-11-13"})
    Follow_Up: str = Field(..., json_schema_extra={"example": "two weeks"})
    Diagnosis_Code: str = Field(..., json_schema_extra={"example": "I10"})

class InferenceOutputResponse(BaseModel):
    disease_risk_probability: float
    high_risk_classification: int
    optimal_decision_threshold_applied: float
    clinical_status: str

# ─── PARSING CONVERSION LOOPS ─────────────────────────────────────────────────
def clean_single_api_record(raw_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.DataFrame()
    cleaned['Medications'] = raw_df['Medications']
    cleaned['City'] = raw_df['City'].fillna('Unknown').astype(str).str.strip().str.title()
    cleaned['Diagnosis_Code'] = raw_df['Diagnosis_Code'].fillna('UNKNOWN').astype(str).str.upper().str.strip()

    word_to_num = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}
    cleaned['Age'] = raw_df['Age'].apply(lambda x: word_to_num.get(str(x).strip().lower(), pd.to_numeric(x, errors='coerce')))
    cleaned['BMI'] = raw_df['BMI'].apply(lambda x: pd.to_numeric(re.sub(r"[^\d.]", "", str(x)), errors='coerce') if pd.notna(x) else np.nan)
    cleaned['Heart_Rate'] = raw_df['Heart_Rate'].apply(lambda x: word_to_num.get(str(x).strip().lower(), pd.to_numeric(x, errors='coerce')))

    def parse_bp(val):
        if pd.isna(val): return np.nan, np.nan
        match = re.search(r"(\d{2,3})\s*(?:/|-|over)\s*(\d{2,3})", str(val).lower().strip())
        if match: return float(match.group(1)), float(match.group(2))
        return np.nan, np.nan

    bp_tuples = raw_df['Blood_Pressure'].apply(parse_bp)
    cleaned['Systolic_BP'] = [t[0] for t in bp_tuples]
    cleaned['Diastolic_BP'] = [t[1] for t in bp_tuples]
    cleaned['Cholesterol'] = raw_df['Cholesterol_Level'].apply(lambda x: 180.0 if str(x).lower().strip() == 'normal' else (260.0 if str(x).lower().strip() == 'high' else pd.to_numeric(x, errors='coerce')))
    cleaned['Gender'] = raw_df['Gender'].apply(lambda x: "Female" if str(x).strip().lower() in ("f", "female") else ("Male" if str(x).strip().lower() in ("m", "male") else "Unknown"))
    cleaned['Diabetic_Score'] = raw_df['Diabetic'].apply(lambda x: 1.0 if str(x).strip().lower() in ("yes", "y", "1") else (0.0 if str(x).strip().lower() in ("no", "n", "0") else np.nan))
    cleaned['Smoker_Score'] = raw_df['Smoker'].apply(lambda x: 1.0 if str(x).strip().lower() in ("yes", "y", "1") else (0.5 if str(x).strip().lower() in ("former", "ex-smoker") else (0.0 if str(x).strip().lower() in ("no", "n", "0") else np.nan)))

    def parse_date(val):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%b %d %Y", "%B %d %Y", "%Y%m%d"):
            try: return pd.to_datetime(str(val).strip(), format=fmt)
            except: continue
        return pd.to_datetime(val, errors='coerce')

    parsed_date = raw_df['Last_Visit_Date'].apply(parse_date)
    cleaned['Days_Since_Last_Visit'] = (datetime(2026, 6, 18) - parsed_date).dt.days
    word_to_days = {"one week": 7, "two weeks": 14, "three weeks": 21, "four weeks": 28, "a month": 30}
    cleaned['Follow_Up_Days'] = raw_df['Follow_Up'].apply(lambda x: word_to_days.get(str(x).strip().lower(), pd.to_numeric(x, errors='coerce')))
    return cleaned

# ─── SERVE THE INTERACTIVE UI HOMEPAGE ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    """Reads the static index.html file directly from the local directory."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h3>Error: index.html file was not found in the server folder!</h3>", status_code=404)

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System Health"])
def health_check():
    if pipeline_package is None:
        raise HTTPException(status_code=503, detail="Pipeline package is missing.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=InferenceOutputResponse, status_code=status.HTTP_200_OK, tags=["Clinical Analysis"])
def generate_patient_inference(payload: PatientVitalsPayload):
    if pipeline_package is None:
        raise HTTPException(status_code=503, detail="Model Engine Offline.")
    try:
        raw_input_df = pd.DataFrame([payload.model_dump()])
        sanitized_input_row = clean_single_api_record(raw_input_df)
        
        transformer = pipeline_package["transformation_pipeline"]
        clf_model = pipeline_package["model"]
        decision_threshold = pipeline_package["optimal_threshold"]
        
        processed_row = transformer.transform(sanitized_input_row)
        risk_probability = float(clf_model.predict_proba(processed_row)[:, 1][0])
        predicted_class = 1 if risk_probability >= decision_threshold else 0
        status_text = "High Risk Action Flagged" if predicted_class == 1 else "Normal Risk Range"
        
        return InferenceOutputResponse(
            disease_risk_probability=round(risk_probability, 4),
            high_risk_classification=predicted_class,
            optimal_decision_threshold_applied=decision_threshold,
            clinical_status=status_text
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Inference Failure: {str(e)}")