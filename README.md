# clinical-risk-stratification-api
A production-grade Clinical Decision Support System (CDSS) that sanitizes unstructured EHR data, trains a threshold-optimized XGBoost pipeline for disease risk prediction, and serves live insights via a FastAPI backend and Tailwind CSS dashboard.
# 🏥 Clinical Risk Stratification Engine & CDSS Dashboard

An end-to-end, production-grade **Clinical Decision Support System (CDSS)** designed to ingest highly unstructured, messy Electronic Health Record (EHR) entries, execute automated data sanitization, train a threshold-optimized machine learning pipeline, and serve live risk predictions via an interactive web interface.

---

## 🔍 The Problem & Project Discovery

### The Core Challenge
Medical data trapped in Electronic Health Records is notoriously messy. Busy healthcare workers input attributes in non-standard formats (e.g., ages written as words like `"forty"`, blood pressures recorded as `"134 over 82"`, and BMIs containing text units like `"28.4 kg/m²"`). Traditional software fails to parse these patterns, resulting in lost predictive signals. Furthermore, standard AI models suffer from **class imbalance** (healthy logs vastly outnumber disease cases) and default to predicting "healthy," dangerously missing high-risk, false-negative outliers.

### Key Discoveries
* **Hidden Bio-Signals:** Utilizing game-theoretic **SHAP values**, the pipeline discovered that derived features like **Pulse Pressure** ($Systolic - Diastolic$) and **Medication Density** (the count of distinct drugs prescribed) hold stronger predictive weight than raw individual metrics.
* **The 0.38 Calibrated Boundary:** Tuning the classification threshold for a clinical **$F_2$-Score** (prioritizing recall to catch sick patients over precision) revealed that a standard `0.50` decision threshold is clinically unsafe. Dropping the decision boundary down to **`0.38`** significantly reduces false negatives without overflowing the clinical workflow.
* **Algorithmic Fairness:** A rigorous multi-panel group parity audit proved that model performance remains stable ($\Delta AUC < 0.02$) across all binned age blocks and gender cohorts, establishing demographic fairness.

---

## 🛠️ System Architecture

The pipeline operates as a modular data manufacturing assembly line:

1.  **Ingestion & Regex Sanitization Layer:** Strips raw units, handles word-to-number mapping, extracts structural metrics via regular expressions, and limits variables to valid human physiological ranges.
2.  **Feature Pipeline Vectorization:** Computes downstream clinical markers (Pulse Pressure, Medication Count), imputes missing values using localized medians, and dynamically encodes high-cardinality medical billing targets (`Diagnosis_Code`).
3.  **XGBoost Optimization Engine:** Trains an ensemble classifier using **Stratified 5-Fold Cross-Validation** to neutralize data leakage and leverages `scale_pos_weight` to counter severe class imbalances.
4.  **Production API & GUI:** Serves inference through a structured, Pydantic-validated **FastAPI** backend linked to a responsive **Tailwind CSS frontend** browser window.

---

## 🚀 Getting Started & Installation

### Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 1. Clone the Repository & Install Dependencies

git clone [https://github.com/your-username/clinical-risk-stratification-api.git](https://github.com/your-username/clinical-risk-stratification-api.git)
cd clinical-risk-stratification-api
pip install fastapi uvicorn pydantic xgboost scikit-learn joblib pandas numpy matplotlib seaborn shap

### To Launch the asynchronous ASGI gateway interface to load the model into memory and bind the frontend template:


uvicorn app:app --reload

### Access the Platform

Interactive Clinical Dashboard (UI): http://127.0.0.1:8000/

Swagger API Docs & Schema Endpoint: http://127.0.0.1:8000/docs

System Infrastructure Liveness Probe: http://127.0.0.1:8000/health

### 📡 API Payload Blueprint
Live Inference Request (POST /predict)

{
  "Age": "forty",
  "Gender": "Female",
  "City": "Boston",
  "BMI": "28.4 kg/m²",
  "Blood_Pressure": "134 over 82",
  "Heart_Rate": "seventy",
  "Cholesterol_Level": "high",
  "Diabetic": "yes",
  "Smoker": "Former",
  "Medications": "Aspirin; Metformin",
  "Last_Visit_Date": "2024-11-13",
  "Follow_Up": "two weeks",
  "Diagnosis_Code": "I10"
}

CDSS Structured Response

{
  "disease_risk_probability": 0.6421,
  "high_risk_classification": 1,
  "optimal_decision_threshold_applied": 0.38,
  "clinical_status": "High Risk Action Flagged"
}

### ⚖️ Model Card & Intended Clinical Use

**Primary Purpose**: Automated triage assistance and risk stratification. This application is a clinical decision support asset and is not intended to replace professional diagnostic judgment.

**Target Population**: Adult patient matrices tracked inside domestic outpatient hospital environments.

**Limitations**: Model out-of-distribution tracking errors may amplify if passed completely unique or unseen primary billing code structures (Diagnosis_Code). Regular retraining and performance drift tracking (via tools like Evidently AI) are strongly advised.

<img width="1901" height="965" alt="image" src="https://github.com/user-attachments/assets/a0e6e647-24f2-4fe2-ab53-acc3cabf4a02" />
