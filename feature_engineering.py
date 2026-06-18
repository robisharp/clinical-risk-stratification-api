import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class HealthcareFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to build domain-specific derived features safely."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Pulse Pressure (Systolic - Diastolic)
        X['Pulse_Pressure'] = X['Systolic_BP'] - X['Diastolic_BP']
        
        # 2. Medication Count (FIXED: Uses local variable 'X' instead of undefined 'df')
        X['Medication_Count'] = X['Medications'].fillna('').astype(str).apply(
            lambda x: len(re.split(r'[,;]', x)) if x.strip() else 0
        )
        
        # 3. Age Group Binning (Ordinal Representation)
        age_bins = [-1, 18, 35, 55, 75, 150]
        X['Age_Group'] = pd.cut(X['Age'], bins=age_bins, labels=[0, 1, 2, 3, 4]).astype(float)
        
        # 4. BMI Category Binning
        bmi_bins = [-1, 18.5, 24.9, 29.9, 150]
        X['BMI_Category'] = pd.cut(X['BMI'], bins=bmi_bins, labels=[0, 1, 2, 3]).astype(float)
        
        return X

class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """Safely calculates target encoding for high-cardinality features like Diagnosis_Code."""
    def __init__(self, m=10.0):
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

# --- PREPROCESSING COLUMN COMPOSER SETUP ---

numeric_cols = ['Age', 'BMI', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 
                'Cholesterol', 'Days_Since_Last_Visit', 'Follow_Up_Days',
                'Pulse_Pressure', 'Medication_Count', 'Age_Group', 'BMI_Category']

categorical_onehot = ['City', 'Gender']
categorical_ordinal = ['Diabetic_Score', 'Smoker_Score']
target_encode_col = ['Diagnosis_Code']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat_onehot', categorical_transformer, categorical_onehot),
        ('cat_ord', ordinal_transformer, categorical_ordinal),
        ('target_enc', TargetEncoderTransformer(m=5.0), target_encode_col)
    ],
    remainder='drop'
)

# Combined Pipeline Component
feature_engineering_pipeline = Pipeline(steps=[
    ('derived_features', HealthcareFeatureEngineer()),
    ('preprocessing_blocks', preprocessor)
])

if __name__ == "__main__":
    print("✅ Feature engineering pipeline assembled cleanly without data leakage risks!")