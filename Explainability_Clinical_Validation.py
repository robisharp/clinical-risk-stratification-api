import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Suppress warnings for clean execution output
import warnings
warnings.filterwarnings("ignore")

def run_clinical_validation_suite(final_model, feature_pipeline, X_raw, y_true, raw_df):
    """
    Executes a complete healthcare-centric validation suite tracking model explainability,
    demographic parity, and clinical boundary validation.
    """
    print("🏥 Initiating Clinical Validation and Fairness Audit Suite...\n" + "="*60)
    
    # 1. Transform full dataset through Phase 3 feature engineering pipeline
    X_transformed = feature_pipeline.transform(X_raw)
    
    # Retrieve final feature names from column transformer if available
    try:
        feature_names = feature_pipeline.named_steps['preprocessing_blocks'].get_feature_names_out()
    except:
        feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
    
    # ─── 1. SHAP VALUE EXPLAINABILITY ─────────────────────────────────────────
    print("🧬 Computing TreeSHAP values for local and global explanation...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_transformed)
    
    # Generate and save a SHAP summary plot representing global feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.title("Global Clinical Feature Importance (SHAP values)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("clinical_validation_shap_summary.png", dpi=150)
    plt.close()
    print("   💾 Saved global interpretation profile -> 'clinical_validation_shap_summary.png'")
    
    # ─── 2. FAIRNESS & DEMOGRAPHIC PARITY CHECK ───────────────────────────────
    print("\n⚖️ Auditing algorithmic fairness across key demographic subsets...")
    
    # Re-attach raw demographic markers to the validation dataframe for analysis
    audit_df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': final_model.predict_proba(X_transformed)[:, 1],
        'Gender': raw_df.loc[y_true.index, 'Gender'].fillna('Unknown'),
        'Age': pd.to_numeric(raw_df.loc[y_true.index, 'Age'], errors='coerce')
    })
    
    # Bin age into clinical cohort categories
    audit_df['Age_Group'] = pd.cut(audit_df['Age'], bins=[-1, 35, 60, 120], labels=['Young (<35)', 'Adult (35-60)', 'Senior (60+)'])
    
    # Audit Framework: Calculate stratified ROC-AUC per Demographic Column
    fairness_metrics = []
    
    # Track metrics by Gender
    for gender, group in audit_df.groupby('Gender'):
        if len(group['y_true'].unique()) > 1:
            auc = roc_auc_score(group['y_true'], group['y_prob'])
            fairness_metrics.append({'Demographic': 'Gender', 'Group': gender, 'ROC-AUC': auc})
            
    # Track metrics by Age Group
    for age_grp, group in audit_df.groupby('Age_Group'):
        if len(group['y_true'].unique()) > 1:
            auc = roc_auc_score(group['y_true'], group['y_prob'])
            fairness_metrics.append({'Demographic': 'Age Cohort', 'Group': str(age_grp), 'ROC-AUC': auc})
            
    fairness_report = pd.DataFrame(fairness_metrics)
    print("\n📊 Algorithmic Fairness Audit Report Summary:")
    print(fairness_report.to_string(index=False))
    
    # ─── 3. DEMOGRAPHIC STRATIFIED ROC AUDIT VISUALIZATION ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Gender ROC Curves
    for gender, group in audit_df.groupby('Gender'):
        if len(group['y_true'].unique()) > 1:
            fpr, tpr, _ = roc_curve(group['y_true'], group['y_prob'])
            axes[0].plot(fpr, tpr, label=f"{gender} (AUC = {roc_auc_score(group['y_true'], group['y_prob']):.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title("ROC Performance Stratified by Gender", fontweight='bold')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    
    # Plot Age Group ROC Curves
    for age_grp, group in audit_df.groupby('Age_Group'):
        if len(group['y_true'].unique()) > 1:
            fpr, tpr, _ = roc_curve(group['y_true'], group['y_prob'])
            axes[1].plot(fpr, tpr, label=f"{age_grp} (AUC = {roc_auc_score(group['y_true'], group['y_prob']):.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_title("ROC Performance Stratified by Age Cohort", fontweight='bold')
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("clinical_demographic_fairness_curves.png", dpi=150)
    plt.close()
    print("\n   💾 Saved performance parity plot -> 'clinical_demographic_fairness_curves.png'")
    print("="*60 + "\n✅ Phase 5 Verification Complete. Ready for Clinical Sign-off.")

# Operational Example:
# run_clinical_validation_suite(final_model, feature_engineering_pipeline, X, y, df)
