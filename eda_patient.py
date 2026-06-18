"""
EDA — Patient Health Records
Run: python eda_patient_health.py
Outputs: eda_report/ directory with PNG plots + console summary
"""

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "Patient_Health_Records_Raw.csv"
OUT_DIR   = Path("eda_report")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
COLORS = {"clean": "#4C9BE8", "dirty": "#E8694C", "missing": "#B0AEAD",
          "pos": "#E8694C", "neg": "#4C9BE8"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)
print(f"\n{'='*60}")
print(f"  Dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIGHTWEIGHT CLEANING  (only enough to enable numeric EDA)
# ─────────────────────────────────────────────────────────────────────────────

def parse_age(val):
    word_map = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                "sixty": 60, "seventy": 70, "eighty": 80}
    try:
        v = float(val)
        return v if 0 <= v <= 120 else np.nan
    except (TypeError, ValueError):
        return word_map.get(str(val).strip().lower(), np.nan)

def parse_bmi(val):
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except (TypeError, ValueError):
        return np.nan

def parse_heart_rate(val):
    word_map = {"eighty": 80, "sixty": 60, "seventy": 70, "ninety": 90}
    try:
        v = float(val)
        return v if 30 <= v <= 250 else np.nan
    except (TypeError, ValueError):
        return word_map.get(str(val).strip().lower(), np.nan)

def parse_bp_systolic(val):
    """Extract systolic from '120/80', '120 - 80', '120 over 80'."""
    if pd.isna(val):
        return np.nan
    m = re.search(r"(\d{2,3})\s*(?:/|-|over)\s*\d{2,3}", str(val))
    return float(m.group(1)) if m else np.nan

def normalise_gender(val):
    if pd.isna(val): return "Unknown"
    v = str(val).strip().lower()
    if v in ("f", "female"): return "Female"
    if v in ("m", "male"):   return "Male"
    return "Unknown"

def normalise_binary(val, yes_tokens=("yes","y","1","true")):
    if pd.isna(val): return np.nan
    return 1 if str(val).strip().lower() in yes_tokens else 0

def parse_cholesterol(val):
    mapping = {"normal": 180, "high": 260}   # rough midpoint proxies
    try:
        return float(val)
    except (TypeError, ValueError):
        return mapping.get(str(val).strip().lower(), np.nan)

def parse_target(val):
    try:
        v = float(val)
        return int(v) if v in (0, 1) else np.nan
    except (TypeError, ValueError):
        return np.nan  # "unknown" → NaN
    
def parse_follow_up(val):
    word_map = {
        "one week": 7, 
        "two weeks": 14, 
        "three weeks": 21, 
        "four weeks": 28, 
        "a month": 30
    }
    if pd.isna(val):
        return np.nan
    v_str = str(val).strip().lower()
    if v_str in word_map:
        return float(word_map[v_str])
    try:
        return float(v_str)
    except (TypeError, ValueError):
        return np.nan

df = df_raw.copy()
df["Age"]               = df["Age"].apply(parse_age)
df["BMI"]               = df["BMI"].apply(parse_bmi)
df["Heart_Rate"]        = df["Heart_Rate"].apply(parse_heart_rate)
df["BP_Systolic"]       = df["Blood_Pressure"].apply(parse_bp_systolic)
df["Gender_clean"]      = df["Gender"].apply(normalise_gender)
df["Diabetic_bin"]      = df["Diabetic"].apply(normalise_binary, yes_tokens=("yes","y","1"))
df["Smoker_bin"]        = df["Smoker"].apply(
    lambda v: 0 if pd.isna(v) else
    (1 if str(v).strip().lower() in ("yes","y","1") else
    (0.5 if str(v).strip().lower() in ("former","ex-smoker") else 0)))
df["Cholesterol_num"]   = df["Cholesterol_Level"].apply(parse_cholesterol)
df["Has_Disease_clean"] = df["Has_Disease"].apply(parse_target)

NUMERIC_COLS   = ["Age","BMI","Heart_Rate","BP_Systolic","Cholesterol_num","Follow_Up"]
TARGET         = "Has_Disease_clean"
CATEGORICAL    = ["Gender_clean","Diabetic_bin","Smoker_bin"]

# ... (existing cleaning steps)
df["Cholesterol_num"]   = df["Cholesterol_Level"].apply(parse_cholesterol)
df["Has_Disease_clean"] = df["Has_Disease"].apply(parse_target)

# ADD THIS LINE:
df["Follow_Up"]         = df["Follow_Up"].apply(parse_follow_up) 

NUMERIC_COLS   = ["Age","BMI","Heart_Rate","BP_Systolic","Cholesterol_num","Follow_Up"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. SECTION A — MISSING VALUE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section A: Missing values")

miss_pct = (df_raw.isnull().mean() * 100).sort_values(ascending=False)
miss_also_unknown = miss_pct.copy()
for col in ["Has_Disease","Gender","Diabetic","Smoker"]:
    extra = (df_raw[col].astype(str).str.strip().str.lower() == "unknown").mean() * 100
    miss_also_unknown[col] = miss_also_unknown.get(col, 0) + extra

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(miss_also_unknown.index, miss_also_unknown.values,
               color=[COLORS["dirty"] if v > 20 else COLORS["missing"] for v in miss_also_unknown.values])
ax.axvline(20, color="gray", linestyle="--", linewidth=0.8, label="20% threshold")
for bar, v in zip(bars, miss_also_unknown.values):
    ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
            f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Missing / ambiguous (%)")
ax.set_title("Missing & ambiguous values by column (including 'unknown' strings)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "A_missing_values.png", dpi=150)
plt.close()
print(f"   Saved → {OUT_DIR}/A_missing_values.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SECTION B — DATA QUALITY ISSUES CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section B: Data quality catalogue")

issues = {
    "Age: impossible (< 0 or > 120)":    (df_raw["Age"].apply(lambda v: not str(v).replace("-","").isdigit() or float(v) < 0 or float(v) > 120)).sum(),
    "Age: text values":                   df_raw["Age"].apply(lambda v: not str(v).lstrip("-").isdigit()).sum(),
    "BMI: has unit string (e.g. kg/m²)": df_raw["BMI"].astype(str).str.contains(r"[a-zA-Z/]", na=False).sum(),
    "Heart Rate: out of range / text":   df_raw["Heart_Rate"].apply(lambda v: not str(v).isdigit() or int(str(v)) < 30 or int(str(v)) > 250 if str(v).isdigit() else True).sum(),
    "Blood Pressure: non-standard format": df_raw["Blood_Pressure"].astype(str).str.contains(r"over|-", na=False).sum(),
    "Gender: inconsistent casing/abbrev": df_raw["Gender"].astype(str).str.lower().isin(["f","m","female","male","unknown"]).sum(),
    "Has_Disease: 'unknown' string":      (df_raw["Has_Disease"].astype(str).str.strip() == "unknown").sum(),
    "Dates: mixed formats":               df_raw["Last_Visit_Date"].astype(str).str.contains(r"[A-Za-z]", na=False).sum(),
}

fig, ax = plt.subplots(figsize=(10, 5))
names = list(issues.keys())
vals  = list(issues.values())
colors = [COLORS["dirty"] if v > 200 else COLORS["missing"] for v in vals]
bars = ax.barh(names, vals, color=colors)
for bar, v in zip(bars, vals):
    ax.text(v + 10, bar.get_y() + bar.get_height()/2,
            f"{v:,}", va="center", fontsize=9)
ax.set_xlabel("Row count affected")
ax.set_title("Data quality issues catalogue", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "B_quality_issues.png", dpi=150)
plt.close()
print(f"   Saved → {OUT_DIR}/B_quality_issues.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SECTION C — NUMERIC DISTRIBUTIONS (clean values only)
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section C: Numeric distributions")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
labels = {"Age":"Age (years)","BMI":"BMI","Heart_Rate":"Heart Rate (bpm)",
          "BP_Systolic":"Systolic BP (mmHg)","Cholesterol_num":"Cholesterol","Follow_Up":"Follow-up (days)"}

for ax, col in zip(axes, NUMERIC_COLS):
    data = df[col].dropna()
    ax.hist(data, bins=30, color=COLORS["clean"], edgecolor="white", linewidth=0.4)
    ax.axvline(data.median(), color=COLORS["dirty"], linewidth=1.5, linestyle="--",
               label=f"Median {data.median():.1f}")
    ax.set_title(labels.get(col, col), fontweight="bold")
    ax.set_xlabel(labels.get(col, col))
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    # annotate n
    ax.text(0.97, 0.95, f"n={len(data):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray")

plt.suptitle("Numeric feature distributions (after light cleaning)", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "C_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {OUT_DIR}/C_numeric_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SECTION D — TARGET VARIABLE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section D: Target variable")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Raw distribution (includes "unknown")
raw_counts = df_raw["Has_Disease"].astype(str).str.strip().value_counts()
colors_raw = [COLORS["pos"] if k=="1" else COLORS["neg"] if k=="0" else COLORS["missing"] for k in raw_counts.index]
axes[0].bar(raw_counts.index, raw_counts.values, color=colors_raw, edgecolor="white")
axes[0].set_title("Raw Has_Disease (including 'unknown')", fontweight="bold")
axes[0].set_ylabel("Count")
for i, (k, v) in enumerate(raw_counts.items()):
    axes[0].text(i, v + 30, str(v), ha="center", fontsize=9)

# After cleaning (only 0/1 rows)
clean_counts = df[TARGET].dropna().value_counts().sort_index()
axes[1].bar(["No disease (0)","Has disease (1)"], clean_counts.values,
            color=[COLORS["neg"], COLORS["pos"]], edgecolor="white")
axes[1].set_title("Cleaned target — class balance", fontweight="bold")
axes[1].set_ylabel("Count")
ratio = clean_counts[1] / clean_counts[0]
axes[1].text(0.5, 0.92, f"Class ratio  0:{1/ratio:.2f}  1:{1:.2f}",
             transform=axes[1].transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", linewidth=0.5))
for i, v in enumerate(clean_counts.values):
    axes[1].text(i, v + 30, str(v), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "D_target_distribution.png", dpi=150)
plt.close()
print(f"   Saved → {OUT_DIR}/D_target_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SECTION E — FEATURE vs TARGET (box plots)
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section E: Feature vs target")

df_labelled = df[df[TARGET].notna()].copy()
df_labelled[TARGET] = df_labelled[TARGET].astype(int)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, col in zip(axes, NUMERIC_COLS):
    groups = [df_labelled[df_labelled[TARGET]==0][col].dropna(),
              df_labelled[df_labelled[TARGET]==1][col].dropna()]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(COLORS["neg"])
    bp["boxes"][1].set_facecolor(COLORS["pos"])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No disease", "Has disease"])
    ax.set_title(labels.get(col, col), fontweight="bold")

plt.suptitle("Feature distributions split by target (cleaned data)", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "E_feature_vs_target.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {OUT_DIR}/E_feature_vs_target.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SECTION F — CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section F: Correlation heatmap")

corr_cols = NUMERIC_COLS + [TARGET]
corr_data = df_labelled[corr_cols].dropna(how="all")
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.4, ax=ax,
            annot_kws={"size": 9})
ax.set_title("Correlation matrix (numeric features + target)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "F_correlation_heatmap.png", dpi=150)
plt.close()
print(f"   Saved → {OUT_DIR}/F_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SECTION G — CATEGORICAL FEATURES vs TARGET
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section G: Categorical features vs target")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Gender
gender_rates = (df_labelled.groupby("Gender_clean")[TARGET].mean() * 100).sort_values()
axes[0].barh(gender_rates.index, gender_rates.values, color=COLORS["pos"])
axes[0].set_xlabel("Disease rate (%)")
axes[0].set_title("Disease rate by gender", fontweight="bold")
for i, v in enumerate(gender_rates.values):
    axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

# Diabetic
diab_rates = df_labelled.groupby("Diabetic_bin")[TARGET].mean() * 100
diab_rates.index = ["Not diabetic" if i==0 else "Diabetic" for i in diab_rates.index]
axes[1].bar(diab_rates.index, diab_rates.values, color=[COLORS["neg"], COLORS["pos"]])
axes[1].set_ylabel("Disease rate (%)")
axes[1].set_title("Disease rate by diabetic status", fontweight="bold")
for i, v in enumerate(diab_rates.values):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

# Smoker
smoker_rates = df_labelled.groupby("Smoker_bin")[TARGET].mean() * 100
smoker_rates.index = ["Non-smoker","Former smoker","Smoker"]
axes[2].bar(smoker_rates.index, smoker_rates.values,
            color=[COLORS["neg"], COLORS["missing"], COLORS["pos"]])
axes[2].set_ylabel("Disease rate (%)")
axes[2].set_title("Disease rate by smoking status", fontweight="bold")
for i, v in enumerate(smoker_rates.values):
    axes[2].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "G_categorical_vs_target.png", dpi=150)
plt.close()
print(f"   Saved → {OUT_DIR}/G_categorical_vs_target.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SECTION H — OUTLIER SUMMARY (IQR method)
# ─────────────────────────────────────────────────────────────────────────────
print("▶ Section H: Outlier summary")

outlier_counts = {}
for col in NUMERIC_COLS:
    series = df[col].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    outliers = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
    outlier_counts[col] = outliers

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(outlier_counts.keys(), outlier_counts.values(),
       color=[COLORS["dirty"] if v > 100 else COLORS["missing"] for v in outlier_counts.values()])
ax.set_ylabel("Outlier count (IQR method)")
ax.set_title("Outliers per numeric feature (after light cleaning)", fontweight="bold")
for i, v in enumerate(outlier_counts.values()):
    ax.text(i, v + 2, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "H_outliers.png", dpi=150)
plt.close()
print(f"   Saved → {OUT_DIR}/H_outliers.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  EDA SUMMARY")
print(f"{'='*60}")
print(f"  Total rows            : {len(df):,}")
print(f"  Labelled rows (0/1)   : {df[TARGET].notna().sum():,}  ({df[TARGET].notna().mean()*100:.1f}%)")
print(f"  Disease prevalence    : {df_labelled[TARGET].mean()*100:.1f}%")
print()
for col in NUMERIC_COLS:
    s = df[col].dropna()
    print(f"  {col:<22} mean={s.mean():.1f}  std={s.std():.1f}  "
          f"missing={df[col].isna().mean()*100:.1f}%")
print()
print(f"  Plots saved to: ./{OUT_DIR}/")
print(f"    A_missing_values.png")
print(f"    B_quality_issues.png")
print(f"    C_numeric_distributions.png")
print(f"    D_target_distribution.png")
print(f"    E_feature_vs_target.png")
print(f"    F_correlation_heatmap.png")
print(f"    G_categorical_vs_target.png")
print(f"    H_outliers.png")
print(f"{'='*60}\n")