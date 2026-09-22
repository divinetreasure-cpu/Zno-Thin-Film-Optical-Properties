"""
03_modeling.py
----------------
Two modeling tasks on the compiled ZnO thin film dataset:

  (A) Regression: predict optical bandgap (eV) from deposition/process
      parameters -- useful for process design (target a bandgap without
      running the deposition first).
  (B) Classification: identify the likely deposition method from a film's
      measured optical signature alone -- useful when reviewing papers
      that report properties without always stating growth conditions
      clearly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = "/home/claude/zno_optical_properties_project/data/zno_thin_films.csv"
PLOTS = "/home/claude/zno_optical_properties_project/plots"

df = pd.read_csv(DATA)
results = {}

# ============================================================
# TASK A: Regression -- predict bandgap from process parameters
# ============================================================
process_num_features = [
    "thickness_nm", "substrate_temp_C", "annealing_temp_C",
    "o2_pressure_Pa", "precursor_molarity_M",
]
process_cat_features = ["deposition_method"]

X = df[process_num_features + process_cat_features]
y = df["bandgap_eV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), process_num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), process_cat_features),
])

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42
    ),
}

reg_results = {}
plt.figure(figsize=(6, 6))
for name, model in reg_models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    rmse = mean_squared_error(y_test, pred) ** 0.5
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    reg_results[name] = {"rmse_eV": round(rmse, 4), "mae_eV": round(mae, 4), "r2": round(r2, 4)}

    plt.scatter(y_test, pred, alpha=0.6, label=f"{name} (R²={r2:.3f})", s=25)

    if name == "Random Forest":
        best_reg_pipe = pipe

lims = [df["bandgap_eV"].min() - 0.01, df["bandgap_eV"].max() + 0.01]
plt.plot(lims, lims, "k--", alpha=0.5, label="Perfect prediction")
plt.xlabel("Actual Bandgap (eV)")
plt.ylabel("Predicted Bandgap (eV)")
plt.title("Bandgap Prediction: Actual vs. Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_bandgap_actual_vs_predicted.png")
plt.close()

results["bandgap_regression"] = reg_results

# --- Feature importance for bandgap prediction -----------------------------
ohe = best_reg_pipe.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(process_cat_features))
all_names = process_num_features + cat_names
importances = best_reg_pipe.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({"feature": all_names, "importance": importances}) \
    .sort_values("importance", ascending=False)

plt.figure(figsize=(7, 5))
sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
plt.title("What Drives Bandgap? — Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_bandgap_feature_importance.png")
plt.close()

# ============================================================
# TASK B: Classification -- identify deposition method from optical signature
# ============================================================
optical_features = [
    "bandgap_eV", "refractive_index_550nm", "extinction_coefficient_550nm",
    "avg_transmittance_pct", "urbach_energy_meV",
]
Xc = df[optical_features]
yc = df["deposition_method"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, stratify=yc, random_state=42
)

scaler = StandardScaler().fit(Xc_train)
Xc_train_s = scaler.transform(Xc_train)
Xc_test_s = scaler.transform(Xc_test)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42
)
clf.fit(Xc_train_s, yc_train)
yc_pred = clf.predict(Xc_test_s)

acc = accuracy_score(yc_test, yc_pred)
report = classification_report(yc_test, yc_pred, output_dict=True)
results["method_classification"] = {
    "accuracy": round(acc, 4),
    "per_class_f1": {k: round(v["f1-score"], 3) for k, v in report.items()
                      if k in df["deposition_method"].unique()},
}

labels = sorted(yc.unique())
cm = confusion_matrix(yc_test, yc_pred, labels=labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Deposition Method Classification (Accuracy={acc:.1%})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{PLOTS}/08_method_classification_confusion_matrix.png")
plt.close()

with open("/home/claude/zno_optical_properties_project/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved bandgap prediction, feature importance, and classification plots.")
