"""
02_eda.py
---------
Exploratory analysis of the compiled ZnO thin film dataset: how deposition
method and process parameters relate to the key optical properties
(bandgap, refractive index, extinction coefficient, transmittance) -- the
same comparisons a systematic review makes qualitatively, done here
quantitatively.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = "/home/claude/zno_optical_properties_project/data/zno_thin_films.csv"
PLOTS = "/home/claude/zno_optical_properties_project/plots"

df = pd.read_csv(DATA)

print("=" * 60)
print("SHAPE:", df.shape)
print(df.groupby("deposition_method").size())
print("=" * 60)
print(df.describe().T)

# --- 1. Bandgap by deposition method --------------------------------------
plt.figure(figsize=(7, 5))
order = df.groupby("deposition_method")["bandgap_eV"].median().sort_values().index
sns.boxplot(data=df, x="deposition_method", y="bandgap_eV", order=order,
            palette="Set2")
plt.axhline(3.37, color="gray", linestyle="--", linewidth=1, label="Bulk ZnO (3.37 eV)")
plt.title("Optical Bandgap by Deposition Method")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_bandgap_by_method.png")
plt.close()

# --- 2. Refractive index by method (mirrors the review's bar chart) -------
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="deposition_method", y="refractive_index_550nm",
            order=order, palette="Set2", errorbar="sd")
plt.title("Refractive Index (550 nm) by Deposition Method")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_refractive_index_by_method.png")
plt.close()

# --- 3. Bandgap vs. carrier concentration (Burstein-Moss effect) -----------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="carrier_concentration_cm3", y="bandgap_eV",
                 hue="deposition_method", alpha=0.7, palette="Set2")
plt.xscale("log")
plt.title("Bandgap vs. Carrier Concentration (Burstein-Moss Widening)")
plt.xlabel("Carrier Concentration (cm$^{-3}$, log scale)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_bandgap_vs_carrier_concentration.png")
plt.close()

# --- 4. Transmittance vs thickness and annealing ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=df, x="thickness_nm", y="avg_transmittance_pct",
                 hue="deposition_method", alpha=0.7, palette="Set2", ax=axes[0])
axes[0].set_title("Transmittance vs. Film Thickness")

sns.boxplot(data=df, x="annealing_temp_C", y="avg_transmittance_pct",
            color="#8172B2", ax=axes[1])
axes[1].set_title("Transmittance vs. Annealing Temperature")
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_transmittance_relationships.png")
plt.close()

# --- 5. Correlation heatmap (numeric features) ------------------------------
numeric_cols = [
    "thickness_nm", "substrate_temp_C", "annealing_temp_C", "grain_size_nm",
    "carrier_concentration_cm3", "urbach_energy_meV", "bandgap_eV",
    "refractive_index_550nm", "extinction_coefficient_550nm",
    "avg_transmittance_pct",
]
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — Process & Optical Parameters")
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_correlation_heatmap.png")
plt.close()

print("\nSaved 5 EDA plots to", PLOTS)
