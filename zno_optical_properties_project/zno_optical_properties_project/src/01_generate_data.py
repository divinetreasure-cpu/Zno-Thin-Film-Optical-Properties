"""
01_generate_data.py
--------------------
Generates a synthetic dataset of ZnO thin film samples across four common
deposition methods (RF magnetron sputtering, PLD, ALD, sol-gel), with
process parameters and resulting optical properties. This mimics the kind
of compiled table a systematic literature review would assemble from many
individual papers -- the same structure as the data behind "A Comprehensive
Review of the Optical Properties of Zinc Oxide Thin Films for Opto-
Electronic Applications" -- but generated here with physically-motivated
relationships (Burstein-Moss shift, defect-driven Urbach broadening,
grain-growth with temperature, film-density effects on refractive index)
since no live network access is available to pull real reported values.

Swap in real extracted literature values (same columns) to run this
pipeline on your actual review data -- see README.md.
"""

import numpy as np
import pandas as pd

np.random.seed(11)

methods = ["RF Sputtering", "PLD", "ALD", "Sol-Gel"]
n_per_method = {"RF Sputtering": 120, "PLD": 90, "ALD": 70, "Sol-Gel": 120}

rows = []
for method in methods:
    n = n_per_method[method]

    if method == "RF Sputtering":
        thickness = np.random.uniform(80, 500, n)
        substrate_temp = np.random.uniform(25, 400, n)
        o2_pressure = np.random.uniform(0.01, 0.3, n)
        precursor_molarity = np.full(n, np.nan)
        base_density_factor = 1.0  # dense films

    elif method == "PLD":
        thickness = np.random.uniform(50, 400, n)
        substrate_temp = np.random.uniform(200, 600, n)
        o2_pressure = np.random.uniform(1, 50, n)
        precursor_molarity = np.full(n, np.nan)
        base_density_factor = 1.03  # very dense, epitaxial-like

    elif method == "ALD":
        thickness = np.random.uniform(10, 200, n)
        substrate_temp = np.random.uniform(100, 300, n)
        o2_pressure = np.full(n, np.nan)  # uses precursor pulses, not O2 pressure
        precursor_molarity = np.full(n, np.nan)
        base_density_factor = 1.05  # highly conformal, low-defect

    else:  # Sol-Gel
        thickness = np.random.uniform(50, 300, n)
        substrate_temp = np.random.uniform(20, 25, n)  # spin-coated at ~RT
        o2_pressure = np.full(n, np.nan)
        precursor_molarity = np.random.uniform(0.3, 0.75, n)
        base_density_factor = 0.88  # more porous

    annealing_temp = np.random.choice(
        [0, 300, 400, 500, 600, 700], size=n,
        p=[0.15, 0.15, 0.2, 0.2, 0.2, 0.1]
    ).astype(float)

    rows.append(pd.DataFrame({
        "deposition_method": method,
        "thickness_nm": thickness,
        "substrate_temp_C": substrate_temp,
        "annealing_temp_C": annealing_temp,
        "o2_pressure_Pa": o2_pressure,
        "precursor_molarity_M": precursor_molarity,
        "density_factor": base_density_factor,
    }))

df = pd.concat(rows, ignore_index=True)
n_total = len(df)

# --- Grain size: grows with substrate + annealing temperature -------------
df["grain_size_nm"] = np.clip(
    12
    + 0.035 * df["substrate_temp_C"]
    + 0.045 * df["annealing_temp_C"]
    + np.random.normal(0, 4, n_total),
    5, 120,
).round(1)

# --- Carrier concentration (defect/oxygen-vacancy driven) -----------------
# Lower O2 pressure (or ALD/sol-gel's inherently oxygen-poor low-temp growth)
# and lower annealing -> more oxygen vacancies -> higher carrier concentration
o2_effect = df["o2_pressure_Pa"].fillna(5.0)  # missing (ALD/sol-gel) treated as moderate
log_carrier = (
    19.6
    - 0.35 * np.log10(o2_effect + 0.5)
    - 0.0009 * df["annealing_temp_C"]
    - 0.0012 * df["grain_size_nm"]
    + np.random.normal(0, 0.25, n_total)
)
df["carrier_concentration_cm3"] = (10 ** log_carrier).round(-14)

# --- Urbach energy (band-tail disorder from defects) -----------------------
df["urbach_energy_meV"] = np.clip(
    55
    + 4.5 * (log_carrier - 18.5)
    - 0.025 * df["annealing_temp_C"]
    - 0.15 * df["grain_size_nm"]
    + np.random.normal(0, 6, n_total),
    40, 220,
).round(1)

# --- Optical bandgap (eV): Burstein-Moss widening minus defect-state
# narrowing, plus a mild quantum-confinement term for very thin films -------
burstein_moss = 0.045 * (log_carrier - 18.0).clip(lower=0) ** 1.4
defect_narrowing = 0.003 * (df["urbach_energy_meV"] - 60).clip(lower=0)
quantum_confinement = 8.0 / (df["thickness_nm"] + 10) ** 1.6  # only matters when thin

df["bandgap_eV"] = np.clip(
    3.30
    + burstein_moss
    - defect_narrowing
    + quantum_confinement
    + np.random.normal(0, 0.02, n_total),
    3.15, 3.45,
).round(3)

# --- Refractive index at 550 nm: denser films -> higher index; loosely
# tracks the Moss relation (higher bandgap -> slightly lower index) --------
df["refractive_index_550nm"] = np.clip(
    2.02 * df["density_factor"]
    - 0.06 * (df["bandgap_eV"] - 3.30)
    - 0.0006 * df["thickness_nm"].clip(upper=200)
    + np.random.normal(0, 0.02, n_total),
    1.55, 2.15,
).round(3)

# --- Extinction coefficient at 550 nm: rises with disorder/carriers --------
df["extinction_coefficient_550nm"] = np.clip(
    0.0015
    + 0.00004 * (df["urbach_energy_meV"] - 50)
    + 1e-22 * df["carrier_concentration_cm3"]
    + np.abs(np.random.normal(0, 0.002, n_total)),
    0.0001, 0.05,
).round(5)

# --- Average visible-range transmittance (%) --------------------------------
df["avg_transmittance_pct"] = np.clip(
    93
    - 0.006 * df["thickness_nm"]
    - 220 * df["extinction_coefficient_550nm"]
    - 0.03 * (df["urbach_energy_meV"] - 50)
    + 0.004 * df["annealing_temp_C"]
    + np.random.normal(0, 1.5, n_total),
    55, 97,
).round(1)

df = df.drop(columns=["density_factor"])

out_path = "/home/claude/zno_optical_properties_project/data/zno_thin_films.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["deposition_method"].value_counts())
print(df.head())
