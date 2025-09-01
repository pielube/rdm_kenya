# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:02:41 2025

@author: ucbvplu
"""

# Scatter: 2050 battery CapitalCost (x) vs 2050 installed battery capacity (y)
# - One point per Future.ID
# - Costs filtered to remove 0 and NaN
# - Technology matched by substring "BESS_TECH" (all sub-techs)
# - 2050 used on both axes for temporal alignment

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Load ---
df_in  = pd.read_csv("OSEMOSYS_Energy_Input.csv", low_memory=False)
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# --- X: 2050 CapitalCost for BESS_TECH (sum/mean across sub-techs per Future.ID) ---
x_cost = (
    df_in.loc[
        (df_in["YEAR"] == 2050) &
        (df_in["TECHNOLOGY"].astype(str).str.contains("BESS_TECH", na=False)),
        ["Future.ID", "CapitalCost"]
    ]
    .assign(CapitalCost=lambda d: pd.to_numeric(d["CapitalCost"], errors="coerce"))
    .dropna(subset=["CapitalCost"])
)
x_cost = x_cost.loc[x_cost["CapitalCost"] != 0]  # remove zeros

# average across sub-techs if multiple rows per future
x_cost = (
    x_cost.groupby("Future.ID", as_index=False)["CapitalCost"].mean()
          .rename(columns={"CapitalCost": "CapitalCost2050_BESS"})
)

# --- Y: 2050 installed capacity for BESS_TECH (sum across sub-techs) ---
y_cap = (
    df_out.loc[
        (df_out["YEAR"] == 2050) &
        (df_out["TECHNOLOGY"].astype(str).str.contains("BESS_TECH", na=False)),
        ["Future.ID", "TotalCapacityAnnual"]
    ]
    .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
    .dropna(subset=["TotalCapacityAnnual"])
    .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"].sum()
    .rename(columns={"TotalCapacityAnnual": "Capacity2050_BESS"})
)

# --- Merge ---
scatter_df = pd.merge(x_cost, y_cap, on="Future.ID", how="inner")

# --- Guard: if empty after filters, stop gracefully ---
if scatter_df.empty:
    raise ValueError(
        "No matching futures after filtering. "
        "Check that BESS_TECH has non-zero 2050 CapitalCost and capacity."
    )

# --- Regression ---
X = scatter_df["CapitalCost2050_BESS"].to_numpy().reshape(-1, 1)
y = scatter_df["Capacity2050_BESS"].to_numpy()
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

# Line
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(x_range)

# --- Plot ---
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=scatter_df,
    x="CapitalCost2050_BESS",
    y="Capacity2050_BESS",
    color="steelblue",
    s=80
)
plt.plot(x_range, y_range, color="black", linewidth=2, label=f"Regression line (R²={r2:.2f})")
plt.title("2050 Battery Capital Cost vs 2050 Battery Capacity (by Future)")
plt.xlabel("Battery Capital Cost in 2050 (BESS_TECH, excl. 0/NaN)")
plt.ylabel("Battery Capacity in 2050 (BESS_TECH)")
plt.legend()
plt.tight_layout()
plt.show()
