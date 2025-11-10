# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:47:24 2025

@author: ucbvplu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Reload outputs (if not already loaded) ---
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# X: geothermal capacity in 2050
geo2050 = (
    df_out.loc[(df_out["YEAR"]==2050) &
               (df_out["TECHNOLOGY"].astype(str).str.contains("PWRGEO", na=False)),
               ["Future.ID","TotalCapacityAnnual"]]
          .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
          .dropna()
          .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"].sum()
          .rename(columns={"TotalCapacityAnnual":"Capacity2050_PwrGeo"})
)

# Y: natural gas capacity in 2050
gas2050 = (
    df_out.loc[(df_out["YEAR"]==2050) &
               (df_out["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False)),
               ["Future.ID","TotalCapacityAnnual"]]
          .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
          .dropna()
          .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"].sum()
          .rename(columns={"TotalCapacityAnnual":"Capacity2050_PwrNgs"})
)

# Merge
scatter_df = pd.merge(geo2050, gas2050, on="Future.ID", how="inner")

# Regression
X = scatter_df["Capacity2050_PwrGeo"].values.reshape(-1,1)
y = scatter_df["Capacity2050_PwrNgs"].values
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range = model.predict(x_range)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=scatter_df,
                x="Capacity2050_PwrGeo",
                y="Capacity2050_PwrNgs",
                color="steelblue",
                s=80)
plt.plot(x_range, y_range, color="black", linewidth=2,
         label=f"Regression line (R²={r2:.2f})")
plt.title("2050 Geothermal vs Natural Gas Capacity (by Future)")
plt.xlabel("Geothermal Capacity in 2050 (PWRGEO*)")
plt.ylabel("Natural Gas Capacity in 2050 (PWRNGS*)")
plt.legend()
plt.tight_layout()
plt.show()
