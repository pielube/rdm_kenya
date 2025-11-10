# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:10:23 2025

@author: ucbvplu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Load ---
df_in  = pd.read_csv("OSEMOSYS_Energy_Input.csv", low_memory=False)
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# X: total specified demand in 2050, summed over all entries
x_demand2050 = (
    df_in.loc[df_in["YEAR"]==2050, ["Future.ID","SpecifiedAnnualDemand"]]
         .assign(SpecifiedAnnualDemand=lambda d: pd.to_numeric(d["SpecifiedAnnualDemand"], errors="coerce"))
         .dropna(subset=["SpecifiedAnnualDemand"])
)
x_demand2050 = x_demand2050.loc[x_demand2050["SpecifiedAnnualDemand"] != 0]

x_demand2050 = (
    x_demand2050.groupby("Future.ID", as_index=False)["SpecifiedAnnualDemand"].sum()
                .rename(columns={"SpecifiedAnnualDemand":"Demand2050_Total"})
)

# Y: 2050 natural gas capacity (PWRNGS*)
y_capacity2050 = (
    df_out.loc[(df_out["YEAR"]==2050) &
               (df_out["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False)),
               ["Future.ID","TotalCapacityAnnual"]]
          .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
          .dropna(subset=["TotalCapacityAnnual"])
          .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"].sum()
          .rename(columns={"TotalCapacityAnnual":"Capacity2050_PWRNGS"})
)

scatter_df2 = pd.merge(x_demand2050, y_capacity2050, on="Future.ID", how="inner")

# Regression
X = scatter_df2["Demand2050_Total"].values.reshape(-1,1)
y = scatter_df2["Capacity2050_PWRNGS"].values
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

# Regression line
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range = model.predict(x_range)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=scatter_df2,
                x="Demand2050_Total",
                y="Capacity2050_PWRNGS",
                color="steelblue", s=80)
plt.plot(x_range, y_range, color="black", linewidth=2, label=f"Regression line (R²={r2:.2f})")
plt.title("2050 Demand vs 2050 Gas Capacity")
plt.xlabel("Total Specified Annual Demand in 2050 (excl. 0/NaN)")
plt.ylabel("Natural Gas Capacity in 2050 (PWRNGS*)")
plt.legend()
plt.tight_layout()
plt.show()
