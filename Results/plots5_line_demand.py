# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 04:48:01 2025

@author: ucbvplu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load input file ---
df_in = pd.read_csv("OSEMOSYS_Energy_Input.csv", low_memory=False)

# --- Filter relevant commodities ---
commodities = ["COMELC", "RESELC", "INDELC"]
df_demand = (
    df_in.loc[df_in["COMMODITY"].isin(commodities), ["Future.ID", "YEAR", "SpecifiedAnnualDemand"]]
        .assign(SpecifiedAnnualDemand=lambda d: pd.to_numeric(d["SpecifiedAnnualDemand"], errors="coerce"))
        .dropna(subset=["SpecifiedAnnualDemand"])
)

# Remove zeros
df_demand = df_demand.loc[df_demand["SpecifiedAnnualDemand"] != 0]

# --- Aggregate demand per year and future ---
df_demand_sum = (
    df_demand.groupby(["Future.ID", "YEAR"], as_index=False)["SpecifiedAnnualDemand"].sum()
              .rename(columns={"SpecifiedAnnualDemand": "TotalDemand"})
)
# Convert PJ to TWh
df_demand_sum["TotalDemand_TWh"] = df_demand_sum["TotalDemand"] * 0.27778

# --- Plot ---
plt.figure(figsize=(10,6))

# Plot all futures in light grey
for fid, grp in df_demand_sum.groupby("Future.ID"):
    if fid == 0:
        continue
    plt.plot(grp["YEAR"], grp["TotalDemand_TWh"], color="lightgrey", linewidth=1, alpha=0.7)

# Highlight Future.ID = 0 in blue
df0 = df_demand_sum.loc[df_demand_sum["Future.ID"] == 0]
plt.plot(df0["YEAR"], df0["TotalDemand_TWh"], color="blue", linewidth=2, label="Scenario 0")

plt.title("Total annual electricity demand")
plt.ylabel("Demand [TWh]")
plt.legend()
plt.tight_layout()
plt.show()