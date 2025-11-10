# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 04:01:38 2025

@author: ucbvplu
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Load input file ---
df_in = pd.read_csv("OSEMOSYS_Energy_Input.csv", low_memory=False)

# --- Filter solar techs ---
df_solar = (
    df_in.loc[df_in["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False),
              ["Future.ID", "YEAR", "CapitalCost"]]
        .assign(CapitalCost=lambda d: pd.to_numeric(d["CapitalCost"], errors="coerce"))
        .dropna(subset=["CapitalCost"])
)

# Remove zeros
df_solar = df_solar.loc[df_solar["CapitalCost"] != 0]

# --- Average across solar sub-techs per Future+Year ---
df_solar_avg = (
    df_solar.groupby(["Future.ID","YEAR"], as_index=False)["CapitalCost"].mean()
             .rename(columns={"CapitalCost":"CapitalCost_Solar"})
)

# --- Plot ---
plt.figure(figsize=(10,6))

# 1. Plot all non-zero scenarios first (grey)
for fid, grp in df_solar_avg.groupby("Future.ID"):
    if fid != 0:
        plt.plot(grp["YEAR"], grp["CapitalCost_Solar"],
                 color="lightgrey", linewidth=1, alpha=0.7)

# 2. Plot Scenario 0 last (blue, in front)
df0 = df_solar_avg.loc[df_solar_avg["Future.ID"] == 0]
plt.plot(df0["YEAR"], df0["CapitalCost_Solar"],
         color="blue", linewidth=2.5, label="Scenario 0")

plt.title("Solar Capital Cost")
plt.xlabel("Year")
plt.ylabel("Cost [USD/kW]")
plt.legend()
plt.tight_layout()
plt.show()
