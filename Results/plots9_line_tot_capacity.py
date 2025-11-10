# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 05:03:44 2025

@author: ucbvplu
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Load output file ---
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# --- Filter PWR* + BESS_TECH ---
df_cap = (
    df_out.loc[
        df_out["TECHNOLOGY"].astype(str).str.contains("PWR", na=False) |
        df_out["TECHNOLOGY"].astype(str).str.contains("BESS_TECH", na=False),
        ["Future.ID", "YEAR", "TotalCapacityAnnual"]
    ]
    .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
    .dropna(subset=["TotalCapacityAnnual"])
)

# Remove zeros
df_cap = df_cap.loc[df_cap["TotalCapacityAnnual"] != 0]

# --- Aggregate capacity per year + future ---
df_cap_sum = (
    df_cap.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"].sum()
           .rename(columns={"TotalCapacityAnnual": "TotalCapacity"})
)

# --- Plot ---
plt.figure(figsize=(6,6))

# Light grey lines for all scenarios except 0
for fid, grp in df_cap_sum.groupby("Future.ID"):
    if fid != 0:
        plt.plot(grp["YEAR"], grp["TotalCapacity"], color="lightgrey", linewidth=1, alpha=0.7)

# Highlight Scenario 0 (blue) on top
df0 = df_cap_sum.loc[df_cap_sum["Future.ID"] == 0]
plt.plot(df0["YEAR"], df0["TotalCapacity"], color="blue", linewidth=2.5, label="Scenario 0")

plt.title("Total Installed Capacity")
plt.xlabel("Year")
plt.ylabel("Capacity [GW]")
plt.legend()
plt.tight_layout()
plt.show()
