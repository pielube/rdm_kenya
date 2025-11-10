# Scatter: 2050 geothermal CapitalCost (x) vs system-wide LCOE (y)
# - One point per Future.ID
# - X from inputs: PWRGEO* CapitalCost in 2050 (mean across sub-techs), filter out 0/NaN
# - Y from outputs: system-wide LCOE in 2050 (no tech filter; mean if multiple rows)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Load ---
df_in  = pd.read_csv("OSEMOSYS_Energy_Input.csv", low_memory=False)
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# --- X: 2050 geothermal CapitalCost ---
x_geo_cost = (
    df_in.loc[
        (df_in["YEAR"] == 2050) &
        (df_in["TECHNOLOGY"].astype(str).str.contains("PWRGEO", na=False)),
        ["Future.ID", "CapitalCost"]
    ]
    .assign(CapitalCost=lambda d: pd.to_numeric(d["CapitalCost"], errors="coerce"))
    .dropna(subset=["CapitalCost"])
)
x_geo_cost = x_geo_cost.loc[x_geo_cost["CapitalCost"] != 0]
x_geo_cost = (
    x_geo_cost.groupby("Future.ID", as_index=False)["CapitalCost"].mean()
              .rename(columns={"CapitalCost": "CapitalCost2050_Geo"})
)

# --- Y: system-wide LCOE in 2050 (generic, no technology filter) ---
y_lcoe = (
    df_out.loc[df_out["YEAR"] == 2050, ["Future.ID", "LCOE"]]
          .assign(LCOE=lambda d: pd.to_numeric(d["LCOE"], errors="coerce"))
          .dropna(subset=["LCOE"])
          .groupby("Future.ID", as_index=False)["LCOE"].mean()
          .rename(columns={"LCOE": "LCOE_System2050"})
)

# --- Merge ---
scatter_df = pd.merge(x_geo_cost, y_lcoe, on="Future.ID", how="inner")
if scatter_df.empty:
    raise ValueError("No overlapping futures after filtering. Check 2050 PWRGEO costs and system LCOE availability.")

# --- Regression ---
X = scatter_df["CapitalCost2050_Geo"].to_numpy().reshape(-1, 1)
y = scatter_df["LCOE_System2050"].to_numpy()
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

# --- Line ---
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(x_range)

# --- Plot ---
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=scatter_df,
    x="CapitalCost2050_Geo",
    y="LCOE_System2050",
    color="steelblue", s=80
)
plt.plot(x_range, y_range, color="black", linewidth=2, label=f"Regression line (R²={r2:.2f})")
plt.title("2050 Geothermal Capital Cost vs System-wide LCOE (by Future)")
plt.xlabel("Geothermal Capital Cost in 2050 (PWRGEO*, excl. 0/NaN)")
plt.ylabel("System-wide LCOE in 2050")
plt.legend()
plt.tight_layout()
plt.show()
