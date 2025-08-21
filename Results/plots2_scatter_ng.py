import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Load ---
df_in  = pd.read_csv("OSEMOSYS_Energy_Input.csv", low_memory=False)
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# X: average variable cost of IMPNGS (exclude NaN and zeros)
x_varcost = (
    df_in.loc[df_in["TECHNOLOGY"]=="IMPNGS", ["Future.ID","VariableCost"]]
         .assign(VariableCost=lambda d: pd.to_numeric(d["VariableCost"], errors="coerce"))
         .dropna(subset=["VariableCost"])
)
x_varcost = x_varcost.loc[x_varcost["VariableCost"] != 0]   # remove zeros

x_varcost = (
    x_varcost.groupby("Future.ID", as_index=False)["VariableCost"].mean()
             .rename(columns={"VariableCost":"Avg_VariableCost_IMPNGS"})
)

# Y: natural gas capacity in 2050 (PWRNGS*, sum across sub-techs)
y_capacity2050 = (
    df_out.loc[(df_out["YEAR"]==2050) &
               (df_out["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False)),
               ["Future.ID","TotalCapacityAnnual"]]
          .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
          .dropna(subset=["TotalCapacityAnnual"])
          .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"].sum()
          .rename(columns={"TotalCapacityAnnual":"Capacity2050_PWRNGS"})
)

# Merge
scatter_df = pd.merge(x_varcost, y_capacity2050, on="Future.ID", how="inner")

# Regression
X = scatter_df["Avg_VariableCost_IMPNGS"].values.reshape(-1,1)
y = scatter_df["Capacity2050_PWRNGS"].values
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

# Line
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range = model.predict(x_range)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=scatter_df,
    x="Avg_VariableCost_IMPNGS",
    y="Capacity2050_PWRNGS",
    color="steelblue",
    s=80
)
plt.plot(x_range, y_range, color="black", linewidth=2,
         label=f"Regression line (R²={r2:.2f})")
plt.title("IMPNGS Variable Cost vs 2050 Gas Capacity")
plt.xlabel("Average Variable Cost of IMPNGS")
plt.ylabel("Natural Gas Capacity in 2050 (PWRNGS*)")
plt.legend()
plt.tight_layout()
plt.show()
