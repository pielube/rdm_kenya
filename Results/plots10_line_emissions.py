
import pandas as pd
import matplotlib.pyplot as plt

# --- Load output file ---
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# --- Select AnnualTechnologyEmission, CO2 only ---
df_em = (
    df_out.loc[df_out.get("EMISSION") == "CO2", 
               ["Future.ID", "YEAR", "AnnualTechnologyEmission"]]
          .assign(AnnualTechnologyEmission=lambda d: pd.to_numeric(d["AnnualTechnologyEmission"], errors="coerce"))
          .dropna(subset=["AnnualTechnologyEmission"])
)

# Remove zeros
df_em = df_em.loc[df_em["AnnualTechnologyEmission"] != 0]

# --- Aggregate over technologies ---
df_em_sum = (
    df_em.groupby(["Future.ID", "YEAR"], as_index=False)["AnnualTechnologyEmission"].sum()
          .rename(columns={"AnnualTechnologyEmission":"CO2_Emissions"})
)

# --- Plot ---
plt.figure(figsize=(10,6))

# Other scenarios in grey
for fid, grp in df_em_sum.groupby("Future.ID"):
    if fid != 0:
        plt.plot(grp["YEAR"], grp["CO2_Emissions"], color="lightgrey", linewidth=1, alpha=0.7)

# Scenario 0 in blue, drawn on top
df0 = df_em_sum.loc[df_em_sum["Future.ID"] == 0]
plt.plot(df0["YEAR"], df0["CO2_Emissions"], color="blue", linewidth=2.5, label="Scenario 0")

plt.title("Annual CO₂ Emissions by Scenario")
plt.xlabel("Year")
plt.ylabel("Emissions [Mt CO2]")
plt.legend()
plt.tight_layout()
plt.show()
