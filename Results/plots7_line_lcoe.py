
import pandas as pd
import matplotlib.pyplot as plt

# --- Load output file ---
df_out = pd.read_csv("OSEMOSYS_Energy_Output.csv", low_memory=False)

# --- Clean LCOE data ---
df_lcoe = (
    df_out.loc[:, ["Future.ID", "YEAR", "LCOE"]]
          .assign(LCOE=lambda d: pd.to_numeric(d["LCOE"], errors="coerce"))
          .dropna(subset=["LCOE"])
)
df_lcoe = df_lcoe.loc[df_lcoe["LCOE"] != 0]

# --- Aggregate across technologies (if multiple per Future+Year) ---
df_lcoe = df_lcoe.groupby(["Future.ID", "YEAR"], as_index=False)["LCOE"].mean()

# --- Filter years 2030–2050 ---
df_lcoe = df_lcoe[(df_lcoe["YEAR"] >= 2030) & (df_lcoe["YEAR"] <= 2050)]
df_lcoe["LCOE"] = df_lcoe["LCOE"]*0.0036

# --- Plot ---
plt.figure(figsize=(10,6))

for fid, grp in df_lcoe.groupby("Future.ID"):
    if fid == 0:
        plt.plot(grp["YEAR"], grp["LCOE"], color="blue", linewidth=2, label="Scenario 0")
    else:
        plt.plot(grp["YEAR"], grp["LCOE"], color="lightgrey", linewidth=1, alpha=0.7)


import matplotlib.ticker as mticker

# Force integer ticks, every 5 years
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

plt.title("System-wide LCOE by Scenario (2030–2050)")
plt.xlabel("Year")
plt.ylabel("LCOE [USD/kWh]")
plt.legend()
plt.tight_layout()
plt.show()
