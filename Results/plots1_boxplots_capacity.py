# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:56:43 2025

@author: ucbvplu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1) Load data ---
CSV_PATH = "OSEMOSYS_Energy_Output.csv"  # adjust if needed
df = pd.read_csv(CSV_PATH, low_memory=False)

# --- 2) Helper function ---
def plot_capacity_boxplot_for_prefix(
    df: pd.DataFrame,
    tech_prefix: str,
    value_col: str = "TotalCapacityAnnual",
    year_col: str = "YEAR",
    future_col: str = "Future.ID",
    title: str | None = None,
    save: bool = False,
    out_dir: str = "/mnt/data",
    figsize=(12,6),
):
    """
    Build a per-year seaborn boxplot of summed capacity across futures 
    for a given technology prefix (substring match).
    """
    # Filter technologies
    mask = df["TECHNOLOGY"].astype(str).str.contains(tech_prefix, na=False)
    subset = df.loc[mask, [year_col, future_col, value_col]].copy()

    if subset.empty:
        print(f"[{tech_prefix}] No rows matched. Skipping.")
        return

    # Ensure numeric
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna(subset=[value_col])
    if subset.empty:
        print(f"[{tech_prefix}] All values NaN. Skipping.")
        return

    # Sum across sub-technologies
    grouped = (
        subset.groupby([year_col, future_col], as_index=False)[value_col]
        .sum()
    )

    # --- Plot ---
    plt.figure(figsize=figsize)
    sns.boxplot(data=grouped, x=year_col, y=value_col, color="skyblue")
    plt.title(title or f"{tech_prefix} Capacity across Futures")
    plt.xlabel("Year")
    plt.ylabel(f"{value_col} (summed across sub-techs)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save:
        out_path = f"{out_dir}/{tech_prefix}_capacity_boxplot.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()

# # --- 3) Technologies to plot ---
# tech_prefixes = [
#     "PWRNGS",  # Natural gas
#     "PWRSOL",  # Solar
#     "PWRWND",  # Wind
#     "PWRGEO",  # Geothermal
#     "BESS_TECH", "PWRBIO", "PWRCOA", 
#     "PWRHFO", "PWRHYD", "PWRPHS"
# ]

# # --- 4) Generate plots ---
# for prefix in tech_prefixes:
#     plot_capacity_boxplot_for_prefix(df, prefix)


# --- 4b) All plots in one figure ---

techs = ["PWRNGS", "PWRSOL", "PWRWND", "PWRGEO", 
         "BESS_TECH", "PWRBIO", 
         "PWRHFO", "PWRHYD", "PWRPHS", "PWRURN"]


# Build aggregated dataset: one row per (Future.ID, YEAR, prefix)
agg_list = []
for prefix in techs:
    subset = df[df["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)].copy()
    if subset.empty:
        continue
    grouped = (
        subset.groupby(["Future.ID","YEAR"], as_index=False)["TotalCapacityAnnual"]
              .sum()
              .assign(TechGroup=prefix)
    )
    agg_list.append(grouped)

df_all = pd.concat(agg_list, ignore_index=True)

import matplotlib.ticker as mticker

# Ensure categorical order is fixed
df_all["YEAR"] = df_all["YEAR"].astype(str)
year_order = sorted(df_all["YEAR"].unique(), key=int)

# Build plot with explicit year order
g = sns.catplot(
    data=df_all,
    x="YEAR", y="TotalCapacityAnnual",
    col="TechGroup", col_wrap=3,
    kind="box", sharey=False,
    height=3.5, aspect=1.2,
    order=year_order
)

g.set_titles("{col_name}")
g.set_axis_labels("Year", "Total Capacity Annual")

###
# Find global min and max across all panels
ymin, ymax = 0,18# df_all["TotalCapacityAnnual"].min(), df_all["TotalCapacityAnnual"].max()
for ax in g.axes.flatten():
    ax.set_ylim(ymin, ymax)
###

# Indices to keep: every 5 years from 2020
years_num = list(map(int, year_order))
keep_idx  = [i for i, y in enumerate(years_num) if y >= 2020 and (y - 2020) % 5 == 0]
keep_labs = [str(years_num[i]) for i in keep_idx]

for ax in g.axes.flatten():
    ax.xaxis.set_major_locator(mticker.FixedLocator(keep_idx))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(keep_labs))

plt.subplots_adjust(top=0.9)
g.fig.suptitle("Capacity Across Futures (Boxplots by Technology)")
plt.show()