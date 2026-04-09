# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 15:45:25 2025

@author: ucbvplu
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# -------------------------------------------------------------------
# Config: paths to input/output CSVs
# -------------------------------------------------------------------
CSV_INPUT = "OSEMOSYS_Energy_Input.csv"
CSV_OUTPUT = "OSEMOSYS_Energy_Output.csv"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------------------------------------------------------
# 1. Boxplots: capacity by tech
# -------------------------------------------------------------------
# def plot_boxplots_capacity(key, df_in=None, df_out=None, save=False):
#     df = df_out

#     def plot_capacity_boxplot_for_prefix(
#         df: pd.DataFrame,
#         tech_prefix: str,
#         value_col: str = "TotalCapacityAnnual",
#         year_col: str = "YEAR",
#         future_col: str = "Future.ID",
#         title: str | None = None,
#         save: bool = False,
#         out_dir: str = ".",
#         figsize=(12, 6),
#     ):
#         # Filter technologies
#         mask = df["TECHNOLOGY"].astype(str).str.contains(tech_prefix, na=False)
#         subset = df.loc[mask, [year_col, future_col, value_col]].copy()

#         if subset.empty:
#             print(f"[{tech_prefix}] No rows matched. Skipping.")
#             return

#         subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
#         subset = subset.dropna(subset=[value_col])
#         if subset.empty:
#             print(f"[{tech_prefix}] All values NaN. Skipping.")
#             return

#         grouped = (
#             subset.groupby([year_col, future_col], as_index=False)[value_col]
#                   .sum()
#         )

#         plt.figure(figsize=figsize)
#         sns.boxplot(data=grouped, x=year_col, y=value_col, color="skyblue")
#         plt.tight_layout()

#         if save:
#             out_path = f"{out_dir}/{tech_prefix}_capacity_boxplot.png"
#             plt.savefig(out_path, dpi=200, bbox_inches="tight")
#             print(f"Saved: {out_path}")

#         plt.show()

#     # Multi-panel boxplots
#     techs = [
#         "PWRSOL", "PWRWND", "BESS_TECH",
#         "PWRNGS", "PWRGEO", "PWRHYD",
#     ]

#     tech_name_map = {
#         "PWRSOL": "Solar",
#         "PWRWND": "Wind",
#         "BESS_TECH": "Battery storage",
#         "PWRNGS": "Natural gas",
#         "PWRGEO": "Geothermal",
#         "PWRHYD": "Hydro",
#     }

#     agg_list = []
#     for prefix in techs:
#         subset = df[df["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)].copy()
#         if subset.empty:
#             continue
#         grouped = (
#             subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"]
#                   .sum()
#                   .assign(TechGroup=prefix)
#         )
#         agg_list.append(grouped)

#     if not agg_list:
#         print("No technologies found for capacity boxplots.")
#         return

#     df_all = pd.concat(agg_list, ignore_index=True)

#     df_all["YEAR"] = df_all["YEAR"].astype(str)
#     year_order = sorted(df_all["YEAR"].unique(), key=int)

#     sns.set_style("whitegrid")

#     g = sns.catplot(
#         data=df_all,
#         x="YEAR", y="TotalCapacityAnnual",
#         col="TechGroup", col_wrap=3,
#         kind="box", sharey=True,
#         height=3, aspect=1.2,
#         order=year_order,
#         whis=[1, 99],
#     )

#     g.set_titles("")
#     g.set_axis_labels("Year", "Installed capacity [GW]")

#     ymin, ymax = df_all["TotalCapacityAnnual"].min()-2, df_all["TotalCapacityAnnual"].max()+2
#     ncols = 3
#     nrows = int(np.ceil(len(g.axes.flatten()) / ncols))

#     years_num = list(map(int, year_order))
#     keep_idx = [i for i, y in enumerate(years_num) if y >= 2020 and (y - 2020) % 5 == 0]
#     keep_labs = [str(years_num[i]) for i in keep_idx]

#     for i, ax in enumerate(g.axes.flatten()):
#         ax.set_ylim(ymin, ymax)

#         ax.xaxis.set_major_locator(mticker.FixedLocator(keep_idx))
#         ax.xaxis.set_major_formatter(mticker.FixedFormatter(keep_labs))

#         row = i // ncols
#         col = i % ncols
#         tech_code = g.col_names[i]

#         ax.set_title(tech_name_map.get(tech_code, tech_code), y=0.95)

#         ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.5)
#         ax.grid(axis="x", visible=False)

#         if col in [1, 2]:
#             ax.tick_params(axis="y", labelleft=False)
#             ax.set_ylabel("")

#         if row == nrows - 1:
#             ax.set_xlabel("")

#     plt.subplots_adjust(top=0.92, hspace=0.30, wspace=0.12)
#     plt.show()

#     if save:
#         out_path = os.path.join(PLOT_DIR, f"{key}.png")
#         plt.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_boxplots_capacity(key, df_in=None, df_out=None, save=False):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if df_out is None:
        raise ValueError("df_out must be provided.")

    df = df_out.copy()

    techs = [
        "PWRSOL", "PWRWND", "BESS_TECH",
        "PWRNGS", "PWRGEO", "PWRHYD",
    ]

    tech_name_map = {
        "PWRSOL": "Solar",
        "PWRWND": "Wind",
        "BESS_TECH": "Battery storage",
        "PWRNGS": "Natural gas",
        "PWRGEO": "Geothermal",
        "PWRHYD": "Hydro",
    }

    # Only plot these years
    plot_years = [2030, 2035, 2040, 2045, 2050]

    agg_list = []
    for prefix in techs:
        subset = df[df["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)].copy()
        if subset.empty:
            continue

        subset["YEAR"] = pd.to_numeric(subset["YEAR"], errors="coerce")
        subset["TotalCapacityAnnual"] = pd.to_numeric(
            subset["TotalCapacityAnnual"], errors="coerce"
        )
        subset = subset.dropna(subset=["Future.ID", "YEAR", "TotalCapacityAnnual"])
        subset = subset[subset["YEAR"].isin(plot_years)].copy()

        if subset.empty:
            continue

        grouped = (
            subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"]
            .sum()
            .assign(TechGroup=prefix)
        )
        agg_list.append(grouped)

    if not agg_list:
        print("No technologies found for capacity boxplots.")
        return

    df_all = pd.concat(agg_list, ignore_index=True)

    if df_all.empty:
        print("No valid data available for capacity boxplots.")
        return

    df_all["YEAR"] = df_all["YEAR"].astype(int).astype(str)
    year_order = [str(y) for y in plot_years]

    ymin = 0
    ymax = df_all["TotalCapacityAnnual"].max() * 1.03

    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    g = sns.catplot(
        data=df_all,
        x="YEAR",
        y="TotalCapacityAnnual",
        col="TechGroup",
        col_wrap=3,
        kind="box",
        width=0.3,
        sharey=True,
        order=year_order,
        height=1.95,
        aspect=1.18,
        whis=[1, 99],
        fliersize=1.2,
        linewidth=0.7,
        boxprops={"edgecolor": "0.25", "linewidth": 0.5},
        whiskerprops={"color": "0.35", "linewidth": 0.5},
        capprops={"color": "0.35", "linewidth": 0.5},
        medianprops={"color": "0.15", "linewidth": 0.7},
    )

    g.set_titles("")
    g.set_axis_labels("Year", "Installed capacity [GW]")

    ncols = 3

    for i, ax in enumerate(g.axes.flatten()):
        col = i % ncols
        tech_code = g.col_names[i]

        ax.set_title(tech_name_map.get(tech_code, tech_code), y=0.97, pad=1.5)
        ax.set_ylim(ymin, ymax)

        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.grid(axis="x", visible=False)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        if col in [1, 2]:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")

        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, pad=1)
        ax.tick_params(axis="y", pad=1)

    g.fig.set_size_inches(7.1, 4.35)
    g.fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.12,
        top=0.93,
        wspace=0.10,
        hspace=0.26
    )

    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        g.fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()

# -------------------------------------------------------------------
# 2. Boxplots: activity by tech
# -------------------------------------------------------------------
def plot_boxplots_activity(key, df_in=None, df_out=None, save=False):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = df_out.copy()

    techs = [
        "PWRSOL", "PWRWND", "BESS_TECH",
        "PWRNGS", "PWRGEO", "PWRHYD",
    ]

    tech_name_map = {
        "PWRSOL": "Solar",
        "PWRWND": "Wind",
        "BESS_TECH": "Battery storage",
        "PWRNGS": "Natural gas",
        "PWRGEO": "Geothermal",
        "PWRHYD": "Hydro",
    }

    # Only plot these years
    plot_years = [2030, 2035, 2040, 2045, 2050]

    agg_list = []
    for prefix in techs:
        subset = df[df["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)].copy()
        if subset.empty:
            continue

        subset["YEAR"] = pd.to_numeric(subset["YEAR"], errors="coerce")
        subset["TotalTechnologyAnnualActivity"] = pd.to_numeric(
            subset["TotalTechnologyAnnualActivity"], errors="coerce"
        )
        subset = subset.dropna(subset=["Future.ID", "YEAR", "TotalTechnologyAnnualActivity"])
        subset = subset[subset["YEAR"].isin(plot_years)].copy()

        if subset.empty:
            continue

        grouped = (
            subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalTechnologyAnnualActivity"]
            .sum()
            .assign(TechGroup=prefix)
        )
        agg_list.append(grouped)

    if not agg_list:
        print("No technologies found for activity boxplots.")
        return

    df_all = pd.concat(agg_list, ignore_index=True)

    if df_all.empty:
        print("No valid data available for activity boxplots.")
        return

    df_all["YEAR"] = df_all["YEAR"].astype(int).astype(str)
    year_order = [str(y) for y in plot_years]

    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    ymin = 0
    ymax = df_all["TotalTechnologyAnnualActivity"].max() * 1.03

    g = sns.catplot(
        data=df_all,
        x="YEAR",
        y="TotalTechnologyAnnualActivity",
        col="TechGroup",
        col_wrap=3,
        kind="box",
        sharey=True,
        order=year_order,
        height=1.95,
        aspect=1.18,
        whis=[1, 99],
        fliersize=1.2,
        linewidth=0.7,
        boxprops={"edgecolor": "0.25", "linewidth": 0.5},
        whiskerprops={"color": "0.35", "linewidth": 0.5},
        capprops={"color": "0.35", "linewidth": 0.5},
        medianprops={"color": "0.15", "linewidth": 0.7},
    )

    g.set_titles("")
    g.set_axis_labels("Year", "Annual activity [PJ]")

    ncols = 3

    for i, ax in enumerate(g.axes.flatten()):
        col = i % ncols
        tech_code = g.col_names[i]

        ax.set_title(tech_name_map.get(tech_code, tech_code), y=0.97, pad=1.5)

        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(ymin, ymax)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        if col in [1, 2]:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")

        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, pad=1)
        ax.tick_params(axis="y", pad=1)

    g.fig.set_size_inches(7.1, 6.8)
    g.fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.08,
        top=0.95,
        wspace=0.10,
        hspace=0.26
    )

    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        g.fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


# -------------------------------------------------------------------
# 3. Bar chart: 2050 gas capacity
# -------------------------------------------------------------------
def plot_bar_gas_capacity(key, df_in=None, df_out=None, save=False):
    df_in_local = df_in
    df_out_local = df_out

    x_varcost = (
        df_in_local.loc[df_in_local["TECHNOLOGY"] == "IMPNGS", ["Future.ID", "VariableCost"]]
        .assign(VariableCost=lambda d: pd.to_numeric(d["VariableCost"], errors="coerce"))
        .dropna(subset=["VariableCost"])
    )
    x_varcost = x_varcost.loc[x_varcost["VariableCost"] != 0]
    x_varcost = (
        x_varcost.groupby("Future.ID", as_index=False)["VariableCost"].mean()
                 .rename(columns={"VariableCost": "Avg_VariableCost_IMPNGS"})
    )

    y_capacity2050 = (
        df_out_local.loc[
            (df_out_local["YEAR"] == 2050)
            & (df_out_local["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False)),
            ["Future.ID", "TotalCapacityAnnual"],
        ]
        .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
        .dropna(subset=["TotalCapacityAnnual"])
        .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"].sum()
        .rename(columns={"TotalCapacityAnnual": "Capacity2050_PWRNGS"})
    )

    scatter_df = pd.merge(x_varcost, y_capacity2050, on="Future.ID", how="inner")
    if scatter_df.empty:
        print("No overlapping futures for bar_gas_capacity.")
        return

    plot_df = scatter_df.sort_values("Capacity2050_PWRNGS", ascending=False)

    plt.figure(figsize=(4, 3))
    sns.barplot(
        data=plot_df,
        x="Future.ID",
        y="Capacity2050_PWRNGS",
        order=plot_df["Future.ID"],
        color="steelblue",
        width=1,
    )
    plt.gca().set_xticklabels([])
    plt.xlabel("")
    plt.ylabel("Gas power capacity 2050 [GW]")
    plt.tight_layout()
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        
# -------------------------------------------------------------------
# 3. Bar chart: 2050 gas capacity / peak demand
# -------------------------------------------------------------------
def plot_bar_gas_capacity_ratio(key, df_in=None, df_out=None, save=False):
    df_in_local = df_in
    df_out_local = df_out

    # ---------------------------------------------------------------
    # 1. Natural gas capacity in 2050
    # ---------------------------------------------------------------
    gas2050 = (
        df_out_local.loc[
            (df_out_local["YEAR"] == 2050)
            & (df_out_local["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False)),
            ["Future.ID", "TotalCapacityAnnual"],
        ]
        .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
        .dropna(subset=["TotalCapacityAnnual"])
        .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"]
        .sum()
        .rename(columns={"TotalCapacityAnnual": "GasCapacity2050"})
    )

    # ---------------------------------------------------------------
    # 2. Peak demand in 2050
    # ---------------------------------------------------------------
    commodities = ["COMELC", "RESELC", "INDELC"]

    demand2050 = (
        df_in_local.loc[
            (df_in_local["YEAR"] == 2050)
            & (df_in_local["COMMODITY"].isin(commodities)),
            ["Future.ID", "SpecifiedAnnualDemand"],
        ]
        .assign(SpecifiedAnnualDemand=lambda d: pd.to_numeric(d["SpecifiedAnnualDemand"], errors="coerce"))
        .dropna(subset=["SpecifiedAnnualDemand"])
        .groupby("Future.ID", as_index=False)["SpecifiedAnnualDemand"]
        .sum()
    )

    # Convert annual demand to peak demand
    demand2050["PeakDemand2050"] = demand2050["SpecifiedAnnualDemand"] * (2.35 / 50.51)

    demand2050 = demand2050[["Future.ID", "PeakDemand2050"]]

    # ---------------------------------------------------------------
    # 3. Merge and compute ratio
    # ---------------------------------------------------------------
    plot_df = pd.merge(gas2050, demand2050, on="Future.ID", how="inner")

    if plot_df.empty:
        print("No overlapping futures for gas capacity ratio plot.")
        return

    plot_df["Gas_to_Peak_Ratio"] = plot_df["GasCapacity2050"] / plot_df["PeakDemand2050"]

    plot_df = plot_df.sort_values("Gas_to_Peak_Ratio", ascending=False)

    # ---------------------------------------------------------------
    # 4. Plot
    # ---------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=plot_df,
        x="Future.ID",
        y="Gas_to_Peak_Ratio",
        order=plot_df["Future.ID"],
        color="steelblue",
        width=1,
    )

    plt.gca().set_xticklabels([])
    plt.xlabel("")
    plt.ylabel("Natural Gas Capacity / Peak Demand (2050)")
    plt.tight_layout()
    plt.show()

    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


# -------------------------------------------------------------------
# 4. Scatter: BESS cost vs gas capacity
# -------------------------------------------------------------------
def plot_scatter_bess_vs_gas(key, df_in=None, df_out=None, save=False):
    df_in_local = df_in
    df_out_local = df_out

    x_cost = (
        df_in_local.loc[
            (df_in_local["YEAR"] == 2050)
            & (df_in_local["TECHNOLOGY"].astype(str).str.contains("BESS_TECH", na=False)),
            ["Future.ID", "CapitalCost"],
        ]
        .assign(CapitalCost=lambda d: pd.to_numeric(d["CapitalCost"], errors="coerce"))
        .dropna(subset=["CapitalCost"])
    )
    x_cost = x_cost.loc[x_cost["CapitalCost"] != 0]
    x_cost = (
        x_cost.groupby("Future.ID", as_index=False)["CapitalCost"].mean()
              .rename(columns={"CapitalCost": "CapitalCost2050_BESS"})
    )

    gas2050 = (
        df_out_local.loc[
            (df_out_local["YEAR"] == 2050)
            & (df_out_local["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False)),
            ["Future.ID", "TotalCapacityAnnual"],
        ]
        .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
        .dropna()
        .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"].sum()
        .rename(columns={"TotalCapacityAnnual": "Capacity2050_PwrNgs"})
    )

    scatter_df = pd.merge(x_cost, gas2050, on="Future.ID", how="inner")
    if scatter_df.empty:
        print("No overlapping futures for scatter_bess_vs_gas.")
        return

    X = scatter_df["CapitalCost2050_BESS"].to_numpy().reshape(-1, 1)
    y = scatter_df["Capacity2050_PwrNgs"].values
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))

    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=scatter_df,
        x="CapitalCost2050_BESS",
        y="Capacity2050_PwrNgs",
        color="steelblue",
        s=80,
    )
    plt.plot(x_range, y_range, color="black", linewidth=2,
             label=f"Regression line (R²={r2:.2f})")
    plt.xlabel("Battery cost in 2050")
    plt.ylabel("Natural Gas Capacity in 2050 (PWRNGS*)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


# -------------------------------------------------------------------
# 5. Line: total annual electricity demand
# -------------------------------------------------------------------
def plot_line_demand(key, df_in=None, df_out=None, save=False):
    df_in_local = df_in

    commodities = ["COMELC", "RESELC", "INDELC"]
    df_demand = (
        df_in_local.loc[df_in_local["COMMODITY"].isin(commodities),
                        ["Future.ID", "YEAR", "SpecifiedAnnualDemand"]]
        .assign(SpecifiedAnnualDemand=lambda d: pd.to_numeric(d["SpecifiedAnnualDemand"], errors="coerce"))
        .dropna(subset=["SpecifiedAnnualDemand"])
    )

    df_demand = df_demand.loc[df_demand["SpecifiedAnnualDemand"] != 0]

    df_demand_sum = (
        df_demand.groupby(["Future.ID", "YEAR"], as_index=False)["SpecifiedAnnualDemand"].sum()
                  .rename(columns={"SpecifiedAnnualDemand": "TotalDemand"})
    )

    df_demand_sum["TotalDemand_TWh"] = df_demand_sum["TotalDemand"] * 0.27778

    plt.figure(figsize=(10, 6))
    for fid, grp in df_demand_sum.groupby("Future.ID"):
        if fid == 0:
            continue
        plt.plot(grp["YEAR"], grp["TotalDemand_TWh"],
                 color="lightgrey", linewidth=1, alpha=0.7)

    df0 = df_demand_sum.loc[df_demand_sum["Future.ID"] == 0]
    if not df0.empty:
        plt.plot(df0["YEAR"], df0["TotalDemand_TWh"],
                 color="blue", linewidth=2, label="Scenario 0")

    plt.title("Total annual electricity demand")
    plt.ylabel("Demand [TWh]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


# -------------------------------------------------------------------
# 6. Line: system-wide LCOE
# -------------------------------------------------------------------
def plot_line_lcoe(key, df_in=None, df_out=None, save=False):
    df_out_local = df_out

    df_lcoe = (
        df_out_local.loc[:, ["Future.ID", "YEAR", "LCOE"]]
        .assign(LCOE=lambda d: pd.to_numeric(d["LCOE"], errors="coerce"))
        .dropna(subset=["LCOE"])
    )
    df_lcoe = df_lcoe.loc[df_lcoe["LCOE"] != 0]
    df_lcoe = df_lcoe.groupby(["Future.ID", "YEAR"], as_index=False)["LCOE"].mean()
    df_lcoe = df_lcoe[(df_lcoe["YEAR"] >= 2030) & (df_lcoe["YEAR"] <= 2050)]
    df_lcoe["LCOE"] = df_lcoe["LCOE"] * 0.0036

    plt.figure(figsize=(10, 6))
    for fid, grp in df_lcoe.groupby("Future.ID"):
        if fid == 0:
            plt.plot(grp["YEAR"], grp["LCOE"],
                     color="blue", linewidth=2, label="Scenario 0")
        else:
            plt.plot(grp["YEAR"], grp["LCOE"],
                     color="lightgrey", linewidth=1, alpha=0.7)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    plt.title("System-wide LCOE by Scenario (2030–2050)")
    plt.xlabel("Year")
    plt.ylabel("LCOE [USD/kWh]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


# -------------------------------------------------------------------
# 7. Line: gas capital cost
# -------------------------------------------------------------------
def plot_line_gas_capex(key, df_in=None, df_out=None, save=False):
    df_in_local = df_in

    df_gas = (
        df_in_local.loc[
            df_in_local["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False),
            ["Future.ID", "YEAR", "CapitalCost"],
        ]
        .assign(CapitalCost=lambda d: pd.to_numeric(d["CapitalCost"], errors="coerce"))
        .dropna(subset=["CapitalCost"])
    )
    df_gas = df_gas.loc[df_gas["CapitalCost"] != 0]

    df_gas_avg = (
        df_gas.groupby(["Future.ID", "YEAR"], as_index=False)["CapitalCost"].mean()
                .rename(columns={"CapitalCost": "CapitalCost_Solar"})
    )

    plt.figure(figsize=(10, 6))
    for fid, grp in df_gas_avg.groupby("Future.ID"):
        if fid != 0:
            plt.plot(grp["YEAR"], grp["CapitalCost_Solar"],
                     color="lightgrey", linewidth=1, alpha=0.7)

    df0 = df_gas_avg.loc[df_gas_avg["Future.ID"] == 0]
    if not df0.empty:
        plt.plot(df0["YEAR"], df0["CapitalCost_Solar"],
                 color="blue", linewidth=2.5, label="Scenario 0")

    plt.title("Import natural gas cost")  # original title, even if tech is PWRNGS
    plt.xlabel("Year")
    plt.ylabel("Cost [USD/kW]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


# -------------------------------------------------------------------
# 8. Line: total installed capacity
# -------------------------------------------------------------------
def plot_line_total_capacity(key, df_in=None, df_out=None, save=False):
    df_out_local = df_out

    df_cap = (
        df_out_local.loc[
            df_out_local["TECHNOLOGY"].astype(str).str.contains("PWR", na=False)
            | df_out_local["TECHNOLOGY"].astype(str).str.contains("BESS_TECH", na=False),
            ["Future.ID", "YEAR", "TotalCapacityAnnual"],
        ]
        .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
        .dropna(subset=["TotalCapacityAnnual"])
    )
    df_cap = df_cap.loc[df_cap["TotalCapacityAnnual"] != 0]

    df_cap_sum = (
        df_cap.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"].sum()
               .rename(columns={"TotalCapacityAnnual": "TotalCapacity"})
    )

    plt.figure(figsize=(6, 6))
    for fid, grp in df_cap_sum.groupby("Future.ID"):
        if fid != 0:
            plt.plot(grp["YEAR"], grp["TotalCapacity"],
                     color="lightgrey", linewidth=1, alpha=0.7)

    df0 = df_cap_sum.loc[df_cap_sum["Future.ID"] == 0]
    if not df0.empty:
        plt.plot(df0["YEAR"], df0["TotalCapacity"],
                 color="blue", linewidth=2.5, label="Scenario 0")

    plt.title("Total Installed Capacity")
    plt.xlabel("Year")
    plt.ylabel("Capacity [GW]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


# -------------------------------------------------------------------
# 9. Line: annual CO2 emissions (from plots10_line_emissions.py)
# -------------------------------------------------------------------
def plot_line_emissions(key, df_in=None, df_out=None, save=False):
    df_out_local = df_out

    df_em = (
        df_out_local.loc[df_out_local.get("EMISSION") == "CO2",
                         ["Future.ID", "YEAR", "AnnualTechnologyEmission"]]
        .assign(AnnualTechnologyEmission=lambda d: pd.to_numeric(d["AnnualTechnologyEmission"], errors="coerce"))
        .dropna(subset=["AnnualTechnologyEmission"])
    )
    df_em = df_em.loc[df_em["AnnualTechnologyEmission"] != 0]

    df_em_sum = (
        df_em.groupby(["Future.ID", "YEAR"], as_index=False)["AnnualTechnologyEmission"].sum()
              .rename(columns={"AnnualTechnologyEmission": "CO2_Emissions"})
    )

    plt.figure(figsize=(10, 6))
    for fid, grp in df_em_sum.groupby("Future.ID"):
        if fid != 0:
            plt.plot(grp["YEAR"], grp["CO2_Emissions"],
                     color="lightgrey", linewidth=1, alpha=0.7)

    df0 = df_em_sum.loc[df_em_sum["Future.ID"] == 0]
    if not df0.empty:
        plt.plot(df0["YEAR"], df0["CO2_Emissions"],
                 color="blue", linewidth=2.5, label="Scenario 0")

    plt.title("Annual CO₂ Emissions by Scenario")
    plt.xlabel("Year")
    plt.ylabel("Emissions [Mt CO2]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        
# -------------------------------------------------------------------
# 10. Boxplot: capacity to peak demand ratio
# -------------------------------------------------------------------      
def plot_boxplots_capacity_to_peak_ratio(key, df_in=None, df_out=None, save=False):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if df_in is None or df_out is None:
        raise ValueError("Both df_in and df_out must be provided.")

    df_inputs = df_in.copy()
    df_outputs = df_out.copy()

    techs = [
        "PWRSOL", "PWRWND", "BESS_TECH",
        "PWRNGS", "PWRGEO", "PWRHYD",
    ]

    tech_name_map = {
        "PWRSOL": "Solar",
        "PWRWND": "Wind",
        "BESS_TECH": "Battery storage",
        "PWRNGS": "Natural gas",
        "PWRGEO": "Geothermal",
        "PWRHYD": "Hydro",
    }

    demand_commodities = ["COMELC", "RESELC", "INDELC"]
    plot_years = [2030, 2035, 2040, 2045, 2050]
    peak_factor = 0.047

    # ------------------------------------------------------------------
    # 1. Build annual peak demand by Future.ID and YEAR
    # ------------------------------------------------------------------
    required_input_cols = ["Future.ID", "COMMODITY", "YEAR", "SpecifiedAnnualDemand"]
    missing_input_cols = [c for c in required_input_cols if c not in df_inputs.columns]
    if missing_input_cols:
        raise KeyError(f"df_in is missing required columns: {missing_input_cols}")

    demand = df_inputs[required_input_cols].copy()
    demand["YEAR"] = pd.to_numeric(demand["YEAR"], errors="coerce")
    demand["SpecifiedAnnualDemand"] = pd.to_numeric(
        demand["SpecifiedAnnualDemand"], errors="coerce"
    )

    demand = demand[
        demand["COMMODITY"].isin(demand_commodities) &
        demand["YEAR"].isin(plot_years)
    ].copy()

    demand = demand.dropna(subset=["Future.ID", "YEAR", "SpecifiedAnnualDemand"])
    demand = demand[demand["SpecifiedAnnualDemand"] != 0]

    peak_by_year = (
        demand.groupby(["Future.ID", "YEAR"], as_index=False)["SpecifiedAnnualDemand"]
        .sum()
        .rename(columns={"SpecifiedAnnualDemand": "AnnualDemand"})
    )

    peak_by_year["PeakDemand"] = peak_by_year["AnnualDemand"] * peak_factor
    peak_by_year = peak_by_year.replace([np.inf, -np.inf], np.nan).dropna(subset=["PeakDemand"])
    peak_by_year = peak_by_year[peak_by_year["PeakDemand"] > 0]

    if peak_by_year.empty:
        raise ValueError("No valid annual peak demand values could be computed from df_in.")

    # ------------------------------------------------------------------
    # 2. Aggregate installed capacity by technology group, Future.ID and YEAR
    # ------------------------------------------------------------------
    required_output_cols = ["Future.ID", "YEAR", "TECHNOLOGY", "TotalCapacityAnnual"]
    missing_output_cols = [c for c in required_output_cols if c not in df_outputs.columns]
    if missing_output_cols:
        raise KeyError(f"df_out is missing required columns: {missing_output_cols}")

    agg_list = []
    for prefix in techs:
        subset = df_outputs[
            df_outputs["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)
        ].copy()

        if subset.empty:
            continue

        subset["YEAR"] = pd.to_numeric(subset["YEAR"], errors="coerce")
        subset["TotalCapacityAnnual"] = pd.to_numeric(
            subset["TotalCapacityAnnual"], errors="coerce"
        )
        subset = subset.dropna(subset=["Future.ID", "YEAR", "TotalCapacityAnnual"])

        subset = subset[subset["YEAR"].isin(plot_years)].copy()
        if subset.empty:
            continue

        grouped = (
            subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"]
            .sum()
            .assign(TechGroup=prefix)
        )

        # Merge on both Future.ID and YEAR so the denominator is the
        # corresponding year's peak demand
        grouped = grouped.merge(
            peak_by_year[["Future.ID", "YEAR", "PeakDemand"]],
            on=["Future.ID", "YEAR"],
            how="inner"
        )

        grouped["CapacityToPeakRatio"] = grouped["TotalCapacityAnnual"] / grouped["PeakDemand"]
        grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["CapacityToPeakRatio"])

        if not grouped.empty:
            agg_list.append(grouped)

    if not agg_list:
        print("No technologies found for capacity-to-peak-demand ratio boxplots.")
        return

    df_all = pd.concat(agg_list, ignore_index=True)

    if df_all.empty:
        print("No valid data available after merging with annual peak demand.")
        return

    df_all["YEAR"] = df_all["YEAR"].astype(int)
    df_all = df_all[df_all["YEAR"].isin(plot_years)].copy()
    df_all["YEAR"] = df_all["YEAR"].astype(str)

    year_order = [str(y) for y in plot_years]

    ymin = 0
    ymax = df_all["CapacityToPeakRatio"].max() * 1.03

    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    g = sns.catplot(
        data=df_all,
        x="YEAR",
        y="CapacityToPeakRatio",
        col="TechGroup",
        col_wrap=3,
        kind="box",
        width=0.3,
        sharey=True,
        order=year_order,
        height=1.95,
        aspect=1.18,
        whis=[1, 99],
        fliersize=1.2,
        linewidth=0.7,
        boxprops={"edgecolor": "0.25", "linewidth": 0.5},
        whiskerprops={"color": "0.35", "linewidth": 0.5},
        capprops={"color": "0.35", "linewidth": 0.5},
        medianprops={"color": "0.15", "linewidth": 0.7},
    )

    g.set_titles("")
    g.set_axis_labels("Year", "Capacity / peak demand")

    ncols = 3

    for i, ax in enumerate(g.axes.flatten()):
        col = i % ncols
        tech_code = g.col_names[i]

        ax.set_title(tech_name_map.get(tech_code, tech_code), y=0.97, pad=1.5)
        ax.set_ylim(ymin, ymax)

        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.grid(axis="x", visible=False)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        if col in [1, 2]:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")

        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, pad=1)
        ax.tick_params(axis="y", pad=1)

    g.fig.set_size_inches(7.1, 4.35)
    g.fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.12,
        top=0.93,
        wspace=0.10,
        hspace=0.26
    )

    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        g.fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()
    
 # -------------------------------------------------------------------
 # 10. Boxplot: capacity and cpacity to peak demand ratio
 # -------------------------------------------------------------------     
    
def plot_boxplots_capacity_and_ratio(key, df_in=None, df_out=None, save=False):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if df_in is None or df_out is None:
        raise ValueError("Both df_in and df_out must be provided.")

    df_inputs = df_in.copy()
    df_outputs = df_out.copy()

    techs = [
        "PWRSOL", "PWRWND", "BESS_TECH",
        "PWRNGS", "PWRGEO", "PWRHYD",
    ]

    tech_name_map = {
        "PWRSOL": "Solar",
        "PWRWND": "Wind",
        "BESS_TECH": "Battery storage",
        "PWRNGS": "Natural gas",
        "PWRGEO": "Geothermal",
        "PWRHYD": "Hydro",
    }

    demand_commodities = ["COMELC", "RESELC", "INDELC"]
    plot_years = [2030, 2035, 2040, 2045, 2050]
    peak_factor = 0.047

    # ------------------------------------------------------------------
    # 1. Build annual peak demand by Future.ID and YEAR
    # ------------------------------------------------------------------
    required_input_cols = ["Future.ID", "COMMODITY", "YEAR", "SpecifiedAnnualDemand"]
    missing_input_cols = [c for c in required_input_cols if c not in df_inputs.columns]
    if missing_input_cols:
        raise KeyError(f"df_in is missing required columns: {missing_input_cols}")

    demand = df_inputs[required_input_cols].copy()
    demand["Future.ID"] = pd.to_numeric(demand["Future.ID"], errors="coerce")
    demand["YEAR"] = pd.to_numeric(demand["YEAR"], errors="coerce")
    demand["SpecifiedAnnualDemand"] = pd.to_numeric(
        demand["SpecifiedAnnualDemand"], errors="coerce"
    )

    demand = demand[
        demand["COMMODITY"].isin(demand_commodities) &
        demand["YEAR"].isin(plot_years)
    ].copy()

    demand = demand.dropna(subset=["Future.ID", "YEAR", "SpecifiedAnnualDemand"])

    peak_by_year = (
        demand.groupby(["Future.ID", "YEAR"], as_index=False)["SpecifiedAnnualDemand"]
        .sum()
        .rename(columns={"SpecifiedAnnualDemand": "AnnualDemand"})
    )

    peak_by_year["PeakDemand"] = peak_by_year["AnnualDemand"] * peak_factor
    peak_by_year = peak_by_year.replace([np.inf, -np.inf], np.nan)
    peak_by_year = peak_by_year.dropna(subset=["Future.ID", "YEAR", "PeakDemand"])
    peak_by_year = peak_by_year[peak_by_year["PeakDemand"] > 0].copy()

    if peak_by_year.empty:
        raise ValueError("No valid annual peak demand values could be computed from df_in.")

    # ------------------------------------------------------------------
    # 2. Aggregate installed capacity and compute ratio
    # ------------------------------------------------------------------
    required_output_cols = ["Future.ID", "YEAR", "TECHNOLOGY", "TotalCapacityAnnual"]
    missing_output_cols = [c for c in required_output_cols if c not in df_outputs.columns]
    if missing_output_cols:
        raise KeyError(f"df_out is missing required columns: {missing_output_cols}")

    df_outputs = df_outputs[required_output_cols].copy()
    df_outputs["Future.ID"] = pd.to_numeric(df_outputs["Future.ID"], errors="coerce")
    df_outputs["YEAR"] = pd.to_numeric(df_outputs["YEAR"], errors="coerce")
    df_outputs["TotalCapacityAnnual"] = pd.to_numeric(
        df_outputs["TotalCapacityAnnual"], errors="coerce"
    )

    agg_list = []

    for prefix in techs:
        subset = df_outputs[
            df_outputs["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)
        ].copy()

        if subset.empty:
            continue

        subset = subset.dropna(subset=["Future.ID", "YEAR", "TotalCapacityAnnual"])
        subset = subset[subset["YEAR"].isin(plot_years)].copy()

        if subset.empty:
            continue

        grouped = (
            subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"]
            .sum()
            .assign(TechGroup=prefix)
        )

        grouped = grouped.merge(
            peak_by_year[["Future.ID", "YEAR", "PeakDemand"]],
            on=["Future.ID", "YEAR"],
            how="inner"
        )

        if grouped.empty:
            continue

        grouped["CapacityToPeakRatio"] = grouped["TotalCapacityAnnual"] / grouped["PeakDemand"]
        grouped = grouped.replace([np.inf, -np.inf], np.nan)
        grouped = grouped.dropna(subset=["TotalCapacityAnnual", "CapacityToPeakRatio"])

        if not grouped.empty:
            agg_list.append(grouped)

    if not agg_list:
        raise ValueError(
            "No objects to concatenate. After filtering and merging, no matching data "
            "were found for the selected technologies and years."
        )

    df_all = pd.concat(agg_list, ignore_index=True)

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.1, 4.35))
    axes = axes.flatten()

    base_positions = np.arange(len(plot_years))
    offset = 0.16
    width = 0.24

    palette = sns.color_palette("colorblind")
    capacity_face = palette[0]
    ratio_face = palette[1]

    left_ymax = df_all["TotalCapacityAnnual"].max() * 1.03
    right_ymax = df_all["CapacityToPeakRatio"].max() * 1.03

    for i, prefix in enumerate(techs):
        ax = axes[i]
        ax2 = ax.twinx()

        tech_df = df_all[df_all["TechGroup"] == prefix].copy()

        cap_data = []
        ratio_data = []

        for yr in plot_years:
            yr_df = tech_df[tech_df["YEAR"] == yr]
            cap_vals = yr_df["TotalCapacityAnnual"].dropna().values
            ratio_vals = yr_df["CapacityToPeakRatio"].dropna().values

            cap_data.append(cap_vals if len(cap_vals) > 0 else np.array([np.nan]))
            ratio_data.append(ratio_vals if len(ratio_vals) > 0 else np.array([np.nan]))

        pos_cap = base_positions - offset
        pos_rat = base_positions + offset

        ax.boxplot(
            cap_data,
            positions=pos_cap,
            widths=width,
            patch_artist=True,
            whis=[1, 99],
            showfliers=True,
            boxprops=dict(facecolor=capacity_face, edgecolor="0.25", linewidth=0.5),
            whiskerprops=dict(color="0.35", linewidth=0.5),
            capprops=dict(color="0.35", linewidth=0.5),
            medianprops=dict(color="0.10", linewidth=0.7),
            flierprops=dict(
                marker="o",
                markersize=1.2,
                markerfacecolor=capacity_face,
                markeredgecolor=capacity_face,
                alpha=0.7,
            ),
        )

        ax2.boxplot(
            ratio_data,
            positions=pos_rat,
            widths=width,
            patch_artist=True,
            whis=[1, 99],
            showfliers=True,
            boxprops=dict(facecolor=ratio_face, edgecolor="0.25", linewidth=0.5),
            whiskerprops=dict(color="0.35", linewidth=0.5),
            capprops=dict(color="0.35", linewidth=0.5),
            medianprops=dict(color="0.10", linewidth=0.7),
            flierprops=dict(
                marker="o",
                markersize=1.2,
                markerfacecolor=ratio_face,
                markeredgecolor=ratio_face,
                alpha=0.7,
            ),
        )

        ax.set_title(tech_name_map.get(prefix, prefix), y=0.97, pad=1.5)

        ax.set_xlim(-0.6, len(plot_years) - 0.4)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(y) for y in plot_years])

        ax.set_ylim(0, left_ymax)
        ax2.set_ylim(0, right_ymax)

        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.grid(axis="x", visible=False)
        ax2.grid(False)

        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax2.spines["right"].set_linewidth(0.6)

        col = i % ncols

        if col == 0:
            ax.set_ylabel("Capacity [GW]")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        if col == 2:
            ax2.set_ylabel("Capacity / peak demand")
        else:
            ax2.set_ylabel("")
            ax2.tick_params(axis="y", labelright=False)

        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, pad=1)
        ax.tick_params(axis="y", pad=1)
        ax2.tick_params(axis="y", pad=1)

    for j in range(len(techs), len(axes)):
        fig.delaxes(axes[j])

    fig.legend(
        handles=[
            Patch(facecolor=capacity_face, edgecolor="0.25", label="Capacity"),
            Patch(facecolor=ratio_face, edgecolor="0.25", label="Capacity / peak demand"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0), # -0.02),
        handlelength=1.4,
        columnspacing=1.5,
    )
    
    fig.subplots_adjust(
        left=0.08,
        right=0.93,
        bottom=0.12,
        top=0.88,
        wspace=0.16,
        hspace=0.28,
    )

    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()
    
    
def plot_boxplots_capacity_and_activity(key, df_in=None, df_out=None, save=False):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if df_out is None:
        raise ValueError("df_out must be provided.")

    df = df_out.copy()

    techs = [
        "PWRSOL", "PWRWND", "BESS_TECH",
        "PWRNGS", "PWRGEO", "PWRHYD",
    ]

    tech_name_map = {
        "PWRSOL": "Solar",
        "PWRWND": "Wind",
        "BESS_TECH": "Battery storage",
        "PWRNGS": "Natural gas",
        "PWRGEO": "Geothermal",
        "PWRHYD": "Hydro",
    }

    plot_years = [2030, 2035, 2040, 2045, 2050]

    cap_list = []
    act_list = []

    for prefix in techs:
        subset = df[df["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)].copy()
        if subset.empty:
            continue

        subset["YEAR"] = pd.to_numeric(subset["YEAR"], errors="coerce")
        subset = subset[subset["YEAR"].isin(plot_years)].copy()
        if subset.empty:
            continue

        # Capacity aggregation
        cap_subset = subset[["Future.ID", "YEAR", "TotalCapacityAnnual"]].copy()
        cap_subset["TotalCapacityAnnual"] = pd.to_numeric(
            cap_subset["TotalCapacityAnnual"], errors="coerce"
        )
        cap_subset = cap_subset.dropna(subset=["Future.ID", "YEAR", "TotalCapacityAnnual"])

        if not cap_subset.empty:
            cap_grouped = (
                cap_subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"]
                .sum()
                .assign(TechGroup=prefix)
            )
            cap_list.append(cap_grouped)

        # Activity aggregation
        act_subset = subset[["Future.ID", "YEAR", "TotalTechnologyAnnualActivity"]].copy()
        act_subset["TotalTechnologyAnnualActivity"] = pd.to_numeric(
            act_subset["TotalTechnologyAnnualActivity"], errors="coerce"
        )
        act_subset = act_subset.dropna(
            subset=["Future.ID", "YEAR", "TotalTechnologyAnnualActivity"]
        )

        if not act_subset.empty:
            act_grouped = (
                act_subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalTechnologyAnnualActivity"]
                .sum()
                .assign(TechGroup=prefix)
            )
            act_list.append(act_grouped)

    if not cap_list:
        raise ValueError("No capacity data found for the selected technologies and years.")

    if not act_list:
        raise ValueError("No activity data found for the selected technologies and years.")

    df_cap = pd.concat(cap_list, ignore_index=True)
    df_act = pd.concat(act_list, ignore_index=True)

    df_all = pd.merge(
        df_cap,
        df_act,
        on=["Future.ID", "YEAR", "TechGroup"],
        how="outer"
    )

    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.1, 4.35))
    axes = axes.flatten()

    base_positions = np.arange(len(plot_years))
    offset = 0.16
    width = 0.24

    palette = sns.color_palette("colorblind")
    capacity_face = palette[0]
    activity_face = palette[1]

    left_ymax = df_cap["TotalCapacityAnnual"].max() * 1.03
    right_ymax = df_act["TotalTechnologyAnnualActivity"].max() * 1.03

    for i, prefix in enumerate(techs):
        ax = axes[i]
        ax2 = ax.twinx()

        tech_df = df_all[df_all["TechGroup"] == prefix].copy()

        cap_data = []
        act_data = []

        for yr in plot_years:
            yr_df = tech_df[tech_df["YEAR"] == yr]

            cap_vals = yr_df["TotalCapacityAnnual"].dropna().values
            act_vals = yr_df["TotalTechnologyAnnualActivity"].dropna().values

            cap_data.append(cap_vals if len(cap_vals) > 0 else np.array([np.nan]))
            act_data.append(act_vals if len(act_vals) > 0 else np.array([np.nan]))

        pos_cap = base_positions - offset
        pos_act = base_positions + offset

        ax.boxplot(
            cap_data,
            positions=pos_cap,
            widths=width,
            patch_artist=True,
            whis=[1, 99],
            showfliers=True,
            boxprops=dict(facecolor=capacity_face, edgecolor="0.25", linewidth=0.5),
            whiskerprops=dict(color="0.35", linewidth=0.5),
            capprops=dict(color="0.35", linewidth=0.5),
            medianprops=dict(color="0.10", linewidth=0.7),
            flierprops=dict(
                marker="o",
                markersize=1.2,
                markerfacecolor=capacity_face,
                markeredgecolor=capacity_face,
                alpha=0.7,
            ),
        )

        ax2.boxplot(
            act_data,
            positions=pos_act,
            widths=width,
            patch_artist=True,
            whis=[1, 99],
            showfliers=True,
            boxprops=dict(facecolor=activity_face, edgecolor="0.25", linewidth=0.5),
            whiskerprops=dict(color="0.35", linewidth=0.5),
            capprops=dict(color="0.35", linewidth=0.5),
            medianprops=dict(color="0.10", linewidth=0.7),
            flierprops=dict(
                marker="o",
                markersize=1.2,
                markerfacecolor=activity_face,
                markeredgecolor=activity_face,
                alpha=0.7,
            ),
        )

        ax.set_title(tech_name_map.get(prefix, prefix), y=0.97, pad=1.5)

        ax.set_xlim(-0.6, len(plot_years) - 0.4)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(y) for y in plot_years])

        ax.set_ylim(0, left_ymax)
        ax2.set_ylim(0, right_ymax)

        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.grid(axis="x", visible=False)
        ax2.grid(False)

        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax2.spines["right"].set_linewidth(0.6)

        col = i % ncols

        if col == 0:
            ax.set_ylabel("Capacity [GW]")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        if col == 2:
            ax2.set_ylabel("Activity [PJ]")
        else:
            ax2.set_ylabel("")
            ax2.tick_params(axis="y", labelright=False)

        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, pad=1)
        ax.tick_params(axis="y", pad=1)
        ax2.tick_params(axis="y", pad=1)

    for j in range(len(techs), len(axes)):
        fig.delaxes(axes[j])

    fig.legend(
        handles=[
            Patch(facecolor=capacity_face, edgecolor="0.25", label="Capacity"),
            Patch(facecolor=activity_face, edgecolor="0.25", label="Activity"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0),
        handlelength=1.4,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.93,
        bottom=0.12,
        top=0.88,
        wspace=0.16,
        hspace=0.28,
    )

    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()
    
def plot_boxplots_gas_activity_to_capacity_ratio(key, df_in=None, df_out=None, save=False):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    if df_out is None:
        raise ValueError("df_out must be provided.")

    df = df_out.copy()
    tech_prefix = "PWRNGS"

    subset = df[df["TECHNOLOGY"].astype(str).str.contains(tech_prefix, na=False)].copy()

    if subset.empty:
        print("No PWRNGS technologies found.")
        return

    subset["YEAR"] = pd.to_numeric(subset["YEAR"], errors="coerce")

    plot_years = [2030, 2035, 2040, 2045, 2050]
    subset = subset[subset["YEAR"].isin(plot_years)].copy()

    if subset.empty:
        print("No PWRNGS data found for the selected years.")
        return

    # Aggregate capacity separately
    cap_subset = subset[["Future.ID", "YEAR", "TotalCapacityAnnual"]].copy()
    cap_subset["TotalCapacityAnnual"] = pd.to_numeric(
        cap_subset["TotalCapacityAnnual"], errors="coerce"
    )
    cap_subset = cap_subset.dropna(subset=["Future.ID", "YEAR", "TotalCapacityAnnual"])

    if cap_subset.empty:
        print("No valid PWRNGS capacity data.")
        return

    df_cap = (
        cap_subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalCapacityAnnual"]
        .sum()
    )

    # Aggregate activity separately
    act_subset = subset[["Future.ID", "YEAR", "TotalTechnologyAnnualActivity"]].copy()
    act_subset["TotalTechnologyAnnualActivity"] = pd.to_numeric(
        act_subset["TotalTechnologyAnnualActivity"], errors="coerce"
    )
    act_subset = act_subset.dropna(
        subset=["Future.ID", "YEAR", "TotalTechnologyAnnualActivity"]
    )

    if act_subset.empty:
        print("No valid PWRNGS activity data.")
        return

    df_act = (
        act_subset.groupby(["Future.ID", "YEAR"], as_index=False)["TotalTechnologyAnnualActivity"]
        .sum()
    )

    # Merge and compute ratio
    grouped = pd.merge(
        df_cap,
        df_act,
        on=["Future.ID", "YEAR"],
        how="inner"
    )

    grouped = grouped[grouped["TotalCapacityAnnual"] > 0].copy()

    if grouped.empty:
        print("No overlapping PWRNGS capacity and activity data after aggregation.")
        return

    grouped["ActCapRatio"] = (
        grouped["TotalTechnologyAnnualActivity"]
        / (grouped["TotalCapacityAnnual"] * 31.356)
    )

    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    grouped = grouped.dropna(subset=["ActCapRatio"])

    if grouped.empty:
        print("No valid ratio values to plot.")
        return

    grouped["YEAR"] = grouped["YEAR"].astype(int).astype(str)
    year_order = [str(y) for y in plot_years]

    ymin = 0
    ymax = grouped["ActCapRatio"].max() * 1.03

    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    fig, ax = plt.subplots(figsize=(2.35, 1.95))

    sns.boxplot(
        data=grouped,
        x="YEAR",
        y="ActCapRatio",
        order=year_order,
        width=0.3,
        whis=[1, 99],
        fliersize=1.2,
        linewidth=0.7,
        boxprops={"edgecolor": "0.25", "linewidth": 0.5},
        whiskerprops={"color": "0.35", "linewidth": 0.5},
        capprops={"color": "0.35", "linewidth": 0.5},
        medianprops={"color": "0.15", "linewidth": 0.7},
        ax=ax,
    )

    ax.set_title("Natural gas", y=0.90, pad=1.5)
    ax.set_xlabel("")
    ax.set_ylabel("Activity / (capacity × 31.356)")
    ax.set_ylim(ymin, ymax)

    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    ax.tick_params(axis="x", rotation=0, pad=1)
    ax.tick_params(axis="y", pad=1)

    fig.tight_layout()

    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()

# -------------------------------------------------------------------
# Main: select which plots to run
# -------------------------------------------------------------------
AVAILABLE_PLOTS = {
    "box_capacity":       plot_boxplots_capacity,
    "box_activity":       plot_boxplots_activity,
    "bar_gas_capacity":   plot_bar_gas_capacity,
    "bar_gas_capacity_ratio": plot_bar_gas_capacity_ratio,
    "scatter_bess_gas":   plot_scatter_bess_vs_gas,
    "line_demand":        plot_line_demand,
    "line_lcoe":          plot_line_lcoe,
    "line_gas_capex":     plot_line_gas_capex,
    "line_total_capacity": plot_line_total_capacity,
    "line_emissions":     plot_line_emissions,
    "box_captodemratio": plot_boxplots_capacity_to_peak_ratio,
    "box_cap_and_capdemratio": plot_boxplots_capacity_and_ratio,
    "box_cap_and_act": plot_boxplots_capacity_and_activity,
    "box_gas_acttocap": plot_boxplots_gas_activity_to_capacity_ratio,
}


if __name__ == "__main__":

    print("Loading input/output CSVs...")
    df_in = pd.read_csv(CSV_INPUT, low_memory=False)
    df_out = pd.read_csv(CSV_OUTPUT, low_memory=False)
    
    # Edit this list to choose which plots to generate
    plots_to_run = [
        # "box_capacity",
        # "box_activity",
        # "bar_gas_capacity",
        # "bar_gas_capacity_ratio",
        # "scatter_bess_gas",
        "line_demand",
        # # "line_lcoe",
        # "line_gas_capex",
        # "line_total_capacity",
        # "line_emissions",
        # "box_captodemratio",
        # "box_cap_and_capdemratio",
        # "box_cap_and_act",
        # "box_gas_acttocap",
    ]

    for key in plots_to_run:
        fn = AVAILABLE_PLOTS.get(key)
        if fn is None:
            print(f"[WARNING] Unknown plot key: {key}")
            continue
        print(f"\n--- Running plot: {key} ---")
        fn(key, df_in=df_in, df_out=df_out, save=True)
