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
def plot_boxplots_capacity(key, df_in=None, df_out=None, save=False):
    df = df_out

    def plot_capacity_boxplot_for_prefix(
        df: pd.DataFrame,
        tech_prefix: str,
        value_col: str = "TotalCapacityAnnual",
        year_col: str = "YEAR",
        future_col: str = "Future.ID",
        title: str | None = None,
        save: bool = False,
        out_dir: str = ".",
        figsize=(12, 6),
    ):
        # Filter technologies
        mask = df["TECHNOLOGY"].astype(str).str.contains(tech_prefix, na=False)
        subset = df.loc[mask, [year_col, future_col, value_col]].copy()

        if subset.empty:
            print(f"[{tech_prefix}] No rows matched. Skipping.")
            return

        subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
        subset = subset.dropna(subset=[value_col])
        if subset.empty:
            print(f"[{tech_prefix}] All values NaN. Skipping.")
            return

        grouped = (
            subset.groupby([year_col, future_col], as_index=False)[value_col]
                  .sum()
        )

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

    # Multi-panel boxplots
    techs = [
        "PWRNGS", "PWRSOL", "PWRWND", "PWRGEO",
        "BESS_TECH", "PWRBIO",
        "PWRHFO", "PWRHYD", "PWRPHS", "PWRURN", "IMPELC", "BACKSTOP"
    ]

    agg_list = []
    for prefix in techs:
        subset = df[df["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)].copy()
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

    df_all["YEAR"] = df_all["YEAR"].astype(str)
    year_order = sorted(df_all["YEAR"].unique(), key=int)

    g = sns.catplot(
        data=df_all,
        x="YEAR", y="TotalCapacityAnnual",
        col="TechGroup", col_wrap=3,
        kind="box", sharey=False,
        height=3.5, aspect=1.2,
        order=year_order,
        whis=[1, 99],
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Installed capacity [GW]")

    ymin, ymax = df_all["TotalCapacityAnnual"].min(), df_all["TotalCapacityAnnual"].max()
    for ax in g.axes.flatten():
        ax.set_ylim(ymin, ymax)

    years_num = list(map(int, year_order))
    keep_idx = [i for i, y in enumerate(years_num) if y >= 2020 and (y - 2020) % 5 == 0]
    keep_labs = [str(years_num[i]) for i in keep_idx]

    for ax in g.axes.flatten():
        ax.xaxis.set_major_locator(mticker.FixedLocator(keep_idx))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(keep_labs))

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Capacity Across Futures (Boxplots by Technology)")
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


# -------------------------------------------------------------------
# 2. Boxplots: activity by tech
# -------------------------------------------------------------------
def plot_boxplots_activity(key, df_in=None, df_out=None, save=False):
    df = df_out

    def plot_capacity_boxplot_for_prefix(
        df: pd.DataFrame,
        tech_prefix: str,
        value_col: str = "TotalTechnologyAnnualActivity",
        year_col: str = "YEAR",
        future_col: str = "Future.ID",
        title: str | None = None,
        save: bool = False,
        out_dir: str = ".",
        figsize=(12, 6),
    ):
        mask = df["TECHNOLOGY"].astype(str).str.contains(tech_prefix, na=False)
        subset = df.loc[mask, [year_col, future_col, value_col]].copy()

        if subset.empty:
            print(f"[{tech_prefix}] No rows matched. Skipping.")
            return

        subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
        subset = subset.dropna(subset=[value_col])
        if subset.empty:
            print(f"[{tech_prefix}] All values NaN. Skipping.")
            return

        grouped = (
            subset.groupby([year_col, future_col], as_index=False)[value_col]
                  .sum()
        )

        plt.figure(figsize=figsize)
        sns.boxplot(data=grouped, x=year_col, y=value_col, color="skyblue")
        plt.title(title or f"{tech_prefix} Activity across Futures")
        plt.xlabel("Year")
        plt.ylabel(f"{value_col} (summed across sub-techs)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.show()

    techs = [
        "PWRNGS", "PWRSOL", "PWRWND", "PWRGEO",
        "BESS_TECH", "PWRBIO",
        "PWRHFO", "PWRHYD", "PWRPHS", "PWRURN", "IMPELC", "BACKSTOP"
    ]

    agg_list = []
    for prefix in techs:
        subset = df[df["TECHNOLOGY"].astype(str).str.contains(prefix, na=False)].copy()
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

    df_all["YEAR"] = df_all["YEAR"].astype(str)
    year_order = sorted(df_all["YEAR"].unique(), key=int)

    g = sns.catplot(
        data=df_all,
        x="YEAR", y="TotalTechnologyAnnualActivity",
        col="TechGroup", col_wrap=3,
        kind="box", sharey=False,
        height=3.5, aspect=1.2,
        order=year_order,
        whis=[1, 99],
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Annual activity [PJ]")

    years_num = list(map(int, year_order))
    keep_idx = [i for i, y in enumerate(years_num) if y >= 2020 and (y - 2020) % 5 == 0]
    keep_labs = [str(years_num[i]) for i in keep_idx]

    for ax in g.axes.flatten():
        ax.xaxis.set_major_locator(mticker.FixedLocator(keep_idx))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(keep_labs))

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Activity Across Futures (Boxplots by Technology)")
    plt.show()
    
    if save:
        out_path = os.path.join(PLOT_DIR, f"{key}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


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

    plt.figure(figsize=(14, 6))
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
    plt.ylabel("Natural Gas Capacity in 2050 (PWRNGS*)")
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

    df_solar = (
        df_in_local.loc[
            df_in_local["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False),
            ["Future.ID", "YEAR", "CapitalCost"],
        ]
        .assign(CapitalCost=lambda d: pd.to_numeric(d["CapitalCost"], errors="coerce"))
        .dropna(subset=["CapitalCost"])
    )
    df_solar = df_solar.loc[df_solar["CapitalCost"] != 0]

    df_solar_avg = (
        df_solar.groupby(["Future.ID", "YEAR"], as_index=False)["CapitalCost"].mean()
                .rename(columns={"CapitalCost": "CapitalCost_Solar"})
    )

    plt.figure(figsize=(10, 6))
    for fid, grp in df_solar_avg.groupby("Future.ID"):
        if fid != 0:
            plt.plot(grp["YEAR"], grp["CapitalCost_Solar"],
                     color="lightgrey", linewidth=1, alpha=0.7)

    df0 = df_solar_avg.loc[df_solar_avg["Future.ID"] == 0]
    if not df0.empty:
        plt.plot(df0["YEAR"], df0["CapitalCost_Solar"],
                 color="blue", linewidth=2.5, label="Scenario 0")

    plt.title("Solar Capital Cost")  # original title, even if tech is PWRNGS
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
# Main: select which plots to run
# -------------------------------------------------------------------
AVAILABLE_PLOTS = {
    "box_capacity":       plot_boxplots_capacity,
    "box_activity":       plot_boxplots_activity,
    "bar_gas_capacity":   plot_bar_gas_capacity,
    "scatter_bess_gas":   plot_scatter_bess_vs_gas,
    "line_demand":        plot_line_demand,
    "line_lcoe":          plot_line_lcoe,
    "line_gas_capex":     plot_line_gas_capex,
    "line_total_capacity": plot_line_total_capacity,
    "line_emissions":     plot_line_emissions,
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
        # "scatter_bess_gas",
        # "line_demand",
        # "line_lcoe",
        # "line_gas_capex",
        # "line_total_capacity",
        "line_emissions",
    ]

    for key in plots_to_run:
        fn = AVAILABLE_PLOTS.get(key)
        if fn is None:
            print(f"[WARNING] Unknown plot key: {key}")
            continue
        print(f"\n--- Running plot: {key} ---")
        fn(key, df_in=df_in, df_out=df_out, save=True)
