# -*- coding: utf-8 -*-
"""
Interactive Plotly bar charts for:
1) Natural gas capacity in 2050
2) Natural gas capacity / peak demand ratio in 2050

Hover info:
- Future ID
- Annual demand in 2050 (COMELC + RESELC + INDELC)
- Natural gas capacity in 2050
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

# ----------------------------
# Config
# ----------------------------
CSV_INPUT = "OSEMOSYS_Energy_Input.csv"
CSV_OUTPUT = "OSEMOSYS_Energy_Output.csv"
OUT_DIR = "plots_plotly"

# Force a renderer that works in scripts (Spyder/terminal)
# Alternatives: "browser", "firefox", "chrome", "edge"
pio.renderers.default = "browser"


def build_dataset(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    commodities = ["COMELC", "RESELC", "INDELC"]

    demand2050 = (
        df_in.loc[
            (df_in["YEAR"] == 2050) & (df_in["COMMODITY"].isin(commodities)),
            ["Future.ID", "SpecifiedAnnualDemand"],
        ]
        .assign(SpecifiedAnnualDemand=lambda d: pd.to_numeric(d["SpecifiedAnnualDemand"], errors="coerce"))
        .dropna(subset=["SpecifiedAnnualDemand"])
        .groupby("Future.ID", as_index=False)["SpecifiedAnnualDemand"]
        .sum()
        .rename(columns={"SpecifiedAnnualDemand": "AnnualDemand2050"})
    )

    # Your conversion from annual demand to peak demand
    demand2050["PeakDemand2050"] = demand2050["AnnualDemand2050"] * (2.35 / 50.51)

    gas2050 = (
        df_out.loc[
            (df_out["YEAR"] == 2050)
            & (df_out["TECHNOLOGY"].astype(str).str.contains("PWRNGS", na=False)),
            ["Future.ID", "TotalCapacityAnnual"],
        ]
        .assign(TotalCapacityAnnual=lambda d: pd.to_numeric(d["TotalCapacityAnnual"], errors="coerce"))
        .dropna(subset=["TotalCapacityAnnual"])
        .groupby("Future.ID", as_index=False)["TotalCapacityAnnual"]
        .sum()
        .rename(columns={"TotalCapacityAnnual": "GasCapacity2050"})
    )

    df = pd.merge(gas2050, demand2050, on="Future.ID", how="inner")
    if df.empty:
        return df

    df["Gas_to_Peak_Ratio"] = df["GasCapacity2050"] / df["PeakDemand2050"]
    return df


def plot_gas_capacity(df: pd.DataFrame):
    d = df.sort_values("GasCapacity2050", ascending=False).copy()
    d["Future.ID"] = d["Future.ID"].astype(int)

    fig = px.bar(d, x="Future.ID", y="GasCapacity2050", title="Natural Gas Capacity in 2050")

    fig.update_traces(
        customdata=np.stack([d["AnnualDemand2050"].to_numpy(), d["GasCapacity2050"].to_numpy()], axis=1),
        hovertemplate=(
            "Future ID: %{x}<br>"
            "Annual demand 2050: %{customdata[0]:,.2f}<br>"
            "Gas capacity 2050: %{customdata[1]:,.2f}<extra></extra>"
        ),
    )

    fig.update_layout(xaxis_title="Future ID", yaxis_title="Natural gas capacity (GW)", bargap=0.0)
    return fig


def plot_gas_ratio(df: pd.DataFrame):
    d = df.sort_values("Gas_to_Peak_Ratio", ascending=False).copy()
    d["Future.ID"] = d["Future.ID"].astype(int)

    fig = px.bar(d, x="Future.ID", y="Gas_to_Peak_Ratio", title="Natural Gas Capacity / Peak Demand Ratio in 2050")

    fig.update_traces(
        customdata=np.stack([d["AnnualDemand2050"].to_numpy(), d["GasCapacity2050"].to_numpy()], axis=1),
        hovertemplate=(
            "Future ID: %{x}<br>"
            "Annual demand 2050: %{customdata[0]:,.2f}<br>"
            "Gas capacity 2050: %{customdata[1]:,.2f}<br>"
            "Gas / peak demand: %{y:.3f}<extra></extra>"
        ),
    )

    fig.update_layout(xaxis_title="Future ID", yaxis_title="Gas capacity / peak demand", bargap=0.0)
    return fig


if __name__ == "__main__":
    print("Loading CSV files...")
    df_in = pd.read_csv(CSV_INPUT, low_memory=False)
    df_out = pd.read_csv(CSV_OUTPUT, low_memory=False)

    df = build_dataset(df_in, df_out)
    if df.empty:
        raise RuntimeError("No overlapping futures found between gas capacity and demand (2050).")

    os.makedirs(OUT_DIR, exist_ok=True)

    fig1 = plot_gas_capacity(df)
    fig2 = plot_gas_ratio(df)

    # Save HTML (always works)
    p1 = os.path.join(OUT_DIR, "plotly_gas_capacity_2050.html")
    p2 = os.path.join(OUT_DIR, "plotly_gas_ratio_2050.html")
    fig1.write_html(p1, include_plotlyjs="cdn")
    fig2.write_html(p2, include_plotlyjs="cdn")
    print(f"Saved:\n- {p1}\n- {p2}")

    # Show (opens browser tabs/windows)
    fig1.show()
    fig2.show()