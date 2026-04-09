"""
PRIM (EMA) script for GAS PRESENT in 2050.

- Reads OSeMOSYS_Energy_Input / Output.
- Builds scenario-level features and a binary outcome.
- Runs PRIM using ema_workbench.analysis.prim.Prim only.

Usage example
-------------
python scenario_discovery2.py
python scenario_discovery2.py --inputs OSEMOSYS_Energy_Input.csv --outputs OSEMOSYS_Energy_Output.csv
"""

import os
import argparse

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---- Configuration ---------------------------------------------------------

INPUTS_PATH_DEFAULT = "OSEMOSYS_Energy_Input.csv"
OUTPUTS_PATH_DEFAULT = "OSEMOSYS_Energy_Output.csv"
OUTDIR_BASE = "scenario_discovery_artifacts"
YEAR = 2050

os.makedirs(OUTDIR_BASE, exist_ok=True)

# ---- IO helpers ------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    elif path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    else:
        raise SystemExit(f"Unsupported file type for: {path}")

# ---- Feature & outcome builders --------------------------------------------

def build_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario-level features based on 2050 values and discount rates.
    """
    if "Scen_fut" not in inputs.columns:
        raise ValueError("Expected 'Scen_fut' key column in inputs.")

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    feats = []

    for scen, sub in inputs.groupby("Scen_fut", sort=False):
        f = {"Scen_fut": scen}
        sub50 = sub.loc[sub.get("YEAR").eq(YEAR) if "YEAR" in sub.columns else []]

        # (1) CapitalCost in 2050 over PWRGEO*
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            geo50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRGEO")]
            vals = to_num(geo50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_geo_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_geo_2050_mean"] = np.nan

        # (2) VariableCost of IMPNGS in 2050 (drop 0 and NaN), median
        if {"TECHNOLOGY", "VariableCost"}.issubset(sub50.columns):
            gas50 = sub50.loc[sub50["TECHNOLOGY"].astype(str) == "IMPNGS", "VariableCost"]
            gas50 = to_num(gas50)
            gas50 = gas50[(~gas50.isna()) & (gas50 != 0)]
            f["gas_price_2050_median_nonzero"] = gas50.median() if not gas50.empty else np.nan
        else:
            f["gas_price_2050_median_nonzero"] = np.nan

        # (3) CapitalCost in 2050 over IMPNGS
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            impngs50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("IMPNGS")]
            vals = to_num(impngs50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_impngs_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_impngs_2050_mean"] = np.nan

        # (4) CapitalCost in 2050 over BESS_TECH
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            bat50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("BESS_TECH")]
            vals = to_num(bat50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_bess_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_bess_2050_mean"] = np.nan

        # (5) CapitalCost in 2050 over PWRSOL
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            sol50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL001")]
            vals = to_num(sol50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_solar_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_solar_2050_mean"] = np.nan

        # (6) CapitalCost in 2050 over PWRWND
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            wnd50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRWND")]
            vals = to_num(wnd50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_wind_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_wind_2050_mean"] = np.nan

        # (7) DiscountRateIdv for batteries (BESS_TECH) in 2050
        if {"TECHNOLOGY", "YEAR", "DiscountRateIdv"}.issubset(sub.columns):
            bat50 = sub.loc[
                (sub["TECHNOLOGY"].astype(str).str.startswith("BESS_TECH"))
                & (sub["YEAR"] == YEAR)
            ]
            vals = to_num(bat50["DiscountRateIdv"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["discount_rate_batt_2050"] = vals.mean() if not vals.empty else np.nan
        else:
            f["discount_rate_batt_2050"] = np.nan

        # (8) TotalAnnualMaxCapacity for PWRGEO006 in 2050
        if {"TECHNOLOGY", "TotalAnnualMaxCapacity"}.issubset(sub50.columns):
            geo006_50 = sub50.loc[
                sub50["TECHNOLOGY"].astype(str) == "PWRGEO006",
                "TotalAnnualMaxCapacity",
            ]
            vals = to_num(geo006_50)
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["maxcap_geo006_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["maxcap_geo006_2050_mean"] = np.nan
            
        # (9) DiscountRateIdv for solar (PWRSOL) in 2050
        if {"TECHNOLOGY", "YEAR", "DiscountRateIdv"}.issubset(sub.columns):
            sol50 = sub.loc[
                (sub["TECHNOLOGY"].astype(str).str.startswith("PWRSOL001"))
                & (sub["YEAR"] == YEAR)
            ]
            vals = to_num(sol50["DiscountRateIdv"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["discount_rate_solar_2050"] = vals.mean() if not vals.empty else np.nan
        else:
            f["discount_rate_solar_2050"] = np.nan
        
        # (10) DiscountRateIdv for wind (PWRWND) in 2050
        if {"TECHNOLOGY", "YEAR", "DiscountRateIdv"}.issubset(sub.columns):
            wnd50 = sub.loc[
                (sub["TECHNOLOGY"].astype(str).str.startswith("PWRWND"))
                & (sub["YEAR"] == YEAR)
            ]
            vals = to_num(wnd50["DiscountRateIdv"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["discount_rate_wind_2050"] = vals.mean() if not vals.empty else np.nan
        else:
            f["discount_rate_wind_2050"] = np.nan
        
        # (11) DiscountRateIdv for gas imports (IMPNGS) in 2050
        if {"TECHNOLOGY", "YEAR", "DiscountRateIdv"}.issubset(sub.columns):
            gas50 = sub.loc[
                (sub["TECHNOLOGY"].astype(str).str.startswith("IMPNGS"))
                & (sub["YEAR"] == YEAR)
            ]
            vals = to_num(gas50["DiscountRateIdv"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["discount_rate_impngs_2050"] = vals.mean() if not vals.empty else np.nan
        else:
            f["discount_rate_impngs_2050"] = np.nan
        
        # (12) DiscountRateIdv for geothermal (PWRGEO) in 2050
        if {"TECHNOLOGY", "YEAR", "DiscountRateIdv"}.issubset(sub.columns):
            geo50 = sub.loc[
                (sub["TECHNOLOGY"].astype(str).str.startswith("PWRGEO006"))
                & (sub["YEAR"] == YEAR)
            ]
            vals = to_num(geo50["DiscountRateIdv"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["discount_rate_geo_2050"] = vals.mean() if not vals.empty else np.nan
        else:
            f["discount_rate_geo_2050"] = np.nan    

        feats.append(f)

    X = pd.DataFrame(feats)

    # Drop all-NaN columns; keep Scen_fut
    keep_cols = [c for c in X.columns if c == "Scen_fut" or X[c].notna().any()]
    X = X[keep_cols]

    nunique = X.drop(columns=["Scen_fut"]).nunique(dropna=True)
    print("Feature variability (nunique):")
    print(nunique.sort_values())

    print("\nFeature summary statistics:")
    summary = X.drop(columns=["Scen_fut"]).describe().T
    print(summary.to_string())

    return X


def build_outcome_gas_present_2050(outputs: pd.DataFrame) -> pd.DataFrame:
    """
    Binary outcome per scenario:
    1 if gas capacity (PWRNGS001) is present in 2050, else 0.
    """
    required = {"Scen_fut", "YEAR", "TECHNOLOGY", "TotalCapacityAnnual"}
    if not required.issubset(outputs.columns):
        miss = sorted(required - set(outputs.columns))
        raise ValueError(f"Outputs missing required columns: {miss}")

    out = outputs.copy()
    out["TotalCapacityAnnual"] = pd.to_numeric(
        out["TotalCapacityAnnual"], errors="coerce"
    ).fillna(0.0)

    gas2050 = (
        out.loc[(out["YEAR"] == YEAR) & (out["TECHNOLOGY"].astype(str) == "PWRNGS001")]
        .groupby("Scen_fut")["TotalCapacityAnnual"]
        .sum()
        .rename("gas_cap_2050")
        .reset_index()
    )

    all_scen = out["Scen_fut"].dropna().unique()
    gas2050 = (
        gas2050.set_index("Scen_fut")
        .reindex(all_scen, fill_value=0.0)
        .reset_index()
    )

    eps = 1e-6
    gas2050["gas_present_2050"] = (gas2050["gas_cap_2050"].abs() > eps).astype(int)

    return gas2050[["Scen_fut", "gas_cap_2050", "gas_present_2050"]]

# ---- PRIM (EMA) runner -----------------------------------------------------

def run_prim_ema(df: pd.DataFrame, feature_cols: list[str], outdir: str) -> None:
    try:
        from ema_workbench.analysis import prim as ema_prim
    except Exception as e:
        raise SystemExit(
            "PRIM_EMA requires `ema_workbench` to be installed "
            "(pip install ema_workbench).\n"
            f"Original import error: {e}"
        )

    FEATURE_LABELS = {
    "capex_geo_2050_mean": "Geothermal\n[USD/kW]",
    "gas_price_2050_median_nonzero": "Gas price\n[USD/GJ]",
    "capex_impngs_2050_mean": "Gas terminal\n[MUSD/PJ/year]",
    "capex_bess_2050_mean": "BESS\n[USD/kW]",
    "capex_solar_2050_mean": "Solar\n[USD/kW]",
    "capex_wind_2050_mean": "Wind\n[USD/kW]",
    "discount_rate_batt_2050": "BESS DR\n[-]",
    "maxcap_geo006_2050_mean": "Max geothermal capacity\n[GW]",
    "discount_rate_solar_2050": "Solar DR\n[-]",
    "discount_rate_wind_2050": "Wind DR\n[-]",
    "discount_rate_impngs_2050": "Gas terminal DR\n[-]",
    "discount_rate_geo_2050": "Geothermal DR\n[-]",
        }
    
    # Swap default colours (e.g. reverse first two colours)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    swapped_colors = [default_colors[1], default_colors[0]] + default_colors[2:]
    
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=swapped_colors)

    os.makedirs(outdir, exist_ok=True)

    dataset_path = os.path.join(outdir, f"prim_ema_dataset_gas_present_{YEAR}.csv")
    df.to_csv(dataset_path, index=False)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["gas_present_2050"].values  # 1 = gas present (cases of interest)

    print("Running PRIM (EMA) for GAS PRESENT in 2050...")
    # Rename columns for plotting only
    X_plot = X.rename(columns=FEATURE_LABELS) 
    p = ema_prim.Prim(X_plot, y, threshold=0.3)
    box = p.find_box()

    # Tradeoff plot
    try:
        fig = box.show_tradeoff()
        if fig is not None and plt is not None:
            trade_png = os.path.join(outdir, f"prim_ema_tradeoff_gas_present_{YEAR}.png")
            fig.savefig(trade_png, dpi=200, bbox_inches="tight")
    except Exception as e:
        print("PRIM (EMA) tradeoff plot failed:", e)


    # Pairwise scatter plot
    try:
        g = box.show_pairs_scatter()
        if g is not None:
            # seaborn PairGrid
            if hasattr(g, "_legend") and g._legend is not None:
                g._legend.remove()
        if g is not None:
            pairs_png = os.path.join(outdir, f"prim_ema_pairs_gas_present_{YEAR}.png")
            g.fig.savefig(pairs_png, dpi=200, bbox_inches="tight")
    except Exception as e:
        print("PRIM (EMA) pairs plot failed:", e)
        
        
    # # Pairwise scatter plot
    # try:
    #     g = box.show_pairs_scatter()
    
    #     if g is not None:
    #         n_vars = len(g.x_vars)
    
    #         for row in range(n_vars):
    #             for col in range(n_vars):
    #                 ax = g.axes[row, col]
    
    #                 # Diagonal panels
    #                 if row == col:
    #                     ax.set_yticks([])
    #                     ax.tick_params(axis="y", left=False, labelleft=False)
    
    #                     # First diagonal panel sits in first column:
    #                     # keep the original left-side y-label there
    #                     if col == 0:
    #                         pass
    #                     else:
    #                         ax.set_ylabel("Density")
    #                         ax.yaxis.set_label_position("left")
    #                         ax.yaxis.label.set_visible(True)
    
    #                 # First column off-diagonal panels:
    #                 # keep original y-ticks and original y-label
    #                 elif col == 0:
    #                     ax.tick_params(axis="y", left=True, labelleft=True)
    
    #                 # All other off-diagonal panels:
    #                 # leave unchanged
    #                 else:
    #                     pass
    
    #         # Remove legend
    #         if hasattr(g, "_legend") and g._legend is not None:
    #             g._legend.remove()
    
    #         pairs_png = os.path.join(outdir, f"prim_ema_pairs_gas_present_{YEAR}.png")
    #         g.fig.savefig(pairs_png, dpi=200, bbox_inches="tight")
    
    # except Exception as e:
    #     print("PRIM (EMA) pairs plot failed:", e)

    # Box limits & stats
    limits_path = os.path.join(outdir, f"prim_ema_box_limits_gas_present_{YEAR}.csv")
    stats_path = os.path.join(outdir, f"prim_ema_box_stats_gas_present_{YEAR}.csv")
    try:
        data = box.inspect(style="data")
        if isinstance(data, list) and data:
            stats, box_lims = data[-1]
            pd.DataFrame([stats]).to_csv(stats_path, index=False)
            box_lims.to_csv(limits_path, index=False)
    except Exception as e:
        print("PRIM (EMA) inspect failed:", e)

    print("PRIM (EMA) artifacts written to:", outdir)

# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run PRIM_EMA on gas-present-2050 outcome."
    )
    parser.add_argument(
        "--inputs",
        default=INPUTS_PATH_DEFAULT,
        help=f"Path to OSeMOSYS inputs (default: {INPUTS_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--outputs",
        default=OUTPUTS_PATH_DEFAULT,
        help=f"Path to OSeMOSYS outputs (default: {OUTPUTS_PATH_DEFAULT})",
    )
    args = parser.parse_args()

    inputs = read_table(args.inputs)
    outputs = read_table(args.outputs)

    X = build_features(inputs)
    ydf = build_outcome_gas_present_2050(outputs)

    df = X.merge(ydf, on="Scen_fut", how="inner").dropna()

    if df.empty:
        raise SystemExit("No scenarios with complete features/outcomes after merging.")

    feature_cols = [
        c for c in df.columns
        if c not in ("Scen_fut", "gas_cap_2050", "gas_present_2050")
    ]

    n = len(df)
    n_gas = int(df["gas_present_2050"].sum())
    n_no_gas = n - n_gas
    print(
        f"Scenarios: {n}  —  gas_present_2050 = 1 in {n_gas} ({n_gas / n:.1%}); "
        f"0 in {n_no_gas} ({n_no_gas / n:.1%})"
    )
    print("Features used:", feature_cols)

    outdir = os.path.join(OUTDIR_BASE, "prim_ema")
    run_prim_ema(df, feature_cols, outdir)


if __name__ == "__main__":
    main()