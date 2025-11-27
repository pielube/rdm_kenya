"""
PRIM example for RDM using your OSeMOSYS inputs/outputs.

What this script does
---------------------
1) Reads the two CSVs you shared (paths below).
2) Builds scenario-level features (uncertainties) from the inputs (gas & coal price proxies, wind/solar CFs, etc.).
3) Aggregates the outputs to a scalar performance metric per scenario: total system cost.
4) Runs PRIM (Project-Platypus `prim` package) to discover vulnerable regions (high-cost cases).
5) Writes artifacts (dataset + box limits + plots).

References
----------
- Friedman & Fisher (1999) introduced PRIM (Patient Rule Induction Method): Statistics and Computing 9:123–143. doi:10.1023/A:1008894516817.
- EMA Workbench PRIM API docs (class `Prim`, `PrimBox` methods incl. `inspect`, `show_tradeoff`, `show_pairs_scatter`).
- Project-Platypus/PRIM README for Python usage (e.g., `p = prim.Prim(...); box = p.find_box(); box.show_tradeoff()`).

Usage
-----
- Ensure the PRIM library is installed: `pip install prim` (the Project-Platypus package).
- Adjust the `INPUTS_CSV` and `OUTPUTS_CSV` paths if needed.
- Run: `python PRIM_RDM_OSeMOSYS_example.py`

Notes
-----
- With very few scenarios, PRIM can only yield coarse boxes. Consider adding more futures if possible.
- This script is defensive against small API differences between EMA Workbench PRIM and Project-Platypus PRIM.
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
from typing import Tuple


# Matplotlib is only used for saving figures if available
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    plt = None

# Try to import Project-Platypus PRIM
try:
    import prim  # Project-Platypus PRIM package
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "The 'prim' package is required. Install with `pip install prim`.\n"
        f"Original import error: {e}"
    )

# ---- Configuration ---------------------------------------------------------
INPUTS_CSV = "OSEMOSYS_Energy_Input.csv"
OUTPUTS_CSV = "OSEMOSYS_Energy_Output.csv"
OUTDIR = "prim_artifacts"
QUANTILE = 0.90  # cost quantile to define 'cases of interest' (high cost)
RANDOM_STATE = 42

os.makedirs(OUTDIR, exist_ok=True)

# ---- Feature engineering ---------------------------------------------------

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _mean_over_years(df: pd.DataFrame, col: str, years: Tuple[int, int] | None = None) -> float:
    if years is not None and {"YEAR", col}.issubset(df.columns):
        y0, y1 = years
        df = df.loc[(df["YEAR"] >= y0) & (df["YEAR"] <= y1)]
    return _to_num(df[col]).mean()


def _median_at_year_else_median(df: pd.DataFrame, col: str, year: int) -> float:
    if "YEAR" in df.columns and (df["YEAR"] == year).any():
        return _to_num(df.loc[df["YEAR"] == year, col]).median()
    return _to_num(df[col]).median()


def build_features(inputs: pd.DataFrame) -> pd.DataFrame:
    if "Scen_fut" not in inputs.columns:
        raise ValueError("Expected 'Scen_fut' key column in inputs.")
    
    
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")
    
    
    feats = []
    for scen, sub in inputs.groupby("Scen_fut", sort=False):
        f = {"Scen_fut": scen}
        sub50 = sub.loc[sub.get("YEAR").eq(2050) if "YEAR" in sub.columns else []]
    
    
        # (1) SpecifiedAnnualDemand in 2050 summed
        f["demand_2050_sum"] = to_num(sub50.get("SpecifiedAnnualDemand", np.nan)).sum()
        
        # (2) CapitalCost in 2050 over PWRGEO*
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            geo50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRGEO")]
            vals = to_num(geo50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["Geothermal cost\n[USD/kW]"] = vals.mean() if not vals.empty else np.nan
        else:
            f["Geothermal cost\n[USD/kW]"] = np.nan
    
        # (3) VariableCost of IMPNGS in 2050 (drop 0 and NaN), median
        if {"TECHNOLOGY", "VariableCost"}.issubset(sub50.columns):
            gas50 = sub50.loc[sub50["TECHNOLOGY"].astype(str) == "IMPNGS", "VariableCost"]
            gas50 = to_num(gas50)
            gas50 = gas50[(~gas50.isna()) & (gas50 != 0)]
            f["Gas price\n[MUSD/PJ]"] = gas50.median() if not gas50.empty else np.nan
        else:
            f["Gas price\n[MUSD/PJ]"] = np.nan
            
        # # (4) CapitalCost in 2050 over PWRURN (nuclear)
        # if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
        #     nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRURN")]
        #     vals = to_num(nuc50["CapitalCost"])
        #     vals = vals[(~vals.isna()) & (vals != 0)]
        #     f["capex_nuc_2050_mean"] = vals.mean() if not vals.empty else np.nan
        # else:
        #     f["capex_nuc_2050_mean"] = np.nan
            
        # (5) CapitalCost in 2050 over BESS_TECH (batteries)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("BESS_TECH")]
            vals = to_num(nuc50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["BESS cost\n[USD/kW]"] = vals.mean() if not vals.empty else np.nan
        else:
            f["BESS cost\n[USD/kW]"] = np.nan
            
        # (6) CapitalCost in 2050 over PWRSOL (solar)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL")]
            vals = to_num(nuc50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["PV cost\n[USD/kW]"] = vals.mean() if not vals.empty else np.nan
        else:
            f["PV cost\n[USD/kW]"] = np.nan
            
        # # (7) CapitalCost in 2050 over PWRWND (wind)
        # if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
        #     nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL")]
        #     vals = to_num(nuc50["CapitalCost"])
        #     vals = vals[(~vals.isna()) & (vals != 0)]
        #     f["capex_wind_2050_mean"] = vals.mean() if not vals.empty else np.nan
        # else:
        #     f["capex_wind_2050_mean"] = np.nan
        
        # (8) Global discount rate (DiscountRate)
        if "DiscountRate" in sub.columns:
            vals = to_num(sub["DiscountRate"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["Global DR\n[-]"] = vals.iloc[0] if not vals.empty else np.nan
        else:
            f["Global DR\n[-]"] = np.nan
        
        # # (9) Technology-specific discount rate (DiscountRateIdv), averaged over time & techs
        # if "DiscountRateIdv" in sub.columns:
        #     vals = to_num(sub["DiscountRateIdv"])
        #     vals = vals[(~vals.isna()) & (vals != 0)]
        #     f["discount_rate_idv_mean"] = vals.mean() if not vals.empty else np.nan
        # else:
        #     f["discount_rate_idv_mean"] = np.nan
        
        # (10) DiscountRateIdv for batteries (BESS_TECH) in 2050
        if {"TECHNOLOGY", "YEAR", "DiscountRateIdv"}.issubset(sub.columns):
            bat50 = sub.loc[(sub["TECHNOLOGY"].astype(str).str.startswith("BESS_TECH")) & (sub["YEAR"] == 2050)]
            vals = to_num(bat50["DiscountRateIdv"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["BESS DR\n[-]"] = vals.mean() if not vals.empty else np.nan
        else:
            f["BESS DR\n[-]"] = np.nan
    
        feats.append(f)
    
    X = pd.DataFrame(feats)
    
    # Drop all-NaN columns; keep Scen_fut
    keep_cols = [c for c in X.columns if c == "Scen_fut" or X[c].notna().any()]
    X = X[keep_cols]

    # Quick variability report
    nunique = X.drop(columns=["Scen_fut"]).nunique(dropna=True)
    print("Feature variability (nunique):")
    print(nunique.sort_values())
    
    return X


# ---- Outcome engineering: define cases with NO gas capacity in 2050 ---------

def build_outcome_no_gas_2050(outputs: pd.DataFrame) -> pd.DataFrame:
    """
    Build a binary outcome per scenario: 1 if *no* natural gas capacity is
    installed in 2050 for TECHNOLOGY == 'PWRNGS001', else 0.

    We read 'TotalCapacityAnnual' from the outputs and sum over all rows that
    match (YEAR==2050, TECHNOLOGY=='PWRNGS001'). If the sum is ~0, we flag the
    scenario as 'no_gas_2050' = 1.
    """
    required_cols = {"Scen_fut", "YEAR", "TECHNOLOGY", "TotalCapacityAnnual"}
    missing = required_cols - set(outputs.columns)
    if missing:
        raise ValueError(f"Outputs missing required columns: {sorted(missing)}")

    out = outputs.copy()
    out["TotalCapacityAnnual"] = _to_num(out["TotalCapacityAnnual"]).fillna(0.0)

    gas2050 = (
        out.loc[(out["YEAR"] == 2050) & (out["TECHNOLOGY"].astype(str) == "PWRNGS001")]
        .groupby("Scen_fut")["TotalCapacityAnnual"].sum()
        .rename("gas_cap")
        .reset_index()
    )

    # If a scenario has no matching rows, the groupby will omit it; fill with 0
    all_scen = out["Scen_fut"].dropna().unique()
    gas2050 = gas2050.set_index("Scen_fut").reindex(all_scen, fill_value=0.0).reset_index()

    eps = 1e-6
    gas2050["no_gas"] = (gas2050["gas_cap"].abs() <= eps).astype(int)

    return gas2050[["Scen_fut", "gas_cap", "no_gas"]]



# ----------------------------------------------------------------------
# EMA-PRIM analysis
# ----------------------------------------------------------------------

from ema_workbench.analysis import prim as ema_prim
PRIM_OUTDIR = "prim_artifacts_ema"
YEAR = 2050  # <-- change this single value to analyse a different year


def run_prim_ema(df: pd.DataFrame, feature_cols):
    """Run EMA Workbench PRIM on the same features/outcome."""
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["no_gas"].values  # 1 = no gas (cases of interest)

    # Prim object
    p = ema_prim.Prim(X, y, threshold=0.5)  # focus on high no-gas
    box = p.find_box()

    # Tradeoff plot
    try:
        fig = box.show_tradeoff()
        if fig is not None:
            trade_png = os.path.join(PRIM_OUTDIR, f"prim_tradeoff_no_gas_{YEAR}.png")
            fig.savefig(trade_png, dpi=200, bbox_inches="tight")
    except Exception as e:
        print("PRIM tradeoff plot failed:", e)

    # Pairwise scatter plot
    try:
        g = box.show_pairs_scatter()
        if g is not None:
            pairs_png = os.path.join(PRIM_OUTDIR, f"prim_pairs_no_gas_{YEAR}.png")
            g.fig.savefig(pairs_png, dpi=200, bbox_inches="tight")
    except Exception as e:
        print("PRIM pairs plot failed:", e)

    # Box limits & stats (if available)
    limits_path = os.path.join(PRIM_OUTDIR, f"prim_box_limits_no_gas_{YEAR}.csv")
    stats_path = os.path.join(PRIM_OUTDIR, f"prim_box_stats_no_gas_{YEAR}.csv")
    try:
        data = box.inspect(style="data")  # list of (stats, box_lims)
        if isinstance(data, list) and data:
            stats, box_lims = data[-1]
            pd.DataFrame([stats]).to_csv(stats_path, index=False)
            box_lims.to_csv(limits_path, index=False)
    except Exception as e:
        print("PRIM inspect failed:", e)

    print("PRIM (EMA) artifacts written to:", PRIM_OUTDIR)
    
    
# ---- Main ------------------------------------------------------------------
if __name__ == "__main__":
    inputs = pd.read_csv(INPUTS_CSV, low_memory=False)
    outputs = pd.read_csv(OUTPUTS_CSV, low_memory=False)

    # Features (uncertainties / conditions)
    X = build_features(inputs)

    # Outcome: NO gas in 2050
    y_gas = build_outcome_no_gas_2050(outputs)

    # Merge and drop scenarios with missing features
    df = X.merge(y_gas, on="Scen_fut", how="inner").dropna()

    if df.empty:
        raise SystemExit("No scenarios with complete features/outcomes after merging.")
    
    #############
    #############
    
    # dataset_path = os.path.join(CART_OUTDIR, f"dataset_no_gas_{YEAR}.csv")
    # df.to_csv(dataset_path, index=False)

    feature_cols = [c for c in df.columns if c not in ("Scen_fut", "gas_cap", "no_gas")]

    print(f"Scenarios: {len(df)}, features: {len(feature_cols)}")
    print("Feature variability (nunique):")
    print(df[feature_cols].nunique(dropna=True).sort_values())

    # # Run CART
    # run_cart(df, feature_cols)

    # Run EMA-PRIM
    run_prim_ema(df, feature_cols)

    print("Done.")



