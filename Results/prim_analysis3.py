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


# def build_features(inputs: pd.DataFrame) -> pd.DataFrame:
#     """Compute exactly the three features you specified.

#     Feature 1: SpecifiedAnnualDemand in 2050, summed over all technologies.
#     Feature 2: CapitalCost in 2050, averaged over all PWRGEO* technologies.
#     Feature 3: VariableCost of IMPNGS in 2050, after removing 0s and NaNs (median).
#     """
#     if "Scen_fut" not in inputs.columns:
#         raise ValueError("Expected 'Scen_fut' key column in inputs.")

#     # Ensure numeric helpers
#     def to_num(s):
#         return pd.to_numeric(s, errors="coerce")

#     feats = []
#     for scen, sub in inputs.groupby("Scen_fut", sort=False):
#         f = {"Scen_fut": scen}
#         # 2050 slice
#         sub50 = sub.loc[sub.get("YEAR").eq(2050) if "YEAR" in sub.columns else []]

#         # Feature 1: demand_2050_sum
#         if "SpecifiedAnnualDemand" in sub50.columns:
#             f["demand_2050_sum"] = to_num(sub50["SpecifiedAnnualDemand"]).sum()
#         else:
#             f["demand_2050_sum"] = float("nan")

#         # Feature 2: capex_geo_2050_mean over PWRGEO*
#         if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
#             geo50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRGEO")]
#             f["capex_geo_2050_mean"] = to_num(geo50["CapitalCost"]).mean() if not geo50.empty else float("nan")
#         else:
#             f["capex_geo_2050_mean"] = float("nan")

#         # Feature 3: gas_price_2050_median_nonzero for IMPNGS
#         if {"TECHNOLOGY", "VariableCost"}.issubset(sub50.columns):
#             gas50 = sub50.loc[sub50["TECHNOLOGY"].astype(str) == "IMPNGS", "VariableCost"]
#             gas50 = to_num(gas50)
#             gas50 = gas50[(~gas50.isna()) & (gas50 != 0)]
#             f["gas_price_2050_median_nonzero"] = gas50.median() if not gas50.empty else float("nan")
#         else:
#             f["gas_price_2050_median_nonzero"] = float("nan")

#         feats.append(f)

#     X = pd.DataFrame(feats)

#     # Drop all-NaN columns; keep Scen_fut
#     keep_cols = [c for c in X.columns if c == "Scen_fut" or X[c].notna().any()]
#     X = X[keep_cols]

#     # Quick variability report
#     nunique = X.drop(columns=["Scen_fut"]).nunique(dropna=True)
#     print("Feature variability (nunique):")
#     print(nunique.sort_values())

#     return X


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
        
        # # (2) CapitalCost in 2050 over PWRGEO*
        # if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
        #     geo50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRGEO")]
        #     vals = to_num(geo50["CapitalCost"])
        #     vals = vals[(~vals.isna()) & (vals != 0)]
        #     f["capex_geo_2050_mean"] = vals.mean() if not vals.empty else np.nan
        # else:
        #     f["capex_geo_2050_mean"] = np.nan
    
        # (3) VariableCost of IMPNGS in 2050 (drop 0 and NaN), median
        # if {"TECHNOLOGY", "VariableCost"}.issubset(sub50.columns):
        #     gas50 = sub50.loc[sub50["TECHNOLOGY"].astype(str) == "IMPNGS", "VariableCost"]
        #     gas50 = to_num(gas50)
        #     gas50 = gas50[(~gas50.isna()) & (gas50 != 0)]
        #     f["gas_price_2050_median_nonzero"] = gas50.median() if not gas50.empty else np.nan
        # else:
        #     f["gas_price_2050_median_nonzero"] = np.nan
            
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
            f["capex_bess_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_bess_2050_mean"] = np.nan
            
        # # (6) CapitalCost in 2050 over PWRSOL (solar)
        # if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
        #     nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL")]
        #     vals = to_num(nuc50["CapitalCost"])
        #     vals = vals[(~vals.isna()) & (vals != 0)]
        #     f["capex_solar_2050_mean"] = vals.mean() if not vals.empty else np.nan
        # else:
        #     f["capex_solar_2050_mean"] = np.nan
            
        # # (7) CapitalCost in 2050 over PWRWND (wind)
        # if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
        #     nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL")]
        #     vals = to_num(nuc50["CapitalCost"])
        #     vals = vals[(~vals.isna()) & (vals != 0)]
        #     f["capex_wind_2050_mean"] = vals.mean() if not vals.empty else np.nan
        # else:
        #     f["capex_wind_2050_mean"] = np.nan
        
        # # (8) Global discount rate (DiscountRate)
        # if "DiscountRate" in sub.columns:
        #     vals = to_num(sub["DiscountRate"])
        #     vals = vals[(~vals.isna()) & (vals != 0)]
        #     f["discount_rate_global"] = vals.iloc[0] if not vals.empty else np.nan
        # else:
        #     f["discount_rate_global"] = np.nan
        
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
            f["discount_rate_batt_2050"] = vals.mean() if not vals.empty else np.nan
        else:
            f["discount_rate_batt_2050"] = np.nan
    
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
        .rename("gas_cap_2050")
        .reset_index()
    )

    # If a scenario has no matching rows, the groupby will omit it; fill with 0
    all_scen = out["Scen_fut"].dropna().unique()
    gas2050 = gas2050.set_index("Scen_fut").reindex(all_scen, fill_value=0.0).reset_index()

    eps = 1e-6
    gas2050["no_gas_2050"] = (gas2050["gas_cap_2050"].abs() <= eps).astype(int)

    return gas2050[["Scen_fut", "gas_cap_2050", "no_gas_2050"]]


# ---- PRIM run --------------------------------------------------------------


def run_prim(x: pd.DataFrame, y: pd.Series, threshold: float, outdir: str) -> None:
    p = prim.Prim(x, y, threshold=threshold, threshold_type=">")
    box = p.find_box()

    try:
        fig = box.show_tradeoff()
        if fig is not None and plt is not None:
            fig.savefig(os.path.join(outdir, "prim_tradeoff.png"), bbox_inches="tight", dpi=200)
    except Exception:
        warnings.warn("Could not create tradeoff plot; continuing.")

    try:
        g = box.show_pairs_scatter()
        if g is not None:
            g.fig.savefig(os.path.join(outdir, "prim_pairs.png"), bbox_inches="tight", dpi=200)
    except Exception:
        warnings.warn("Could not create pairs scatter; continuing.")

    limits_path = os.path.join(outdir, "prim_box_limits.csv")
    stats_path = os.path.join(outdir, "prim_box_stats.csv")
    try:
        data = box.inspect(style="data")  # returns list of (stats, box_lims)
        if isinstance(data, list) and data:
            stats, box_lims = data[-1]
            pd.DataFrame([stats]).to_csv(stats_path, index=False)
            box_lims.to_csv(limits_path, index=False)
        else:
            raise AttributeError
    except Exception:
        try:
            if hasattr(box, "peeling_trajectory"):
                pt = box.peeling_trajectory
                pt.to_csv(stats_path, index=False)
        except Exception:
            pass
        try:
            if hasattr(p, "boxes"):
                pd.DataFrame(p.boxes[-1]).to_csv(limits_path, index=False)
        except Exception:
            pass


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

    # Inspect class balance
    n = len(df)
    n_no = int(df["no_gas_2050"].sum())
    n_yes = n - n_no
    print(f"Scenarios: {n}  —  no_gas_2050 = 1 in {n_no} ({n_no/n:.1%}); 0 in {n_yes} ({n_yes/n:.1%})")

    # Save dataset
    dataset_path = os.path.join(OUTDIR, "prim_dataset_no_gas_2050.csv")
    df.to_csv(dataset_path, index=False)

    # Run PRIM: treat y as numeric {0,1} and look for boxes with y > 0.5
    feature_cols = [c for c in df.columns if c not in ("Scen_fut", "gas_cap_2050", "no_gas_2050")]
    Xnum = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print("Running PRIM to find conditions (boxes) associated with NO gas in 2050…")
    p = prim.Prim(Xnum, df["no_gas_2050"], threshold=0.5, threshold_type=">")
    box = p.find_box()

    # Plots
    try:
        fig = box.show_tradeoff()
        if fig is not None and plt is not None:
            fig.savefig(os.path.join(OUTDIR, "prim_tradeoff_no_gas_2050.png"), bbox_inches="tight", dpi=200)
    except Exception:
        warnings.warn("Could not create tradeoff plot; continuing.")

    try:
        g = box.show_pairs_scatter()
        if g is not None:
            g.fig.savefig(os.path.join(OUTDIR, "prim_pairs_no_gas_2050.png"), bbox_inches="tight", dpi=200)
    except Exception:
        warnings.warn("Could not create pairs scatter; continuing.")

    # Export box limits/statistics if available
    limits_path = os.path.join(OUTDIR, "prim_box_limits_no_gas_2050.csv")
    stats_path = os.path.join(OUTDIR, "prim_box_stats_no_gas_2050.csv")
    try:
        data = box.inspect(style="data")
        if isinstance(data, list) and data:
            stats, box_lims = data[-1]
            pd.DataFrame([stats]).to_csv(stats_path, index=False)
            box_lims.to_csv(limits_path, index=False)
    except Exception:
        pass

    print("Artifacts written to:")
    print(f"- {dataset_path}")
    print(f"- {os.path.join(OUTDIR, 'prim_tradeoff_no_gas_2050.png')} (if plotting succeeded)")
    print(f"- {os.path.join(OUTDIR, 'prim_pairs_no_gas_2050.png')} (if plotting succeeded)")
    print(f"- {limits_path} (if extraction succeeded)")
    print(f"- {stats_path} (if extraction succeeded)")
