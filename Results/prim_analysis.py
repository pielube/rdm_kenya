# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 16:51:28 2025

@author: ucbvplu
"""

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


def build_features(inputs: pd.DataFrame) -> pd.DataFrame:
    if "Scen_fut" not in inputs.columns:
        raise ValueError("Expected 'Scen_fut' key column in inputs.")

    tech = inputs["TECHNOLOGY"].astype(str)
    com = inputs["COMMODITY"].astype(str)

    wind_mask = tech.str.startswith("PWRWND")
    solar_mask = tech.str.startswith("PWRSOL")
    pwr_mask = tech.str.startswith("PWR")
    gas_imp_mask = tech == "IMPNGS"
    coal_imp_mask = tech == "IMPCOA"

    feats = []
    for scen, df in inputs.groupby("Scen_fut", sort=False):
        f = {"Scen_fut": scen}

        sub = df
        f["gas_price_med"] = _to_num(sub.loc[gas_imp_mask, "VariableCost"]).median()
        f["coal_price_med"] = _to_num(sub.loc[coal_imp_mask, "VariableCost"]).median()
        f["wind_cf_mean"] = _to_num(sub.loc[wind_mask, "CapacityFactor"]).mean()
        f["solar_cf_mean"] = _to_num(sub.loc[solar_mask, "CapacityFactor"]).mean()
        therm_mask = pwr_mask & ~wind_mask & ~solar_mask
        f["thermal_avail_mean"] = _to_num(sub.loc[therm_mask, "AvailabilityFactor"]).mean()
        disc = _to_num(sub["DiscountRate"]).dropna()
        f["discount_rate"] = disc.iloc[0] if not disc.empty else np.nan

        for prefix, name in [("PWRWND", "capex_wind_2030"), ("PWRSOL", "capex_solar_2030")]:
            mask = sub["TECHNOLOGY"].astype(str).str.startswith(prefix)
            df_cap = sub.loc[mask]
            if df_cap.empty:
                f[name] = np.nan
            elif (df_cap["YEAR"] == 2030).any():
                f[name] = _to_num(df_cap.loc[df_cap["YEAR"] == 2030, "CapitalCost"]).median()
            else:
                f[name] = _to_num(df_cap["CapitalCost"]).median()

        feats.append(f)

    features = pd.DataFrame(feats)

    features = features.dropna(axis=1, how="all")

    return features


# ---- Response engineering --------------------------------------------------

def build_response(outputs: pd.DataFrame) -> pd.DataFrame:
    if "Scen_fut" not in outputs.columns:
        raise ValueError("Expected 'Scen_fut' key column in outputs.")

    out = outputs.copy()
    for c in ["CapitalInvestment", "AnnualFixedOperatingCost", "AnnualVariableOperatingCost"]:
        if c in out.columns:
            out[c] = _to_num(out[c])
        else:
            out[c] = 0.0

    syscost = (
        out.groupby("Scen_fut")[
            ["CapitalInvestment", "AnnualFixedOperatingCost", "AnnualVariableOperatingCost"]
        ]
        .sum()
        .sum(axis=1)
        .rename("total_system_cost")
        .reset_index()
    )
    return syscost


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

    X = build_features(inputs)
    y_df = build_response(outputs)
    df = X.merge(y_df, on="Scen_fut", how="inner").dropna()

    if df.empty:
        raise SystemExit("No scenarios with complete features/outcomes after merging.")

    thr = df["total_system_cost"].quantile(QUANTILE)

    dataset_path = os.path.join(OUTDIR, "prim_dataset.csv")
    df.to_csv(dataset_path, index=False)

    feature_cols = [c for c in df.columns if c not in ("Scen_fut", "total_system_cost")]
    Xnum = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Prepared {len(df)} scenarios; running PRIM with threshold = {thr:,.3f}.")
    run_prim(Xnum, df["total_system_cost"], thr, OUTDIR)

    print("Artifacts written to:")
    print(f"- {dataset_path}")
    print(f"- {os.path.join(OUTDIR, 'prim_tradeoff.png')} (if plotting succeeded)")
    print(f"- {os.path.join(OUTDIR, 'prim_pairs.png')} (if plotting succeeded)")
    print(f"- {os.path.join(OUTDIR, 'prim_box_limits.csv')} (if extraction succeeded)")
    print(f"- {os.path.join(OUTDIR, 'prim_box_stats.csv')} (if extraction succeeded)")
    
    plt.show()
