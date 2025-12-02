
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker


PLOT_DIR = "plots"   # if you want to save figures in the same folder as plots.py
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------- Helpers copied from your original logic ----------

def capital_recovery_factor_original(r: float, n: float) -> float:
    """
    CRF as in your original script:

      CRF = (1 - (1+r)^-1) / (1 - (1+r)^-n)

    with r as decimal and n in years.
    """
    try:
        r = float(r)
        n = float(n)
    except Exception:
        return 0.0
    if n <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return 1.0 / n
    one_plus_r = 1.0 + r
    num = 1.0 - one_plus_r ** (-1.0)
    den = 1.0 - one_plus_r ** (-n)
    if den == 0.0:
        return 0.0
    return num / den


def read_operational_life(df_input: pd.DataFrame) -> pd.Series:
    """
    Exact analogue of _read_operational_life_from_parquet:
    TECHNOLOGY -> OperationalLife (max per tech), keeping zeros and dropping NaNs.
    """
    if "TECHNOLOGY" not in df_input.columns or "OperationalLife" not in df_input.columns:
        return pd.Series(dtype="float64")

    sub = df_input[["TECHNOLOGY", "OperationalLife"]].copy()
    sub["OperationalLife"] = pd.to_numeric(sub["OperationalLife"], errors="coerce")
    sub = sub[sub["OperationalLife"].notna()]
    if sub.empty:
        return pd.Series(dtype="float64")

    sub["TECHNOLOGY"] = sub["TECHNOLOGY"].astype(str).str.strip()
    s = sub.groupby("TECHNOLOGY", sort=False)["OperationalLife"].max()
    return s


def read_discount_rate_by_tech_year(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Analogue of _read_discount_rate_by_tech_year:
    returns DataFrame with ['TECHNOLOGY','YEAR','__rate_by_ty'].
    """
    if df_input is None:
        return None

    used_col = None
    if "DiscountRateIdv" in df_input.columns:
        used_col = "DiscountRateIdv"
    elif "DiscountRate" in df_input.columns:
        used_col = "DiscountRate"
    else:
        return None

    need_cols = ["TECHNOLOGY", "YEAR", used_col]
    if "TIMESLICE" in df_input.columns:
        need_cols.append("TIMESLICE")

    sub = df_input[need_cols].copy()
    sub["TECHNOLOGY"] = sub["TECHNOLOGY"].astype(str).str.strip()
    sub["YEAR"] = pd.to_numeric(sub["YEAR"], errors="coerce")
    sub[used_col] = pd.to_numeric(sub[used_col], errors="coerce")

    # filter TIMESLICE == empty/0 as in original
    if "TIMESLICE" in sub.columns:
        ts = sub["TIMESLICE"]
        mask_ts = ts.isna() | (pd.to_numeric(ts, errors="coerce") == 0) | (ts.astype(str).str.strip() == "")
        sub = sub[mask_ts]

    sub = sub.dropna(subset=["TECHNOLOGY", "YEAR", used_col])
    if sub.empty:
        return None

    sub["YEAR"] = sub["YEAR"].astype(int)

    # last non-zero preferred, else last non-null
    any_last = sub.groupby(["TECHNOLOGY", "YEAR"], sort=False)[used_col].last()
    nz = sub[sub[used_col] != 0]
    if not nz.empty:
        nz_last = nz.groupby(["TECHNOLOGY", "YEAR"], sort=False)[used_col].last()
        rate_ser = nz_last.reindex(any_last.index)
        rate_ser = rate_ser.where(rate_ser.notna(), any_last)
    else:
        rate_ser = any_last

    grp = rate_ser.reset_index().rename(columns={used_col: "__rate_by_ty"})
    if grp.empty:
        return None
    return grp[["TECHNOLOGY", "YEAR", "__rate_by_ty"]]


# ---------- Core: compute LCOE per Future.ID, YEAR ----------

def compute_lcoe_from_csv(input_csv: str,
                          output_csv: str,
                          fallback_discount_rate: float = 0.07) -> pd.DataFrame:
    """
    Apply the same AIC + rolling logic as your original function
    to the CSV files and return a DataFrame:

        ['Future.ID', 'YEAR', 'LCOE', 'Strategy']
    """

    # --- Read CSVs ---
    df_in_all = pd.read_csv(input_csv, low_memory=False)
    df_out_all = pd.read_csv(output_csv, low_memory=False)

    # Remove BACKSTOP everywhere
    if "TECHNOLOGY" in df_in_all.columns:
        df_in_all = df_in_all[df_in_all["TECHNOLOGY"].astype(str) != "BACKSTOP"]
    if "TECHNOLOGY" in df_out_all.columns:
        df_out_all = df_out_all[df_out_all["TECHNOLOGY"].astype(str) != "BACKSTOP"]

    # Demand (Future.ID, YEAR) from input
    dem = df_in_all[["Future.ID", "YEAR", "SpecifiedAnnualDemand"]].copy()
    dem["YEAR"] = pd.to_numeric(dem["YEAR"], errors="coerce")
    dem["SpecifiedAnnualDemand"] = pd.to_numeric(dem["SpecifiedAnnualDemand"], errors="coerce")
    dem = dem.dropna(subset=["YEAR", "SpecifiedAnnualDemand"])
    dem["YEAR"] = dem["YEAR"].astype(int)
    demand_by_fy = dem.groupby(["Future.ID", "YEAR"])["SpecifiedAnnualDemand"].sum()

    lcoe_records = []

    # Process each Future.ID separately
    for fid in sorted(df_out_all["Future.ID"].dropna().unique()):
        fid_int = int(fid)

        # Skip futures with no demand at all
        if fid_int not in demand_by_fy.index.get_level_values(0):
            continue

        df_out = df_out_all[df_out_all["Future.ID"] == fid_int].copy()
        df_in = df_in_all[df_in_all["Future.ID"] == fid_int].copy()
        if df_out.empty or df_in.empty:
            continue

        # Strategy label (if present)
        strategy_val = None
        if "Strategy" in df_out.columns:
            strategy_val = df_out["Strategy"].dropna().astype(str).iloc[0]

        # Life & discount maps
        life_by_tech = read_operational_life(df_in)
        rate_by_ty = read_discount_rate_by_tech_year(df_in)

        df = df_out.copy()
        if "YEAR" not in df.columns or "TECHNOLOGY" not in df.columns:
            continue

        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df = df.dropna(subset=["YEAR"])
        df["YEAR"] = df["YEAR"].astype(int)
        tech_series = df["TECHNOLOGY"].astype(str).str.strip()
        year_series = df["YEAR"]

        # Map life
        if life_by_tech is not None and not life_by_tech.empty:
            life_map = tech_series.map(life_by_tech)
        else:
            life_map = pd.Series(np.nan, index=df.index, dtype="float64")

        # Map discount rate per-tech-year
        rate_map = pd.Series(np.nan, index=df.index, dtype="float64")
        if rate_by_ty is not None and not rate_by_ty.empty:
            s_ty = rate_by_ty.set_index(["TECHNOLOGY", "YEAR"])["__rate_by_ty"]
            mask_y = year_series.notna()
            tech_arr = tech_series[mask_y].values
            year_arr = year_series[mask_y].values
            mi = pd.MultiIndex.from_arrays([tech_arr, year_arr])
            mapped = s_ty.reindex(mi)
            rate_map.loc[mask_y] = pd.to_numeric(mapped.values, errors="coerce")

        # Fallback scalar
        rate_map = rate_map.fillna(float(fallback_discount_rate))

        # Capital investment
        cap = pd.to_numeric(df.get("CapitalInvestment", 0.0), errors="coerce").fillna(0.0)

        # Incremental annualised cost
        crf_vals = [
            capital_recovery_factor_original(r, n)
            for r, n in zip(pd.to_numeric(rate_map, errors="coerce"),
                            pd.to_numeric(life_map, errors="coerce"))
        ]
        inc = cap * pd.to_numeric(crf_vals, errors="coerce")

        tmp = pd.DataFrame({
            "TECHNOLOGY": tech_series,
            "YEAR": year_series,
            "life": pd.to_numeric(life_map, errors="coerce"),
            "inc": pd.to_numeric(inc, errors="coerce").fillna(0.0),
        })
        tmp = tmp[tmp["YEAR"].notna()].copy()
        tmp["YEAR"] = tmp["YEAR"].astype(int)

        # Rolling AIC per (Future, Technology)
        aic_records = []
        for tech, g in tmp.groupby("TECHNOLOGY", sort=False):
            g = g.sort_values("YEAR")
            life_vals = pd.to_numeric(g["life"], errors="coerce").dropna()
            if life_vals.empty:
                window = 1
            else:
                window = int(max(1, math.ceil(float(life_vals.max()))))

            per_year = g.groupby("YEAR", sort=True)["inc"].max()
            roll = per_year.rolling(window=window, min_periods=1).sum()

            rec = pd.DataFrame({
                "TECHNOLOGY": tech,
                "YEAR": roll.index.values,
                "AIC": roll.values,
            })
            aic_records.append(rec)

        if aic_records:
            aic_df = pd.concat(aic_records, ignore_index=True)
            aic_by_year = aic_df.groupby("YEAR", sort=True)["AIC"].sum()
        else:
            aic_by_year = pd.Series(dtype="float64")

        # Variable + fixed O&M
        var_col = "AnnualVariableOperatingCost"
        fix_col = "AnnualFixedOperatingCost"
        for col in [var_col, fix_col]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        om_by_year = df.groupby("YEAR", sort=True)[[var_col, fix_col]].sum().sum(axis=1)

        # Total cost
        cost_by_year = aic_by_year.add(om_by_year, fill_value=0.0)

        # Demand for this Future.ID
        dem_f = demand_by_fy.xs(fid_int, level=0)
        years_common = cost_by_year.index.intersection(dem_f.index)
        if years_common.empty:
            continue

        lcoe_f = (cost_by_year.loc[years_common] / dem_f.loc[years_common]).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        for y, val in lcoe_f.items():
            lcoe_records.append({
                "Future.ID": fid_int,
                "YEAR": int(y),
                "LCOE": float(val),
                "Strategy": strategy_val,
            })

    if not lcoe_records:
        # Return empty with correct columns instead of raising
        return pd.DataFrame(columns=["Future.ID", "YEAR", "LCOE", "Strategy"])

    lcoe_df = pd.DataFrame(lcoe_records)
    return lcoe_df.sort_values(["Future.ID", "YEAR"]).reset_index(drop=True)

def plot_line_lcoe_full_horizon(lcoe_df: pd.DataFrame,
                                save: bool = False,
                                filename: str = "line_lcoe_full.png") -> None:
    """
    Plot system-wide LCOE by Future.ID over the full time horizon (2019–2050),
    using the same style as plot_line_lcoe in plots.py:
      - Scenario 0 in blue, others in light grey
      - LCOE converted to USD/kWh (×0.0036)
      - x-ticks every 5 years
    """
    df_lcoe = lcoe_df.copy()

    # Ensure numeric and drop zeros/NaNs
    df_lcoe["LCOE"] = pd.to_numeric(df_lcoe["LCOE"], errors="coerce")
    df_lcoe = df_lcoe.dropna(subset=["LCOE"])
    df_lcoe = df_lcoe.loc[df_lcoe["LCOE"] != 0]

    # Convert to USD/kWh as in plots.py
    df_lcoe["LCOE_kWh"] = df_lcoe["LCOE"] * 0.0036

    plt.figure(figsize=(10, 6))
    for fid, grp in df_lcoe.groupby("Future.ID"):
        grp = grp.sort_values("YEAR")
        if fid == 0:
            plt.plot(grp["YEAR"], grp["LCOE_kWh"],
                     color="blue", linewidth=2, label="Scenario 0")
        else:
            plt.plot(grp["YEAR"], grp["LCOE_kWh"],
                     color="lightgrey", linewidth=1, alpha=0.7)

    ax = plt.gca()
    ax.set_xlim(2019, 2050)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    plt.title("System-wide LCOE by Scenario (2019–2050)")
    plt.xlabel("Year")
    plt.ylabel("LCOE [USD/kWh]")
    plt.legend()
    plt.tight_layout()

    if save:
        out_path = os.path.join(PLOT_DIR, filename)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()



# ---------- Run + plot ----------

if __name__ == "__main__":
    input_path = "OSEMOSYS_Energy_Input.csv"
    output_path = "OSEMOSYS_Energy_Output.csv"

    lcoe = compute_lcoe_from_csv(input_path, output_path, fallback_discount_rate=0.07)

    # quick check
    print(lcoe.head(20).to_string(index=False))

    plot_line_lcoe_full_horizon(lcoe, save=True)
