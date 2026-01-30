"""Postprocessing utilities for RDM outputs."""

from __future__ import annotations

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

CSV_INPUT = "OSEMOSYS_Energy_Input.csv"
CSV_OUTPUT = "OSEMOSYS_Energy_Output.csv"
PLOT_DIR = "plots"
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

    Numerator components per (Future.ID, YEAR):
      - Rolling annualised investment from new builds (AIC)
      - Annual fixed + variable operating cost
      - Annualised investment cost of residual capacity (ResidualAIC)
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

        # Life & discount maps from input
        life_by_tech = read_operational_life(df_in)
        rate_by_ty = read_discount_rate_by_tech_year(df_in)

        # --- AIC from new builds (rolling) ---

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

        # Fallback scalar for missing rates
        rate_map = rate_map.fillna(float(fallback_discount_rate))

        # Capital investment
        cap = pd.to_numeric(df.get("CapitalInvestment", 0.0), errors="coerce").fillna(0.0)

        # Incremental annualised cost per row (build-year AIC)
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

            per_year = g.groupby("YEAR", sort=True)["inc"].max()  # match original: max per year
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

        # --- Variable + fixed O&M by year for this future ---
        var_col = "AnnualVariableOperatingCost"
        fix_col = "AnnualFixedOperatingCost"
        for col in [var_col, fix_col]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        om_by_year = df.groupby("YEAR", sort=True)[[var_col, fix_col]].sum().sum(axis=1)

        # --- Residual capacity component (AnnualizedCapitalInvestmentResidual analogue) ---
        resid_by_year = pd.Series(dtype="float64")
        if "ResidualCapacity" in df_in.columns:
            res = df_in[["TECHNOLOGY", "YEAR", "ResidualCapacity"]].copy()
            res["TECHNOLOGY"] = res["TECHNOLOGY"].astype(str).str.strip()
            res["YEAR"] = pd.to_numeric(res["YEAR"], errors="coerce")
            res["ResidualCapacity"] = pd.to_numeric(res["ResidualCapacity"], errors="coerce")
            res = res.dropna(subset=["YEAR", "ResidualCapacity"])
            res = res[res["YEAR"] >= 1900]
            if not res.empty:
                res["YEAR"] = res["YEAR"].astype(int)

                # Base year (aligned with first CapitalCost year if possible)
                base_year = None
                if "CapitalCost" in df_in.columns:
                    cc_df = df_in[["TECHNOLOGY", "YEAR", "CapitalCost"]].copy()
                    cc_df["TECHNOLOGY"] = cc_df["TECHNOLOGY"].astype(str).str.strip()
                    cc_df["YEAR"] = pd.to_numeric(cc_df["YEAR"], errors="coerce")
                    cc_df["CapitalCost"] = pd.to_numeric(cc_df["CapitalCost"], errors="coerce")
                    cc_df = cc_df.dropna(subset=["YEAR", "CapitalCost"])
                    years_ok = cc_df["YEAR"][cc_df["YEAR"] >= 1900]
                    if not years_ok.empty:
                        base_year = int(years_ok.min())
                if base_year is None:
                    years_ok = res["YEAR"]
                    if not years_ok.empty:
                        base_year = int(years_ok.min())

                # Base discount rate scalar: from rate_by_ty at base_year if available, else fallback
                if rate_by_ty is not None and base_year is not None and not rate_by_ty.empty:
                    ty_year = rate_by_ty[rate_by_ty["YEAR"].astype(int) == base_year]
                    if not ty_year.empty:
                        r_base_scalar = float(pd.to_numeric(ty_year["__rate_by_ty"], errors="coerce").max())
                    else:
                        r_base_scalar = float(fallback_discount_rate)
                else:
                    r_base_scalar = float(fallback_discount_rate)

                # Per-tech base-year rate map (if available)
                r_base_map = None
                if rate_by_ty is not None and base_year is not None and not rate_by_ty.empty:
                    ty_year = rate_by_ty[rate_by_ty["YEAR"].astype(int) == base_year]
                    if not ty_year.empty:
                        r_base_map = ty_year.set_index("TECHNOLOGY")["__rate_by_ty"]
                        r_base_map.index = r_base_map.index.astype(str).str.strip()

                # Life per tech (from life_by_tech)
                life_s_local = None
                if life_by_tech is not None and not life_by_tech.empty:
                    life_s_local = pd.to_numeric(life_by_tech, errors="coerce")
                    life_s_local.index = life_s_local.index.astype(str).str.strip()

                # CRF per tech at base rate
                crf_base_s = pd.Series(dtype="float64")
                if life_s_local is not None:
                    if r_base_map is not None and not r_base_map.empty:
                        rate_vec = life_s_local.index.to_series().map(
                            lambda t: r_base_map.get(t, r_base_scalar)
                        )
                    else:
                        rate_vec = pd.Series(r_base_scalar, index=life_s_local.index)

                    crf_base_s = pd.Series(
                        [
                            capital_recovery_factor_original(r, n)
                            for r, n in zip(pd.to_numeric(rate_vec, errors="coerce"),
                                            pd.to_numeric(life_s_local, errors="coerce"))
                        ],
                        index=life_s_local.index,
                    )

                # Aggregate residual capacity per (TECHNOLOGY, YEAR)
                res_agg = res.groupby(["TECHNOLOGY", "YEAR"], sort=False)["ResidualCapacity"].max().reset_index()

                # Map base CRF
                if not crf_base_s.empty:
                    res_agg["__crf_base"] = pd.to_numeric(
                        res_agg["TECHNOLOGY"].map(crf_base_s),
                        errors="coerce",
                    ).fillna(0.0)
                else:
                    # No life information: no residual AIC
                    res_agg["__crf_base"] = 0.0

                # CapitalCost[first model year] per tech
                capcost_map = pd.Series(dtype="float64")
                if "CapitalCost" in df_in.columns:
                    cap_df = df_in[["TECHNOLOGY", "YEAR", "CapitalCost"]].copy()
                    cap_df["TECHNOLOGY"] = cap_df["TECHNOLOGY"].astype(str).str.strip()
                    cap_df["YEAR"] = pd.to_numeric(cap_df["YEAR"], errors="coerce")
                    cap_df["CapitalCost"] = pd.to_numeric(cap_df["CapitalCost"], errors="coerce")
                    cap_df = cap_df.dropna(subset=["CapitalCost"])
                    years_ok = cap_df["YEAR"].dropna()
                    years_ok = years_ok[years_ok >= 1900]
                    if not years_ok.empty:
                        first_year = int(years_ok.min())
                        first_cap = cap_df[cap_df["YEAR"] == first_year]
                        capcost_map = first_cap.groupby("TECHNOLOGY", sort=False)["CapitalCost"].max()
                    else:
                        capcost_map = cap_df.groupby("TECHNOLOGY", sort=False)["CapitalCost"].max()

                res_agg["__cap_base"] = pd.to_numeric(
                    res_agg["TECHNOLOGY"].map(capcost_map),
                    errors="coerce",
                ).fillna(0.0)

                # Residual investment and annualised residual AIC
                resid_base_invest = (
                    pd.to_numeric(res_agg["ResidualCapacity"], errors="coerce").fillna(0.0)
                    * res_agg["__cap_base"]
                )
                res_agg["__resid_aic"] = resid_base_invest * res_agg["__crf_base"]

                # Sum residual AIC by year
                resid_by_year = res_agg.groupby("YEAR", sort=True)["__resid_aic"].sum()

        # --- Total cost per year for this Future.ID ---
        cost_by_year = aic_by_year.add(om_by_year, fill_value=0.0)
        cost_by_year = cost_by_year.add(resid_by_year, fill_value=0.0)

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
    in the same style as plot_line_lcoe, ensuring Scenario 0 is drawn LAST
    so that it is on top of all other lines.
    """

    df_lcoe = lcoe_df.copy()
    df_lcoe["LCOE"] = pd.to_numeric(df_lcoe["LCOE"], errors="coerce")
    df_lcoe = df_lcoe.dropna(subset=["LCOE"])
    df_lcoe = df_lcoe.loc[df_lcoe["LCOE"] != 0]

    # Convert to USD/kWh (same as plots.py)
    df_lcoe["LCOE_kWh"] = df_lcoe["LCOE"] * 0.0036

    plt.figure(figsize=(10, 6))

    # --- 1. Plot all scenarios EXCEPT 0 first (background) ---
    for fid, grp in df_lcoe.groupby("Future.ID"):
        if fid == 0:
            continue
        grp = grp.sort_values("YEAR")
        plt.plot(
            grp["YEAR"],
            grp["LCOE_kWh"],
            color="lightgrey",
            linewidth=1,
            alpha=0.7
        )

    # --- 2. Plot Scenario 0 LAST (foreground) ---
    if 0 in df_lcoe["Future.ID"].unique():
        grp0 = df_lcoe[df_lcoe["Future.ID"] == 0].sort_values("YEAR")
        plt.plot(
            grp0["YEAR"],
            grp0["LCOE_kWh"],
            color="blue",
            linewidth=2.5,
            label="Scenario 0"
        )

    # --- Styling ---
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


def plots_main(plots_to_run=None) -> None:
    """CLI entrypoint for running standard plots."""
    print("Loading input/output CSVs...")
    df_in = pd.read_csv(CSV_INPUT, low_memory=False)
    df_out = pd.read_csv(CSV_OUTPUT, low_memory=False)

    if plots_to_run is None:
        plots_to_run = [
            "box_capacity",
            "box_activity",
            "bar_gas_capacity",
            "scatter_bess_gas",
            "line_demand",
            "line_lcoe",
            "line_gas_capex",
            "line_total_capacity",
            "line_emissions",
        ]

    for key in plots_to_run:
        fn = AVAILABLE_PLOTS.get(key)
        if fn is None:
            print(f"[WARNING] Unknown plot key: {key}")
            continue
        print(f"\n--- Running plot: {key} ---")
        fn(key, df_in=df_in, df_out=df_out, save=True)


def lcoe_main() -> None:
    """CLI entrypoint for computing and plotting LCOE."""
    input_path = CSV_INPUT
    output_path = CSV_OUTPUT

    lcoe = compute_lcoe_from_csv(input_path, output_path, fallback_discount_rate=0.07)

    print(lcoe.head(20).to_string(index=False))

    plot_line_lcoe_full_horizon(lcoe, save=True)


def main() -> None:
    """Default CLI entrypoint (plots)."""
    plots_main()

if __name__ == "__main__":
    main()
