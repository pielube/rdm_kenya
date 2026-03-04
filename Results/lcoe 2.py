
import os
import math
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio


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




def compute_total_electricity_demand_2050(df_input: pd.DataFrame) -> pd.Series:
    """
    Series indexed by Future.ID with total SpecifiedAnnualDemand in 2050
    for COMELC + INDELC + RESELC.
    """
    commodities = ["COMELC", "INDELC", "RESELC"]
    sub = df_input.loc[
        (pd.to_numeric(df_input["YEAR"], errors="coerce") == 2050)
        & (df_input["COMMODITY"].isin(commodities)),
        ["Future.ID", "SpecifiedAnnualDemand"],
    ].copy()

    sub["SpecifiedAnnualDemand"] = pd.to_numeric(sub["SpecifiedAnnualDemand"], errors="coerce")
    sub = sub.dropna(subset=["Future.ID", "SpecifiedAnnualDemand"])
    return sub.groupby("Future.ID")["SpecifiedAnnualDemand"].sum()


def compute_backstop_capacity_2050(df_output: pd.DataFrame) -> pd.Series:
    """
    Series indexed by Future.ID with total BACKSTOP capacity in 2050 from TotalCapacityAnnual.
    """
    sub = df_output.loc[
        (pd.to_numeric(df_output["YEAR"], errors="coerce") == 2050)
        & (df_output["TECHNOLOGY"].astype(str).str.strip() == "BACKSTOP"),
        ["Future.ID", "TotalCapacityAnnual"],
    ].copy()

    sub["TotalCapacityAnnual"] = pd.to_numeric(sub["TotalCapacityAnnual"], errors="coerce")
    sub = sub.dropna(subset=["Future.ID", "TotalCapacityAnnual"])
    return sub.groupby("Future.ID")["TotalCapacityAnnual"].sum()


def plot_lcoe_plotly_browser(
    lcoe_df: pd.DataFrame,
    demand2050: pd.Series,
    backstop2050: pd.Series,
    *,
    title: str = "System-wide LCOE by Scenario (2019–2050)",
    output_html: str = "plots/line_lcoe_plotly.html",
    lcoe_unit: str = "USD/kWh",
    demand_unit: str = "PJ",
    backstop_unit: str = "GW",
    convert_lcoe_to_kwh: bool = True,
    ) -> str:

    df = lcoe_df.copy()

    df["Future.ID"] = pd.to_numeric(df["Future.ID"], errors="coerce").astype("Int64")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["LCOE"] = pd.to_numeric(df["LCOE"], errors="coerce")

    df = df.dropna(subset=["Future.ID", "YEAR", "LCOE"])
    df = df[df["LCOE"] != 0]

    if convert_lcoe_to_kwh:
        df["LCOE_plot"] = df["LCOE"] * 0.0036
    else:
        df["LCOE_plot"] = df["LCOE"]

    futures = pd.Index(sorted(df["Future.ID"].unique()), name="Future.ID")

    demand2050 = pd.to_numeric(demand2050, errors="coerce").reindex(futures).fillna(0.0)
    backstop2050 = pd.to_numeric(backstop2050, errors="coerce").reindex(futures).fillna(0.0)

    df = df.merge(
        demand2050.rename("Demand2050").reset_index(),
        on="Future.ID",
        how="left"
    ).merge(
        backstop2050.rename("Backstop2050").reset_index(),
        on="Future.ID",
        how="left"
    )

    df["Demand2050"] = df["Demand2050"].fillna(0.0)
    df["Backstop2050"] = df["Backstop2050"].fillna(0.0)

    fig = go.Figure()

    hovertemplate = (
        "Future.ID: %{customdata[0]}<br>"
        f"Demand 2050: %{{customdata[1]:.3f}} {demand_unit}<br>"
        f"BACKSTOP 2050: %{{customdata[2]:.3f}} {backstop_unit}<br>"
        "Year: %{x}<br>"
        f"LCOE: %{{y:.4f}} {lcoe_unit}"
        "<extra></extra>"
    )

    def add_trace(fid, color, width):

        g = df[df["Future.ID"] == fid].sort_values("YEAR")

        customdata = np.column_stack([
            np.full(len(g), int(fid)),
            g["Demand2050"].to_numpy(),
            g["Backstop2050"].to_numpy(),
        ])

        fig.add_trace(
            go.Scatter(
                x=g["YEAR"],
                y=g["LCOE_plot"],
                mode="lines",
                customdata=customdata,
                hovertemplate=hovertemplate,
                line=dict(color=color, width=width),
                showlegend=False
            )
        )

    # Background scenarios (grey)
    for fid in futures:
        if fid == 0:
            continue
        add_trace(fid, color="rgba(150,150,150,0.5)", width=1)

    # Scenario 0 highlighted
    if 0 in futures:
        add_trace(0, color="blue", width=3)

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=f"LCOE [{lcoe_unit}]",
        hovermode="closest",
        template="plotly_white",
        showlegend=False,
        #yaxis=dict(rangemode="tozero")
    )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    pio.write_html(
        fig,
        file=output_html,
        auto_open=True,
        include_plotlyjs="cdn"
    )

    return output_html


if __name__ == "__main__":
    input_path = "OSEMOSYS_Energy_Input.csv"
    output_path = "OSEMOSYS_Energy_Output.csv"

    # 1) Compute LCOE (your existing function)
    lcoe = compute_lcoe_from_csv(input_path, output_path, fallback_discount_rate=0.07)

    # 2) Compute metadata scalars for hover
    df_input = pd.read_csv(input_path, low_memory=False)
    df_output = pd.read_csv(output_path, low_memory=False)

    dem2050 = compute_total_electricity_demand_2050(df_input)         # PJ (unless you convert)
    backstop2050 = compute_backstop_capacity_2050(df_output)          # GW (as stored)

    # Optional: convert demand to TWh to match your earlier plot convention
    # dem2050 = dem2050 * 0.27778

    # 3) Plot in browser
    plot_lcoe_plotly_browser(
        lcoe,
        dem2050,
        backstop2050,
        output_html="plots/line_lcoe_plotly.html",
        demand_unit="PJ",      # or "TWh" if you convert above
        backstop_unit="GW",
        lcoe_unit="USD/kWh",
        convert_lcoe_to_kwh=True,
    )