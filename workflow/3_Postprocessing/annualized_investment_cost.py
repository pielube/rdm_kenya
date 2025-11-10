import os
import time
import pandas as pd

# Module-level debug flag and helper to avoid noisy output when debug=False
_DEBUG = False

def _dbg(msg: str):
    try:
        if _DEBUG:
            print(msg)
    except Exception:
        pass


def _find_case_parquet(base_dir: str, kind: str):
    """
    Find a parquet file for the case in `base_dir`.
    kind: either 'Input' or 'Output'.
    Returns full path or None.
    """
    target_suffix = f"_{kind}.parquet".lower()
    for name in os.listdir(base_dir):
        if name.lower().endswith(target_suffix):
            fp = os.path.join(base_dir, name)
            if os.path.isfile(fp):
                return fp
    return None


def _read_operational_life_from_parquet(df_input: pd.DataFrame) -> pd.Series | None:
    """
    Extract operational life (exact column 'OperationalLife') as a Series indexed by 'TECHNOLOGY'.
    Returns None if the exact columns are not present or contain no numeric values.
    """
    if 'TECHNOLOGY' not in df_input.columns or 'OperationalLife' not in df_input.columns:
        return None
    try:
        sub = df_input[['TECHNOLOGY', 'OperationalLife']].copy()
        sub['OperationalLife'] = pd.to_numeric(sub['OperationalLife'], errors='coerce')
        # Keep zeros as valid values; drop only NaNs
        sub = sub[sub['OperationalLife'].notna()]
        if sub.empty:
            return None
        # Prefer the maximum per technology to capture the defined lifetime
        # (rows for other parameters often carry 0.0 in this wide schema)
        sub['TECHNOLOGY'] = sub['TECHNOLOGY'].astype(str).str.strip()
        s = sub.groupby('TECHNOLOGY', sort=False)['OperationalLife'].max()
        return s
    except Exception:
        return None


def _capital_recovery_factor(r: float, n: float) -> float:
    """
    Compute CRF per user-specified formula:
      CRF = (1 - (1+r)^-1) / (1 - (1+r)^-n)
    with r as decimal (e.g., 0.05) and n in years. For r≈0, CRF ≈ 1/n.
    Returns 0.0 if n <= 0 or inputs invalid.
    """
    try:
        r = float(r)
        n = float(n)
        if n <= 0:
            return 0.0
        if abs(r) < 1e-12:
            return 1.0 / n
        one_plus_r = 1.0 + r
        num = 1.0 - (one_plus_r) ** (-1.0)
        den = 1.0 - (one_plus_r) ** (-n)
        if den == 0.0:
            return 0.0
        return num / den
    except Exception:
        return 0.0


def _read_discount_rate_from_parquet(df_input: pd.DataFrame):
    """
    Extract DiscountRateIdv from the input parquet.
    Returns a tuple: (rate_by_year: Series|None, scalar_rate: float|None, used_col: str|None)

    Behavior:
    - Prefer exact 'DiscountRateIdv'; accept 'DiscountRate' if the former is absent.
    - If 'YEAR' exists and per-year values are present, return a YEAR-indexed Series.
    - Otherwise, if a single non-null value exists (or all values are identical), return a scalar.
    """
    used_col = None
    if 'DiscountRateIdv' in df_input.columns:
        used_col = 'DiscountRateIdv'
    elif 'DiscountRate' in df_input.columns:
        used_col = 'DiscountRate'
    else:
        return None, None, None

    ser = pd.to_numeric(df_input[used_col], errors='coerce')
    non_null = ser.dropna()
    if non_null.empty:
        return None, None, used_col

    # Try per-year series first if YEAR exists
    if 'YEAR' in df_input.columns:
        try:
            years = pd.to_numeric(df_input['YEAR'], errors='coerce')
            sub = pd.DataFrame({'YEAR': years, used_col: ser})
            # Keep rows with a plausible calendar year and a non-null rate
            # Keep rows with TIMESLICE empty and valid years
            if 'TIMESLICE' in df_input.columns:
                ts = df_input['TIMESLICE']
                mask_ts = ts.isna() | (pd.to_numeric(ts, errors='coerce') == 0) | (ts.astype(str).str.strip() == '')
                sub = sub[mask_ts]
            sub = sub.dropna(subset=[used_col])
            # Accept any non-blank YEAR
            sub = sub[sub['YEAR'].notna()]
            if not sub.empty:
                # Debug: show the exact rows feeding the per-year series
                try:
                    _dbg("[AIC] Rate rows used (TIMESLICE empty, first 12):")
                    _dbg(sub[["YEAR", used_col]].head(12).to_string(index=False))
                except Exception:
                    pass
                sub['YEAR'] = sub['YEAR'].astype(int)
                # Robust: take the maximum DiscountRateIdv per year (ensures 0.035 preferred over smaller values)
                by_year = sub.groupby('YEAR', sort=True)[used_col].max().astype(float)
                try:
                    _dbg(f"[AIC] by_year (first 12): {by_year.head(12).to_dict()}")
                except Exception:
                    pass
                if not by_year.empty:
                    return by_year, None, used_col
        except Exception:
            pass

    # Fallback: scalar if there's a single unique value; else last non-null
    unique_vals = non_null[non_null != 0].unique()
    if unique_vals.size == 1:
        return None, float(unique_vals[0]), used_col
    # If no single unique non-zero, take the last non-zero; else last non-null
    non_zero = non_null[non_null != 0]
    if not non_zero.empty:
        return None, float(non_zero.iloc[-1]), used_col
    return None, float(non_null.iloc[-1]), used_col


def _read_operational_life_storage_from_parquet(df_input: pd.DataFrame) -> pd.Series | None:
    """
    Extract storage operational life (exact column 'OperationalLifeStorage') as a Series indexed by 'STORAGE'.
    Keeps zeros as valid, drops only NaNs. Uses max per storage to capture defined lifetime.
    """
    if 'STORAGE' not in df_input.columns or 'OperationalLifeStorage' not in df_input.columns:
        return None
    try:
        sub = df_input[['STORAGE', 'OperationalLifeStorage']].copy()
        sub['OperationalLifeStorage'] = pd.to_numeric(sub['OperationalLifeStorage'], errors='coerce')
        sub = sub[sub['OperationalLifeStorage'].notna()]
        if sub.empty:
            return None
        sub['STORAGE'] = sub['STORAGE'].astype(str).str.strip()
        s = sub.groupby('STORAGE', sort=False)['OperationalLifeStorage'].max()
        return s
    except Exception:
        return None


def _read_discount_rate_storage_from_parquet(df_input: pd.DataFrame):
    """
    Extract DiscountRateStorage from the input parquet.
    Returns (rate_by_year: Series|None, scalar_rate: float|None, used_col: str|None).
    Prefers exact 'DiscountRateStorage'; if absent, tries 'DiscountRateIdv' then 'DiscountRate'.
    """
    if 'DiscountRateStorage' in df_input.columns:
        col = 'DiscountRateStorage'
    elif 'DiscountRateIdv' in df_input.columns:
        col = 'DiscountRateIdv'
    elif 'DiscountRate' in df_input.columns:
        col = 'DiscountRate'
    else:
        return None, None, None

    ser = pd.to_numeric(df_input[col], errors='coerce')
    non_null = ser.dropna()
    if non_null.empty:
        return None, None, col

    if 'YEAR' in df_input.columns:
        try:
            years = pd.to_numeric(df_input['YEAR'], errors='coerce')
            sub = pd.DataFrame({'YEAR': years, col: ser})
            sub = sub.dropna(subset=[col])
            sub = sub[sub['YEAR'].notna()]
            if not sub.empty:
                sub['YEAR'] = sub['YEAR'].astype(int)
                per_year = {}
                for y, chunk in sub.groupby('YEAR', sort=False):
                    nz = chunk[chunk[col] != 0]
                    if not nz.empty:
                        per_year[y] = float(nz[col].iloc[-1])
                    else:
                        per_year[y] = float(chunk[col].iloc[-1])
                if per_year:
                    by_year = pd.Series(per_year).sort_index()
                    return by_year, None, col
        except Exception:
            pass

    unique_vals = non_null[non_null != 0].unique()
    if unique_vals.size == 1:
        return None, float(unique_vals[0]), col
    non_zero = non_null[non_null != 0]
    if not non_zero.empty:
        return None, float(non_zero.iloc[-1]), col
    return None, float(non_null.iloc[-1]), col


def _read_discount_rate_by_tech_year(df_input: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build a per-(TECHNOLOGY, YEAR) map for DiscountRateIdv (or DiscountRate fallback).
    Returns a DataFrame with columns ['TECHNOLOGY','YEAR','__rate_by_ty'] or None if unavailable.

    Selection rules per (TECHNOLOGY, YEAR):
    - Filter TIMESLICE empty/zero when the column exists.
    - Prefer non-zero values; otherwise use the last non-null value in input order.
    - YEAR is cast to int for grouping and joining.
    """
    if df_input is None:
        return None
    used_col = None
    if 'DiscountRateIdv' in df_input.columns:
        used_col = 'DiscountRateIdv'
    elif 'DiscountRate' in df_input.columns:
        used_col = 'DiscountRate'
    else:
        return None

    need_cols = ['TECHNOLOGY', 'YEAR', used_col]
    if 'TIMESLICE' in df_input.columns:
        need_cols.append('TIMESLICE')
    try:
        sub = df_input[need_cols].copy()
        sub['TECHNOLOGY'] = sub['TECHNOLOGY'].astype(str).str.strip()
        sub['YEAR'] = pd.to_numeric(sub['YEAR'], errors='coerce')
        sub[used_col] = pd.to_numeric(sub[used_col], errors='coerce')
        if 'TIMESLICE' in sub.columns:
            ts = sub['TIMESLICE']
            mask_ts = ts.isna() | (pd.to_numeric(ts, errors='coerce') == 0) | (ts.astype(str).str.strip() == '')
            sub = sub[mask_ts]
        sub = sub.dropna(subset=['TECHNOLOGY', 'YEAR', used_col])
        if sub.empty:
            return None
        sub['YEAR'] = sub['YEAR'].astype(int)

        # Choose per (TECHNOLOGY, YEAR) a rate: prefer last non-zero, else last non-null (fast path)
        # Keep input order for 'last' by avoiding sorts
        any_last = sub.groupby(['TECHNOLOGY', 'YEAR'], sort=False)[used_col].last()
        nz = sub[sub[used_col] != 0]
        if not nz.empty:
            nz_last = nz.groupby(['TECHNOLOGY', 'YEAR'], sort=False)[used_col].last()
            # Align to union of groups and prefer non-zero where present
            rate_ser = nz_last.reindex(any_last.index)
            rate_ser = rate_ser.where(rate_ser.notna(), any_last)
        else:
            rate_ser = any_last
        grp = rate_ser.reset_index().rename(columns={used_col: '__rate_by_ty'})
        if grp.empty:
            return None
        return grp[['TECHNOLOGY', 'YEAR', '__rate_by_ty']]
    except Exception:
        return None


def add_annualized_investment_cost(
    df: pd.DataFrame,
    case_folder_path: str,
    discount_rate: float,
    capital_col: str = "CapitalInvestment",
    output_col: str = "AnnualizedCapitalInvestment",
    default_operational_life: float | int | dict | None = None,
    debug: bool = False,
    focus_tech: str | None = "PWRGEO001",
    residual_output_col: str = "AnnualizedCapitalInvestmentResidual",
) -> pd.DataFrame:
    """
    Append-only computation of investment metrics and LCOE.

    Produces four compact blocks and appends them to the base rows (which remain unchanged):
    - TECHNOLOGY–YEAR: `{output_col}` (default 'AnnualizedCapitalInvestment')
    - TECHNOLOGY–YEAR: `{residual_output_col}` (default 'AnnualizedCapitalInvestmentResidual')
    - STORAGE–YEAR: 'AnnualizedCapitalInvestmentStorage'
    - YEAR: 'LCOE'

    Strategy and Future.ID are derived from `case_folder_path` by splitting the
    case folder name on the last underscore, e.g. 'Scenario1_5' -> ('Scenario1', '5').
    """
    # Sync module-level debug flag to control all prints
    global _DEBUG
    _DEBUG = bool(debug)

    # Keep an untouched copy of base rows; never merge metrics into it
    df_base = df.copy()

    # Derive Strategy/Future.ID from case folder name
    strategy_val, future_id_val = None, None
    try:
        case = os.path.basename(case_folder_path.rstrip(os.sep))
        strategy_val, future_id_val = case.rsplit('_', 1)
    except Exception:
        pass

    # Locate and read the case parquet files
    input_parquet = _find_case_parquet(case_folder_path, "Input")
    # Read Input first (authoritative for parameters)
    df_input = None
    if input_parquet and os.path.isfile(input_parquet):
        try:
            t0 = time.perf_counter()
            # Read only needed columns for performance; fallback to full read if unavailable
            need_cols = [
                'TECHNOLOGY', 'YEAR', 'TIMESLICE',
                'OperationalLife', 'OperationalLifeStorage',
                'DiscountRateIdv', 'DiscountRate',
                'ResidualCapacity', 'CapitalCost',
                'SpecifiedAnnualDemand',
            ]
            try:
                df_input = pd.read_parquet(input_parquet, columns=need_cols, engine='pyarrow')
            except Exception:
                df_input = pd.read_parquet(input_parquet, engine='pyarrow')
            if debug:
                dt = time.perf_counter() - t0
                print(f"[AIC] Read input parquet '{os.path.basename(input_parquet)}' shape={df_input.shape} in {dt:.3f}s")
            # Debug: verify DiscountRateIdv presence and show minimal sample
            try:
                if debug and df_input is not None:
                    print(f"[AIC] Input cols: {list(df_input.columns)} size={len(df_input)}")
                    if 'DiscountRateIdv' in df_input.columns:
                        sub = df_input[['TECHNOLOGY','YEAR','TIMESLICE','DiscountRateIdv']].copy()
                        print("[AIC] Rate sample (raw, first 10):")
                        print(sub.head(10).to_string(index=False))
                        # Optional origin tracer for suspicious values (e.g., 0.08)
                        try:
                            sus = 0.08
                            crev = df_input.loc[
                                df_input['DiscountRateIdv'].astype(str) == str(sus),
                                ['TECHNOLOGY','TIMESLICE','YEAR','DiscountRateIdv']
                            ].head(8)
                            if not crev.empty:
                                print(f"[AIC] Where does {sus} come from (first 8 rows)?")
                                print(crev.to_string(index=False))
                        except Exception:
                            pass
                    else:
                        print("[AIC] DiscountRateIdv column missing in df_input")
            except Exception:
                pass
        except Exception as e:
            if debug:
                print(f"[AIC] Failed to read input parquet: {e}")
            df_input = None

    # Extract operational life and discount series from parquet when possible
    life_by_tech = None
    rate_by_year = None
    if df_input is not None:
        try:
            life_by_tech = _read_operational_life_from_parquet(df_input)
        except Exception:
            life_by_tech = None
        try:
            rate_by_year, rate_scalar, rate_used_col = _read_discount_rate_from_parquet(df_input)
        except Exception:
            rate_by_year, rate_scalar, rate_used_col = None, None, None
        if debug:
            print(f"[AIC] Columns in input parquet: {list(df_input.columns)[:12]}...")
            life_count = (0 if life_by_tech is None else int(life_by_tech.notna().sum()))
            rate_count = (0 if rate_by_year is None else int(rate_by_year.notna().sum()))
            print(f"[AIC] Found OperationalLife map: {life_count} tech entries")
            if rate_by_year is not None:
                print(f"[AIC] Found {rate_used_col} series: {rate_count} year entries")
            elif rate_scalar is not None:
                print(f"[AIC] Found scalar {rate_used_col}: {rate_scalar}")
            else:
                print(f"[AIC] No discount rate found in input parquet")
            if life_by_tech is not None and not life_by_tech.empty:
                try:
                    print(f"[AIC] OperationalLife stats: min={life_by_tech.min()}, max={life_by_tech.max()}")
                    if focus_tech:
                        tech_key = str(focus_tech).strip()
                        if tech_key in life_by_tech.index:
                            print(f"[AIC] OperationalLife[{tech_key}] = {life_by_tech.loc[tech_key]}")
                except Exception:
                    pass
            if rate_by_year is not None and not rate_by_year.empty:
                try:
                    years_sorted = sorted(rate_by_year.dropna().index.astype(int).tolist())
                    yr_min = years_sorted[0]
                    yr_max = years_sorted[-1]
                    print(f"[AIC] DiscountRate year range: {yr_min}..{yr_max}; min={rate_by_year.min()}, mean={rate_by_year.mean()}, max={rate_by_year.max()}")
                    # Show a small sample around typical build years if present
                    for yr in (2025, 2030, 2035):
                        if yr in rate_by_year.index:
                            print(f"[AIC] {rate_used_col}[{yr}] = {rate_by_year.loc[yr]}")
                except Exception:
                    pass

    # Ensure minimal keys exist to compute technology-based metrics
    has_tech = ("TECHNOLOGY" in df_base.columns)
    has_year = ("YEAR" in df_base.columns)
    if not has_year:
        if debug:
            print(f"[AIC] Missing YEAR in output; returning base rows only")
        return df_base

    # Map life and per-row discount rate
    # Normalize join keys
    tech_series = df_base["TECHNOLOGY"].astype(str).str.strip() if has_tech else pd.Series(index=df_base.index, dtype='object')
    year_series = pd.to_numeric(df_base["YEAR"], errors='coerce')

    # Start with life from parquet if available
    if life_by_tech is not None and not life_by_tech.empty:
        life_map = tech_series.map(life_by_tech)
    else:
        life_map = pd.Series(index=df.index, dtype="float64")

    # Apply YAML/default overrides for missing life values
    if default_operational_life is not None:
        if isinstance(default_operational_life, dict):
            life_map = life_map.fillna(df["TECHNOLOGY"].map(default_operational_life))
        else:
            try:
                life_default = float(default_operational_life)
                life_map = life_map.fillna(life_default)
            except Exception:
                pass

    # If life is still entirely missing, abort quietly
    if life_map.isna().all():
        return df
    # Prefer (TECHNOLOGY, YEAR) specific rate if present; fallback to per-year, then scalar
    rate_map = pd.Series(index=df.index, dtype='float64')
    rate_by_ty = None
    try:
        rate_by_ty = _read_discount_rate_by_tech_year(df_input) if df_input is not None else None
    except Exception:
        rate_by_ty = None

    if rate_by_ty is not None and not rate_by_ty.empty and has_tech:
        # Fast map using MultiIndex reindex instead of merge
        try:
            s_ty = rate_by_ty.set_index(['TECHNOLOGY', 'YEAR'])['__rate_by_ty']
            mask_y = year_series.notna()
            tech_arr = tech_series[mask_y].astype(str).values
            year_arr = year_series[mask_y].astype(int).values
            mi = pd.MultiIndex.from_arrays([tech_arr, year_arr])
            mapped = s_ty.reindex(mi)
            if not mapped.empty:
                rate_map = pd.Series(index=df_base.index, dtype='float64')
                rate_map.loc[mask_y] = pd.to_numeric(mapped.values, errors='coerce')
        except Exception:
            # Fallback to merge if anything goes wrong
            key_df = pd.DataFrame({
                'idx': df_base.index,
                'TECHNOLOGY': tech_series,
                'YEAR': year_series.astype('Int64')
            })
            key_df = key_df[key_df['YEAR'].notna()].copy()
            key_df['YEAR'] = key_df['YEAR'].astype(int)
            merged = key_df.merge(rate_by_ty, on=['TECHNOLOGY', 'YEAR'], how='left')
            rate_map = pd.Series(index=df_base.index, dtype='float64')
            if '__rate_by_ty' in merged.columns:
                rate_map.loc[merged['idx'].values] = pd.to_numeric(merged['__rate_by_ty'], errors='coerce').values
    # Fill remaining from per-year series if available
    if (rate_map.isna().any()) and (rate_by_year is not None and not rate_by_year.empty):
        try:
            rate_by_year_idx_int = rate_by_year.copy()
            rate_by_year_idx_int.index = rate_by_year_idx_int.index.astype(int)
        except Exception:
            rate_by_year_idx_int = rate_by_year
        rate_map = rate_map.fillna(year_series.map(rate_by_year_idx_int))
    # Fill remaining from scalar or provided default
    if 'rate_scalar' in locals() and rate_scalar is not None:
        rate_map = rate_map.fillna(rate_scalar)
    rate_map = rate_map.fillna(discount_rate)

    # Prepare append-only containers
    appended_blocks: list[pd.DataFrame] = []

    # Annualized capital investment by TECHNOLOGY–YEAR (append-only)
    aic_map_df = pd.DataFrame(columns=['TECHNOLOGY', 'YEAR', output_col])
    cap = pd.Series(dtype='float64')
    if has_tech and (capital_col in df_base.columns):
        # Compute CRF row-wise (per technology-year)
        crf = pd.Series([
            _capital_recovery_factor(r, n)
            for r, n in zip(pd.to_numeric(rate_map, errors='coerce').tolist(), pd.to_numeric(life_map, errors='coerce').tolist())
        ], index=df_base.index)

        cap = pd.to_numeric(df_base[capital_col], errors="coerce")
        increment = cap * crf

        try:
            tmp = pd.DataFrame({
                'TECHNOLOGY': tech_series,
                'YEAR': year_series.astype('Int64'),
                'life': pd.to_numeric(life_map, errors='coerce'),
                'inc': pd.to_numeric(increment, errors='coerce').fillna(0.0),
            })
            tmp = tmp[tmp['YEAR'].notna()].copy()
            tmp['YEAR'] = tmp['YEAR'].astype(int)

            aic_records = []
            for tech, g in tmp.groupby('TECHNOLOGY', sort=False):
                g = g.sort_values('YEAR')
                life_val = pd.to_numeric(g['life'], errors='coerce').dropna()
                if life_val.empty:
                    window = 1
                else:
                    window = int(max(1, int(float(pd.Series([float(l) for l in life_val]).max() + 0.999999))))
                per_year = g.groupby('YEAR', sort=True)['inc'].max()
                roll = per_year.rolling(window=window, min_periods=1).sum()
                aic_records.append(pd.DataFrame({
                    'TECHNOLOGY': tech,
                    'YEAR': roll.index.values,
                    output_col: roll.values,
                }))
            aic_map_df = pd.concat(aic_records, ignore_index=True) if aic_records else aic_map_df
            aic_map_df['TECHNOLOGY'] = aic_map_df['TECHNOLOGY'].astype(str)
            if not aic_map_df.empty:
                app = aic_map_df[['TECHNOLOGY','YEAR', output_col]].copy()
                app.insert(0, 'Future.ID', future_id_val)
                app.insert(0, 'Strategy', strategy_val)
                appended_blocks.append(app)
            if debug and not aic_map_df.empty:
                stats = pd.to_numeric(aic_map_df[output_col], errors='coerce')
                if not stats.empty:
                    print(f"[AIC] {output_col} rows={len(stats)}, min={stats.min():.6g}, mean={stats.mean():.6g}, max={stats.max():.6g}")
        except Exception as e:
            if debug:
                print(f"[AIC] Tech AIC computation failed: {e}")

    # Add ResidualCapacity annualized component using CRF at the first/smallest model year
    # Residual capacity component by TECHNOLOGY–YEAR (append-only)
    try:
        if df_input is not None and 'ResidualCapacity' in df_input.columns:
            res = df_input[['TECHNOLOGY','YEAR','ResidualCapacity']].copy()
            res['TECHNOLOGY'] = res['TECHNOLOGY'].astype(str).str.strip()
            res['YEAR'] = pd.to_numeric(res['YEAR'], errors='coerce')
            res = res[res['YEAR'].notna() & (res['YEAR'] >= 1900)]
            res['YEAR'] = res['YEAR'].astype(int)
            res['ResidualCapacity'] = pd.to_numeric(res['ResidualCapacity'], errors='coerce')
            res = res.dropna(subset=['ResidualCapacity'])
            if not res.empty:
                # Determine base year aligned with CapitalCost first model year when possible
                base_year = None
                try:
                    if 'CapitalCost' in df_input.columns:
                        cc_df = df_input[['TECHNOLOGY','YEAR','CapitalCost']].copy()
                        cc_df['YEAR'] = pd.to_numeric(cc_df['YEAR'], errors='coerce')
                        cc_df['CapitalCost'] = pd.to_numeric(cc_df['CapitalCost'], errors='coerce')
                        cc_df = cc_df.dropna(subset=['YEAR','CapitalCost'])
                        years_ok = cc_df['YEAR'][cc_df['YEAR'] >= 1900]
                        if not years_ok.empty:
                            base_year = int(years_ok.min())
                except Exception:
                    base_year = None
                if base_year is None:
                    try:
                        years_ok = res['YEAR'][res['YEAR'] >= 1900]
                        if not years_ok.empty:
                            base_year = int(years_ok.min())
                    except Exception:
                        base_year = None
                if debug:
                    try:
                        print(f"[AIC] Residual base year used: {base_year}")
                    except Exception:
                        pass

                # Determine base discount rate using per-tech-year for base_year when available; otherwise fallbacks
                # Global scalar fallback from per-year series at base_year if present; else scalar; else provided
                if rate_by_year is not None and not rate_by_year.empty and base_year is not None and base_year in set(rate_by_year.index.astype(int)):
                    try:
                        r_base_scalar = float(rate_by_year.loc[int(base_year)])
                    except Exception:
                        r_base_scalar = float(discount_rate)
                elif 'rate_scalar' in locals() and rate_scalar is not None:
                    r_base_scalar = float(rate_scalar)
                else:
                    r_base_scalar = float(discount_rate)

                # Per-tech rate specifically at base_year
                r_base_map = None
                try:
                    if 'rate_by_ty' in locals() and rate_by_ty is not None and not rate_by_ty.empty and base_year is not None:
                        ty_year = rate_by_ty[rate_by_ty['YEAR'].astype(int) == int(base_year)]
                        if not ty_year.empty:
                            r_base_map = ty_year.set_index('TECHNOLOGY')['__rate_by_ty']
                            r_base_map.index = r_base_map.index.astype(str).str.strip()
                except Exception:
                    r_base_map = None

                # CRF per technology at base rate using life_by_tech
                crf_base_s = pd.Series(dtype='float64')
                life_s_local = None
                if life_by_tech is not None and not life_by_tech.empty:
                    life_s_local = pd.to_numeric(life_by_tech, errors='coerce')
                    life_s_local.index = life_s_local.index.astype(str).str.strip()
                    if r_base_map is not None and not r_base_map.empty:
                        # Compose per-tech rate with scalar fallback
                        rate_vec = life_s_local.index.to_series().map(lambda t: r_base_map.get(t, r_base_scalar))
                        crf_base_s = pd.Series([
                            _capital_recovery_factor(float(r), float(n))
                            for r, n in zip(pd.to_numeric(rate_vec, errors='coerce'), pd.to_numeric(life_s_local, errors='coerce'))
                        ], index=life_s_local.index)
                    else:
                        crf_base_s = life_s_local.apply(lambda n: _capital_recovery_factor(r_base_scalar, n))
                # Aggregate residual capacity per (tech, year)
                res_agg = res.groupby(['TECHNOLOGY','YEAR'], sort=False)['ResidualCapacity'].max().reset_index()
                res_agg['__crf_base'] = res_agg['TECHNOLOGY'].map(crf_base_s)
                # Annotate inputs for debugging: rate and life used
                try:
                    if r_base_map is not None and not r_base_map.empty:
                        res_agg['__r_base'] = pd.to_numeric(res_agg['TECHNOLOGY'].map(r_base_map), errors='coerce').fillna(r_base_scalar)
                    else:
                        res_agg['__r_base'] = float(r_base_scalar)
                except Exception:
                    res_agg['__r_base'] = float(r_base_scalar)
                try:
                    if life_s_local is not None:
                        res_agg['__life'] = pd.to_numeric(res_agg['TECHNOLOGY'].map(life_s_local), errors='coerce')
                except Exception:
                    pass

                # Multiply residual capacity by CapitalCost of first model year before applying CRF
                capcost_map = pd.Series(dtype='float64')
                if 'CapitalCost' in df_input.columns:
                    cap_df = df_input[['TECHNOLOGY','YEAR','CapitalCost']].copy()
                    cap_df['TECHNOLOGY'] = cap_df['TECHNOLOGY'].astype(str).str.strip()
                    cap_df['YEAR'] = pd.to_numeric(cap_df['YEAR'], errors='coerce')
                    cap_df['CapitalCost'] = pd.to_numeric(cap_df['CapitalCost'], errors='coerce')
                    cap_df = cap_df.dropna(subset=['CapitalCost'])
                    years_ok = cap_df['YEAR'].dropna()
                    years_ok = years_ok[years_ok >= 1900]
                    if not years_ok.empty:
                        first_year = int(years_ok.min())
                        first_cap = cap_df[cap_df['YEAR'] == first_year]
                        capcost_map = first_cap.groupby('TECHNOLOGY', sort=False)['CapitalCost'].max()
                    else:
                        capcost_map = cap_df.groupby('TECHNOLOGY', sort=False)['CapitalCost'].max()
                # Compute residual AIC = ResidualCapacity * CapitalCost[first_year] * CRF_base
                res_agg['__cap_base'] = res_agg['TECHNOLOGY'].map(capcost_map)
                res_agg['__cap_base'] = pd.to_numeric(res_agg['__cap_base'], errors='coerce').fillna(0.0)
                resid_base_invest = pd.to_numeric(res_agg['ResidualCapacity'], errors='coerce').fillna(0.0) * res_agg['__cap_base']
                res_agg['__resid_aic'] = resid_base_invest * pd.to_numeric(res_agg['__crf_base'], errors='coerce').fillna(0.0)

                # Debug preview of residual inputs
                if debug:
                    try:
                        prev_cols = ['TECHNOLOGY','YEAR','ResidualCapacity','__cap_base','__r_base','__life','__crf_base','__resid_aic']
                        prev = res_agg[[c for c in prev_cols if c in res_agg.columns]].copy()
                        print("[AIC] Residual inputs sample (first 12 rows):")
                        print(prev.sort_values(['TECHNOLOGY','YEAR']).head(12).to_string(index=False))
                        if focus_tech:
                            fprev = prev[prev['TECHNOLOGY'].astype(str) == str(focus_tech)]
                            if not fprev.empty:
                                print(f"[AIC] Residual inputs for {focus_tech} (first 12 years):")
                                print(fprev.sort_values('YEAR').head(12).to_string(index=False))
                    except Exception:
                        pass

                res_app = res_agg[['TECHNOLOGY','YEAR','__resid_aic']].rename(columns={'__resid_aic': residual_output_col})
                res_app.insert(0, 'Future.ID', future_id_val)
                res_app.insert(0, 'Strategy', strategy_val)
                appended_blocks.append(res_app)
                if debug:
                    print(f"[AIC] Residual component rows={len(res_app)}")
    except Exception as e:
        if debug:
            print(f"[AIC] ResidualCapacity processing failed: {e}")

    # Storage annualized investment cost by STORAGE–YEAR (append-only)
    storage_output_col = 'AnnualizedCapitalInvestmentStorage'
    try:
        storage_cost_col = 'CapitalInvestmentStorage'
        if storage_cost_col in df_base.columns and df_input is not None and 'STORAGE' in df_base.columns:
            life_storage = _read_operational_life_storage_from_parquet(df_input)
            storage_series = df_base['STORAGE'].astype(str)
            if life_storage is not None and not life_storage.empty:
                life_st_map = storage_series.map(life_storage)
            else:
                life_st_map = pd.Series(index=df_base.index, dtype='float64')

            # Per instruction: hard-code storage discount rate to 0.05
            rate_st_map = pd.Series(0.05, index=df_base.index)

            crf_st = pd.Series([
                _capital_recovery_factor(r, n)
                for r, n in zip(pd.to_numeric(rate_st_map, errors='coerce').tolist(), pd.to_numeric(life_st_map, errors='coerce').tolist())
            ], index=df_base.index)
            inv_st = pd.to_numeric(df_base[storage_cost_col], errors='coerce')
            inc_st = inv_st * crf_st

            tmp_st = pd.DataFrame({
                'STORAGE': storage_series,
                'YEAR': year_series.astype('Int64'),
                'life': pd.to_numeric(life_st_map, errors='coerce'),
                'inc': pd.to_numeric(inc_st, errors='coerce').fillna(0.0),
            })
            tmp_st = tmp_st[tmp_st['YEAR'].notna()].copy()
            tmp_st['YEAR'] = tmp_st['YEAR'].astype(int)

            aic_st_records = []
            for stor, g in tmp_st.groupby('STORAGE', sort=False):
                g = g.sort_values('YEAR')
                life_vals = pd.to_numeric(g['life'], errors='coerce').dropna()
                if life_vals.empty:
                    window = 1
                else:
                    window = int(max(1, int(float(pd.Series([float(l) for l in life_vals]).max() + 0.999999))))
                per_year = g.groupby('YEAR', sort=True)['inc'].max()
                roll = per_year.rolling(window=window, min_periods=1).sum()
                aic_st_records.append(pd.DataFrame({
                    'STORAGE': stor,
                    'YEAR': roll.index.values,
                    storage_output_col: roll.values,
                }))
            aic_st_df = pd.concat(aic_st_records, ignore_index=True) if aic_st_records else pd.DataFrame(columns=['STORAGE','YEAR',storage_output_col])
            aic_st_df['STORAGE'] = aic_st_df['STORAGE'].astype(str)
            if not aic_st_df.empty:
                st_app = aic_st_df[['STORAGE','YEAR', storage_output_col]].copy()
                st_app.insert(0, 'Future.ID', future_id_val)
                st_app.insert(0, 'Strategy', strategy_val)
                appended_blocks.append(st_app)
            if debug and not aic_st_df.empty:
                stats = pd.to_numeric(aic_st_df[storage_output_col], errors='coerce')
                if not stats.empty:
                    print(f"[AIC] {storage_output_col} rows={len(stats)}, min={stats.min():.6g}, mean={stats.mean():.6g}, max={stats.max():.6g}")
    except Exception as e:
        if debug:
            print(f"[AIC] Storage AIC computation failed: {e}")

    # Compute LCOE per YEAR using appended metrics + base Variable/FixedCost (append-only)
    try:
        # Helper to sum a column by YEAR from a df
        def sum_by_year_df(frame: pd.DataFrame, value_col: str) -> pd.Series:
            if frame is None or frame.empty or value_col not in frame.columns or 'YEAR' not in frame.columns:
                return pd.Series(dtype='float64')
            s = pd.to_numeric(frame[value_col], errors='coerce').fillna(0.0)
            yrs = pd.to_numeric(frame['YEAR'], errors='coerce')
            mask = yrs.notna()
            if not mask.any():
                return pd.Series(dtype='float64')
            return s[mask].groupby(yrs[mask].astype(int), sort=True).sum()

        # Base costs from base rows only (support both legacy and new column names)
        var_col = next((c for c in ['AnnualVariableOperatingCost', 'VariableCost'] if c in df_base.columns), None)
        fix_col = next((c for c in ['AnnualFixedOperatingCost', 'FixedCost'] if c in df_base.columns), None)
        var_by_year = sum_by_year_df(df_base[['YEAR', var_col]], var_col) if var_col else pd.Series(dtype='float64')
        fix_by_year = sum_by_year_df(df_base[['YEAR', fix_col]], fix_col) if fix_col else pd.Series(dtype='float64')
        # (Removed per request) Variable/Fixed cost preview by year

        # Appended metrics by year
        aic_by_year = sum_by_year_df(aic_map_df, output_col)

        # Find residual and storage blocks among appended ones
        resid_block = None
        stor_block = None
        for blk in appended_blocks:
            if isinstance(blk, pd.DataFrame) and residual_output_col in blk.columns:
                resid_block = blk
            if isinstance(blk, pd.DataFrame) and storage_output_col in blk.columns:
                stor_block = blk
        resid_by_year = sum_by_year_df(resid_block, residual_output_col)
        stor_by_year = sum_by_year_df(stor_block, storage_output_col)

        # Aggregate costs by year
        cost_by_year = pd.Series(dtype='float64')
        for s_ in (var_by_year, fix_by_year, aic_by_year, resid_by_year, stor_by_year):
            if s_ is not None and not s_.empty:
                cost_by_year = s_ if cost_by_year.empty else cost_by_year.add(s_, fill_value=0.0)

        # Demand from input parquet
        demand_by_year = None
        if df_input is not None and 'SpecifiedAnnualDemand' in df_input.columns:
            din = df_input[['YEAR', 'SpecifiedAnnualDemand']].copy()
            din['YEAR'] = pd.to_numeric(din['YEAR'], errors='coerce')
            mask = din['YEAR'].notna() & (din['YEAR'] >= 1900)
            if mask.any():
                din = din.loc[mask]
                din['YEAR'] = din['YEAR'].astype(int)
                din['SpecifiedAnnualDemand'] = pd.to_numeric(din['SpecifiedAnnualDemand'], errors='coerce').fillna(0.0)
                demand_by_year = din.groupby('YEAR', sort=True)['SpecifiedAnnualDemand'].sum()

        # Debug: Show LCOE numerator (costs) and denominator (demand) for 2019–2025
        if debug:
            try:
                years_check = list(range(2019, 2026))
                num_dict = {}
                den_dict = {}
                if cost_by_year is not None and not cost_by_year.empty:
                    for y in years_check:
                        if y in cost_by_year.index:
                            try:
                                num_dict[y] = float(cost_by_year.loc[y])
                            except Exception:
                                pass
                if demand_by_year is not None and not demand_by_year.empty:
                    for y in years_check:
                        if y in demand_by_year.index:
                            try:
                                den_dict[y] = float(demand_by_year.loc[y])
                            except Exception:
                                pass
                print(f"[AIC] LCOE numerator (cost) 2019–2025: {num_dict}")
                print(f"[AIC] LCOE denominator (demand) 2019–2025: {den_dict}")
            except Exception:
                pass

        if demand_by_year is not None and not demand_by_year.empty and not cost_by_year.empty:
            common_years = cost_by_year.index.intersection(demand_by_year.index)
            if not common_years.empty:
                lcoe_by_year = cost_by_year.loc[common_years] / demand_by_year.loc[common_years]
                lcoe_df = lcoe_by_year.reset_index()
                lcoe_df.columns = ['YEAR', 'LCOE']
                lcoe_df['YEAR'] = lcoe_df['YEAR'].astype(int)
                lcoe_df.insert(0, 'Future.ID', future_id_val)
                lcoe_df.insert(0, 'Strategy', strategy_val)
                appended_blocks.append(lcoe_df)
    except Exception as e:
        if debug:
            print(f"[AIC] LCOE computation failed: {e}")

    # Focus table for first 10 years (CapitalInvestment, OperationalLife, DiscountRateIdv, CRF, AnnualizedCapitalInvestment)
    try:
        # Debug: show mapped per-row discount rate for focus tech
        if debug and focus_tech:
            try:
                m = (df['TECHNOLOGY'].astype(str) == str(focus_tech)) & df['YEAR'].notna()
                preview = pd.DataFrame({
                    'YEAR': pd.to_numeric(df.loc[m,'YEAR'], errors='coerce').astype('Int64'),
                    'DiscountRateIdv(mapped)': pd.to_numeric(rate_map.loc[m], errors='coerce')
                }).dropna().sort_values('YEAR').head(12)
                print(f"[AIC] Mapped DiscountRateIdv for {focus_tech} (first 12 years):")
                print(preview.to_string(index=False))
            except Exception:
                pass
        if debug and focus_tech and 'TECHNOLOGY' in df.columns:
            tech_key = str(focus_tech)
            tech_mask = df['TECHNOLOGY'].astype(str) == tech_key
            years = pd.to_numeric(df.loc[tech_mask, 'YEAR'], errors='coerce')
            caps = pd.to_numeric(df.loc[tech_mask, capital_col], errors='coerce')
            valid = years.notna()
            if valid.any():
                years_i = years[valid].astype(int)
                cap_per_year = caps[valid].fillna(0.0).groupby(years_i, sort=True).sum()
                # life per tech
                life_val = None
                if 'life_by_tech' in locals() and life_by_tech is not None and tech_key in life_by_tech.index:
                    life_val = float(pd.to_numeric(life_by_tech.loc[tech_key], errors='coerce'))
                # rate per year: prefer per-tech-year mapping used in computation
                rate_series = None
                try:
                    if 'rate_by_ty' in locals() and rate_by_ty is not None and not rate_by_ty.empty:
                        ty_map = rate_by_ty[rate_by_ty['TECHNOLOGY'].astype(str) == tech_key]
                        if not ty_map.empty:
                            ty_map = ty_map.set_index('YEAR')['__rate_by_ty']
                            ty_map.index = ty_map.index.astype(int)
                            rate_series = pd.to_numeric(ty_map.reindex(cap_per_year.index), errors='coerce')
                except Exception:
                    rate_series = None
                if rate_series is None or rate_series.isna().all():
                    if 'rate_by_year' in locals() and rate_by_year is not None and not rate_by_year.empty:
                        try:
                            r_idx = rate_by_year.copy(); r_idx.index = r_idx.index.astype(int)
                            rate_series = r_idx.reindex(cap_per_year.index).astype(float)
                        except Exception:
                            rate_series = None
                if rate_series is None:
                    r_scalar = None
                    if 'rate_scalar' in locals() and rate_scalar is not None:
                        r_scalar = float(rate_scalar)
                    elif 'discount_rate' in locals() and discount_rate is not None:
                        r_scalar = float(discount_rate)
                    rate_series = pd.Series(r_scalar if r_scalar is not None else 0.0, index=cap_per_year.index)
                # CRF and AIC roll
                def _crf_for_year(r):
                    return _capital_recovery_factor(float(r), float(life_val) if life_val is not None else 0.0)
                crf_series = rate_series.apply(_crf_for_year)
                inc_per_year = cap_per_year * crf_series
                window = int(max(1, int((life_val if life_val is not None else 0.0) + 0.999999)))
                aic_roll = inc_per_year.rolling(window=window, min_periods=1).sum()
                focus_table = pd.DataFrame({
                    'YEAR': cap_per_year.index,
                    'CapitalInvestment': cap_per_year.values,
                    'OperationalLife': life_val if life_val is not None else float('nan'),
                    'DiscountRateIdv': rate_series.values,
                    'CRF': crf_series.values,
                    'AnnualizedCapitalInvestment': aic_roll.values,
                })
                print(f"[AIC] Focus '{tech_key}' first 10 years:")
                print(focus_table.sort_values('YEAR').head(10).to_string(index=False))
    except Exception as e:
        if debug:
            print(f"[AIC] Focus table failed: {e}")

    if debug:
        try:
            non_null_cap = int(cap.notna().sum()) if not cap.empty else 0
            non_zero_cap = int((cap.fillna(0) != 0).sum()) if not cap.empty else 0
            non_null_life = int(life_map.notna().sum())
            non_zero_life = int((pd.to_numeric(life_map, errors='coerce').fillna(0) > 0).sum())
            non_null_rate = int(pd.to_numeric(rate_map, errors='coerce').notna().sum())
            non_zero_rate = int((pd.to_numeric(rate_map, errors='coerce').fillna(0) != 0).sum())
            print(f"[AIC] cap non-null={non_null_cap}, non-zero={non_zero_cap}; life mapped={non_null_life} (>0: {non_zero_life}); rate mapped={non_null_rate} (non-zero: {non_zero_rate})")
        except Exception:
            pass

    # Concatenate base rows with appended compact blocks (append-only)
    if appended_blocks:
        try:
            df_final = pd.concat([df_base] + appended_blocks, ignore_index=True, sort=False)
            return df_final
        except Exception:
            pass
    return df_base
