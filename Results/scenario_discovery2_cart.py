"""
CART-only RDM script on GAS PRESENT in 2050.

- Reads OSeMOSYS_Energy_Input / Output.
- Builds scenario-level features and a binary outcome: no_gas_2050.
- Runs CART only: sklearn DecisionTreeClassifier

Usage example
-------------
python scenario_discovery_cart.py
"""

import os
import argparse
from typing import List

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
RANDOM_STATE = 42

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



def build_outcome_no_gas_2050(outputs: pd.DataFrame) -> pd.DataFrame:
    """
    Binary outcome per scenario:
    1 if gas capacity is present for PWRNGS001 in 2050, else 0.
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
    gas2050["no_gas_2050"] = (gas2050["gas_cap_2050"].abs() > eps).astype(int)

    return gas2050[["Scen_fut", "gas_cap_2050", "no_gas_2050"]]

# ---- CART runner -----------------------------------------------------------

def run_cart(df: pd.DataFrame, feature_cols: List[str], outdir: str) -> None:
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    os.makedirs(outdir, exist_ok=True)
    
    feature_name_map = {
    "capex_geo_2050_mean": "Geothermal [USD/kW]",
    "gas_price_2050_median_nonzero": "Gas price [USD/GJ]",
    "capex_impngs_2050_mean": "Gas terminal [MUSD/PJ/year]",
    "capex_bess_2050_mean": "BESS [USD/kW]",
    "capex_solar_2050_mean": "Solar [USD/kW]",
    "capex_wind_2050_mean": "Wind [USD/kW]",
    "discount_rate_batt_2050": "BESS DR [-]",
    "maxcap_geo006_2050_mean": "Max geothermal capacity [GW]",
    "discount_rate_solar_2050": "Solar DR [-]",
    "discount_rate_wind_2050": "Wind DR [-]",
    "discount_rate_impngs_2050": "Gas terminal DR [-]",
    "discount_rate_geo_2050": "Geothermal DR [-]",
    }
    
    pretty_feature_names = [
    feature_name_map.get(f, f) for f in feature_cols
    ]

    dataset_path = os.path.join(outdir, f"cart_dataset_no_gas_{YEAR}.csv")
    df.to_csv(dataset_path, index=False)

    Xmat = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    y = df["no_gas_2050"].values

    min_class_count = np.unique(y, return_counts=True)[1].min()
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise SystemExit(
            "Not enough observations in the minority class to run StratifiedKFold CV."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": [
            None,
            #"balanced",
            #{0: 1, 1: 2},
            #{0: 1, 1: 3},
            #{0: 1, 1: 5},
            #{0: 1, 1: 1}
        ],
        "random_state": [RANDOM_STATE],
    }

    clf = DecisionTreeClassifier()
    gs = GridSearchCV(
        clf,
        grid,
        scoring="recall_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    gs.fit(Xmat, y)

    # Save CV results
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_path = os.path.join(outdir, "cart_cv_results.csv")
    cv_df.to_csv(cv_path, index=False)

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)
    print("CV balanced accuracy:", round(gs.best_score_, 3))

    # In-sample diagnostics
    y_pred = best.predict(Xmat)
    bal_acc = balanced_accuracy_score(y, y_pred)
    print("In-sample balanced accuracy:", round(bal_acc, 3))

    cm = confusion_matrix(y, y_pred)
    rep = classification_report(y, y_pred, digits=3)

    pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(os.path.join(outdir, "cart_confusion_matrix.csv"))

    with open(os.path.join(outdir, "cart_classification_report.txt"), "w") as f:
        f.write(rep)

    # Feature importances
    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": best.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(outdir, "cart_feature_importances.csv"), index=False)

    # Tree plot
    if plt is not None:
        plt.figure(figsize=(16, 10))
        plot_tree(
            best,
            feature_names=pretty_feature_names,
            class_names=["Most runs without gas","Most runs with gas",],
            filled=True,
            rounded=True,
            impurity=False,
            label = "root",
            proportion = True,
            fontsize=8
        )
        plt.tight_layout()
        tree_png = os.path.join(outdir, "cart_tree.png")
        plt.savefig(tree_png, dpi=600, bbox_inches="tight")
        plt.show()

    # Tree rules
    rules = export_text(best, feature_names=feature_cols)
    with open(os.path.join(outdir, "cart_tree_rules.txt"), "w") as f:
        f.write(rules)

    print("CART artifacts written to:", outdir)

# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CART-only RDM script on gas-present-2050 outcome."
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
    ydf = build_outcome_no_gas_2050(outputs)

    df = X.merge(ydf, on="Scen_fut", how="inner").dropna()

    if df.empty:
        raise SystemExit("No scenarios with complete features/outcomes after merging.")

    feature_cols = [
        c for c in df.columns
        if c not in ("Scen_fut", "gas_cap_2050", "no_gas_2050")
    ]

    n = len(df)
    n_gas = int(df["no_gas_2050"].sum())
    n_no_gas = n - n_gas

    print(
        f"Scenarios: {n}  —  gas_present_2050 = 1 in {n_gas} ({n_gas / n:.1%}); "
        f"0 in {n_no_gas} ({n_no_gas / n:.1%})"
    )
    print("Features used:", feature_cols)

    outdir = os.path.join(OUTDIR_BASE, "cart")
    run_cart(df, feature_cols, outdir)


if __name__ == "__main__":
    main()