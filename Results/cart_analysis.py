
# =============================================================================
# CART (Decision Tree) analysis for NO GAS in 2050
# =============================================================================
# File: CART_RDM_OSeMOSYS_example.py
"""
CART analysis mirroring the PRIM setup, starting from the same two files.
- Computes the *same three features* you specified:
  (1) demand_2050_sum, (2) capex_geo_2050_mean, (3) gas_price_2050_median_nonzero.
- Defines the binary outcome `no_gas_2050` based on TotalCapacityAnnual in 2050
  for TECHNOLOGY == 'PWRNGS001'.
- Trains a CART classifier (scikit-learn DecisionTreeClassifier) with a small
  hyperparameter grid and stratified CV using balanced accuracy.
- Exports: dataset CSV, CV table, feature importances CSV, a PNG of the tree,
  and a text file with human-readable rules.

Run: `python CART_RDM_OSeMOSYS_example.py`
"""
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---- Configuration ---------------------------------------------------------
INPUTS_PATH = "OSEMOSYS_Energy_Input.csv"   # or .xlsx
OUTPUTS_PATH = "OSEMOSYS_Energy_Output.csv"  # or .xlsx
OUTDIR = "cart_artifacts"
RANDOM_STATE = 42
os.makedirs(OUTDIR, exist_ok=True)

# ---- IO helpers ------------------------------------------------------------

def read_table(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    elif path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    else:
        raise SystemExit(f"Unsupported file type for: {path}")

# ---- Feature & outcome builders (identical logic to PRIM version) ----------

def build_features(inputs: pd.DataFrame) -> pd.DataFrame:
    if "Scen_fut" not in inputs.columns:
        raise ValueError("Expected 'Scen_fut' key column in inputs.")
    
    
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")
    
    
    feats = []
    for scen, sub in inputs.groupby("Scen_fut", sort=False):
        f = {"Scen_fut": scen}
        sub50 = sub.loc[sub.get("YEAR").eq(2050) if "YEAR" in sub.columns else []]
    
    
        # # (1) SpecifiedAnnualDemand in 2050 summed
        # f["demand_2050_sum"] = to_num(sub50.get("SpecifiedAnnualDemand", np.nan)).sum()
        
        # (2) CapitalCost in 2050 over PWRGEO*
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            geo50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRGEO")]
            vals = to_num(geo50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_geo_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_geo_2050_mean"] = np.nan
    
        # (3) VariableCost of IMPNGS in 2050 (drop 0 and NaN), median
        if {"TECHNOLOGY", "VariableCost"}.issubset(sub50.columns):
            gas50 = sub50.loc[sub50["TECHNOLOGY"].astype(str) == "IMPNGS", "VariableCost"]
            gas50 = to_num(gas50)
            gas50 = gas50[(~gas50.isna()) & (gas50 != 0)]
            f["gas_price_2050_median_nonzero"] = gas50.median() if not gas50.empty else np.nan
        else:
            f["gas_price_2050_median_nonzero"] = np.nan
            
        # (4) CapitalCost in 2050 over PWRURN (nuclear)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRURN")]
            vals = to_num(nuc50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_nuc_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_nuc_2050_mean"] = np.nan
            
        # (5) CapitalCost in 2050 over BESS_TECH (batteries)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("BESS_TECH")]
            vals = to_num(nuc50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_bess_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_bess_2050_mean"] = np.nan
            
        # (6) CapitalCost in 2050 over PWRSOL (solar)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL")]
            vals = to_num(nuc50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_solar_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_solar_2050_mean"] = np.nan
            
        # (7) CapitalCost in 2050 over PWRWND (wind)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            nuc50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL")]
            vals = to_num(nuc50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_wind_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_wind_2050_mean"] = np.nan
        
        # (8) Global discount rate (DiscountRate)
        if "DiscountRate" in sub.columns:
            vals = to_num(sub["DiscountRate"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["discount_rate_global"] = vals.iloc[0] if not vals.empty else np.nan
        else:
            f["discount_rate_global"] = np.nan
        
        # (9) Technology-specific discount rate (DiscountRateIdv), averaged over time & techs
        if "DiscountRateIdv" in sub.columns:
            vals = to_num(sub["DiscountRateIdv"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["discount_rate_idv_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["discount_rate_idv_mean"] = np.nan
        
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


def build_outcome_no_gas_2050(outputs: pd.DataFrame) -> pd.DataFrame:
    required = {"Scen_fut", "YEAR", "TECHNOLOGY", "TotalCapacityAnnual"}
    if not required.issubset(outputs.columns):
        miss = sorted(required - set(outputs.columns))
        raise ValueError(f"Outputs missing required columns: {miss}")

    out = outputs.copy()
    out["TotalCapacityAnnual"] = pd.to_numeric(out["TotalCapacityAnnual"], errors="coerce").fillna(0.0)

    gas2050 = (
        out.loc[(out["YEAR"] == 2050) & (out["TECHNOLOGY"].astype(str) == "PWRNGS001")]
        .groupby("Scen_fut")["TotalCapacityAnnual"].sum()
        .rename("gas_cap_2050")
        .reset_index()
    )

    all_scen = out["Scen_fut"].dropna().unique()
    gas2050 = gas2050.set_index("Scen_fut").reindex(all_scen, fill_value=0.0).reset_index()

    eps = 1e-6
    gas2050["no_gas_2050"] = (gas2050["gas_cap_2050"].abs() <= eps).astype(int)
    return gas2050[["Scen_fut", "gas_cap_2050", "no_gas_2050"]]

# ---- Main ------------------------------------------------------------------
if __name__ == "__main__":
    inputs = read_table(INPUTS_PATH)
    outputs = read_table(OUTPUTS_PATH)

    X = build_features(inputs)
    ydf = build_outcome_no_gas_2050(outputs)

    df = X.merge(ydf, on="Scen_fut", how="inner").dropna()
    if df.empty:
        raise SystemExit("No scenarios with complete features/outcomes after merging.")

    dataset_path = os.path.join(OUTDIR, "cart_dataset_no_gas_2050.csv")
    df.to_csv(dataset_path, index=False)

    feature_cols = [c for c in df.columns if c not in ("Scen_fut", "gas_cap_2050", "no_gas_2050")]
    Xmat = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    y = df["no_gas_2050"].values

    # CV setup
    cv = StratifiedKFold(n_splits=min(5, np.unique(y, return_counts=True)[1].min()), shuffle=True, random_state=RANDOM_STATE)

    grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced"],
        "random_state": [RANDOM_STATE],
    }

    clf = DecisionTreeClassifier()
    gs = GridSearchCV(clf, grid, scoring="balanced_accuracy", cv=cv, n_jobs=-1, refit=True, return_train_score=True)
    gs.fit(Xmat, y)

    # Save CV results
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_path = os.path.join(OUTDIR, "cart_cv_results.csv")
    cv_df.to_csv(cv_path, index=False)

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)
    print("CV balanced accuracy:", round(gs.best_score_, 3))

    # In-sample diagnostics (since data is limited, we rely on CV for generalization)
    y_pred = best.predict(Xmat)
    bal_acc = balanced_accuracy_score(y, y_pred)
    print("In-sample balanced accuracy:", round(bal_acc, 3))

    # Confusion matrix & report
    cm = confusion_matrix(y, y_pred)
    rep = classification_report(y, y_pred, digits=3)
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(os.path.join(OUTDIR, "cart_confusion_matrix.csv"))
    with open(os.path.join(OUTDIR, "cart_classification_report.txt"), "w") as f:
        f.write(rep)

    # Feature importances
    fi = pd.DataFrame({"feature": feature_cols, "importance": best.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(OUTDIR, "cart_feature_importances.csv"), index=False)

    # Tree plot
    plt.figure(figsize=(12, 8))
    plot_tree(best, feature_names=feature_cols, class_names=["gas_present","no_gas"], filled=True, rounded=True)
    plt.tight_layout()
    tree_png = os.path.join(OUTDIR, "cart_tree.png")
    plt.savefig(tree_png, dpi=200)

    # Text rules
    rules = export_text(best, feature_names=feature_cols)
    with open(os.path.join(OUTDIR, "cart_tree_rules.txt"), "w") as f:
        f.write(rules)

    print("Artifacts written to:")
    print(f"- {dataset_path}")
    print(f"- {cv_path}")
    print(f"- {os.path.join(OUTDIR, 'cart_feature_importances.csv')}")
    print(f"- {tree_png}")
    print(f"- {os.path.join(OUTDIR, 'cart_tree_rules.txt')}")
