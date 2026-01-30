"""
Unified RDM script for CART / PRIM (EMA) / PRIM (Project-Platypus) on NO GAS in 2050.

- Reads OSeMOSYS_Energy_Input / Output.
- Builds scenario-level features and a binary outcome: no_gas_2050.
- Depending on --algo, runs:
    * CART           : sklearn DecisionTreeClassifier
    * PRIM_EMA       : ema_workbench.analysis.prim.Prim
    * PRIM_PLATYPUS  : prim.Prim (Project-Platypus package)

Usage examples
--------------
python unified_rdm.py --algo CART
python unified_rdm.py --algo PRIM_EMA
python unified_rdm.py --algo PRIM_PLATYPUS
"""

import os
import warnings
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

# ---- Plotting helpers ------------------------------------------------------
        
# def plot_cart_sankey(clf, feature_names, class_names, outpath=None):
#     """
#     Build a Sankey diagram from a fitted sklearn DecisionTreeClassifier.

#     - flow width = number of samples in each child node
#     - node labels show split condition (internal) or class probabilities (leaves)
#     """
#     from sklearn.tree import _tree
#     import plotly.graph_objects as go

#     tree = clf.tree_
#     n_nodes = tree.node_count

#     # Build node labels
#     node_labels = []
#     for i in range(n_nodes):
#         is_leaf = tree.children_left[i] == _tree.TREE_LEAF
#         if not is_leaf:
#             fname = feature_names[tree.feature[i]]
#             thr = tree.threshold[i]
#             node_labels.append(f"{i}: {fname} <= {thr:.3g}")
#         else:
#             counts = tree.value[i][0]
#             total = counts.sum()
#             if total > 0:
#                 probs = counts / total
#                 cls_str = ", ".join(f"{c}={p:.2f}" for c, p in zip(class_names, probs))
#             else:
#                 cls_str = "empty"
#             node_labels.append(f"{i}: leaf\n{cls_str}")

#     # Build links (parent -> left/right child)
#     sources, targets, values = [], [], []
#     for i in range(n_nodes):
#         for child in (tree.children_left[i], tree.children_right[i]):
#             if child != _tree.TREE_LEAF:
#                 sources.append(i)
#                 targets.append(child)
#                 values.append(float(tree.n_node_samples[child]))

#     fig = go.Figure(
#         data=[
#             go.Sankey(
#                 node=dict(
#                     label=node_labels,
#                     pad=20,
#                     thickness=20,
#                 ),
#                 link=dict(
#                     source=sources,
#                     target=targets,
#                     value=values,
#                 ),
#             )
#         ]
#     )

#     if outpath is not None:
#         fig.write_html(outpath)
#     else:
#         fig.show()



# ---- Feature & outcome builders (shared across all algorithms) -------------

def build_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario-level features based on 2050 values and discount rates.
    Matches the extended feature set from the CART script.
    """
    if "Scen_fut" not in inputs.columns:
        raise ValueError("Expected 'Scen_fut' key column in inputs.")

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    feats = []
    for scen, sub in inputs.groupby("Scen_fut", sort=False):
        f = {"Scen_fut": scen}
        sub50 = sub.loc[sub.get("YEAR").eq(YEAR) if "YEAR" in sub.columns else []]

        # (1) SpecifiedAnnualDemand in 2050 summed
        f["demand_2050_sum"] = to_num(sub50.get("SpecifiedAnnualDemand", np.nan)).sum()

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
            bat50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("BESS_TECH")]
            vals = to_num(bat50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_bess_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_bess_2050_mean"] = np.nan

        # (6) CapitalCost in 2050 over PWRSOL (solar)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            sol50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRSOL")]
            vals = to_num(sol50["CapitalCost"])
            vals = vals[(~vals.isna()) & (vals != 0)]
            f["capex_solar_2050_mean"] = vals.mean() if not vals.empty else np.nan
        else:
            f["capex_solar_2050_mean"] = np.nan

        # (7) CapitalCost in 2050 over PWRWND (wind)
        if {"TECHNOLOGY", "CapitalCost"}.issubset(sub50.columns):
            wnd50 = sub50.loc[sub50["TECHNOLOGY"].astype(str).str.startswith("PWRWND")]
            vals = to_num(wnd50["CapitalCost"])
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
            bat50 = sub.loc[
                (sub["TECHNOLOGY"].astype(str).str.startswith("BESS_TECH"))
                & (sub["YEAR"] == YEAR)
            ]
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

    nunique = X.drop(columns=["Scen_fut"]).nunique(dropna=True)
    print("Feature variability (nunique):")
    print(nunique.sort_values())

    return X


def build_outcome_no_gas_2050(outputs: pd.DataFrame) -> pd.DataFrame:
    """
    Binary outcome per scenario: 1 if no gas capacity (PWRNGS001) in 2050, else 0.
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
    gas2050 = gas2050.set_index("Scen_fut").reindex(all_scen, fill_value=0.0).reset_index()

    eps = 1e-6
    gas2050["no_gas_2050"] = (gas2050["gas_cap_2050"].abs() <= eps).astype(int)

    return gas2050[["Scen_fut", "gas_cap_2050", "no_gas_2050"]]


# ---- CART runner -----------------------------------------------------------

def run_cart(df: pd.DataFrame, feature_cols: List[str], outdir: str) -> None:
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

    os.makedirs(outdir, exist_ok=True)

    dataset_path = os.path.join(outdir, f"cart_dataset_no_gas_{YEAR}.csv")
    df.to_csv(dataset_path, index=False)

    Xmat = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    y = df["no_gas_2050"].values

    # CV setup
    cv = StratifiedKFold(
        n_splits=min(5, np.unique(y, return_counts=True)[1].min()),
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced"],
        "random_state": [RANDOM_STATE],
    }

    clf = DecisionTreeClassifier()
    gs = GridSearchCV(
        clf,
        grid,
        scoring="balanced_accuracy",
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
        cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
    ).to_csv(os.path.join(outdir, "cart_confusion_matrix.csv"), index=False)
    with open(os.path.join(outdir, "cart_classification_report.txt"), "w") as f:
        f.write(rep)

    # Feature importances
    fi = (
        pd.DataFrame({"feature": feature_cols, "importance": best.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    fi.to_csv(os.path.join(outdir, "cart_feature_importances.csv"), index=False)
   
    # Tree plot & rules
    if plt is not None:
        plt.figure(figsize=(12, 8))
        plot_tree(
            best,
            feature_names=feature_cols,
            class_names=["gas_present", "no_gas"],
            filled=True,
            rounded=True,
        )
        plt.tight_layout()
        tree_png = os.path.join(outdir, "cart_tree.png")
        plt.savefig(tree_png, dpi=200)
    
    rules = export_text(best, feature_names=feature_cols)
    with open(os.path.join(outdir, "cart_tree_rules.txt"), "w") as f:
        f.write(rules)
    
    # # NEW: Sankey flow visualisation (requires: pip install plotly)
    # class_names = [str(c) for c in best.classes_]
    # sankey_html = os.path.join(outdir, "cart_sankey.html")
    # plot_cart_sankey(best, feature_cols, class_names, sankey_html)
    
    print("CART artifacts written to:", outdir)


# ---- PRIM (EMA) runner -----------------------------------------------------

def run_prim_ema(df: pd.DataFrame, feature_cols: List[str], outdir: str) -> None:
    try:
        from ema_workbench.analysis import prim as ema_prim
    except Exception as e:
        raise SystemExit(
            "EMA-PRIM requires `ema_workbench` to be installed "
            "(pip install ema_workbench).\n"
            f"Original import error: {e}"
        )

    os.makedirs(outdir, exist_ok=True)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["no_gas_2050"].values  # 1 = no gas (cases of interest)

    p = ema_prim.Prim(X, y, threshold=0.5)  # focus on high no_gas
    box = p.find_box()

    # Tradeoff plot
    try:
        fig = box.show_tradeoff()
        if fig is not None and plt is not None:
            trade_png = os.path.join(outdir, f"prim_ema_tradeoff_no_gas_{YEAR}.png")
            fig.savefig(trade_png, dpi=200, bbox_inches="tight")
    except Exception as e:
        print("PRIM (EMA) tradeoff plot failed:", e)

    # Pairwise scatter plot
    try:
        g = box.show_pairs_scatter()
        if g is not None:
            pairs_png = os.path.join(outdir, f"prim_ema_pairs_no_gas_{YEAR}.png")
            g.fig.savefig(pairs_png, dpi=200, bbox_inches="tight")
    except Exception as e:
        print("PRIM (EMA) pairs plot failed:", e)

    # Box limits & stats (if available)
    limits_path = os.path.join(outdir, f"prim_ema_box_limits_no_gas_{YEAR}.csv")
    stats_path = os.path.join(outdir, f"prim_ema_box_stats_no_gas_{YEAR}.csv")
    try:
        data = box.inspect(style="data")  # list of (stats, box_lims)
        if isinstance(data, list) and data:
            stats, box_lims = data[-1]
            pd.DataFrame([stats]).to_csv(stats_path, index=False)
            box_lims.to_csv(limits_path, index=False)
    except Exception as e:
        print("PRIM (EMA) inspect failed:", e)

    print("PRIM (EMA) artifacts written to:", outdir)


# ---- PRIM (Project-Platypus) runner ---------------------------------------

def run_prim_platypus(df: pd.DataFrame, feature_cols: List[str], outdir: str) -> None:
    try:
        import prim  # Project-Platypus PRIM package
    except Exception as e:
        raise SystemExit(
            "Project-Platypus PRIM requires the `prim` package "
            "(pip install prim).\n"
            f"Original import error: {e}"
        )

    os.makedirs(outdir, exist_ok=True)

    Xnum = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["no_gas_2050"]

    print("Running PRIM (Project-Platypus) for NO gas in 2050…")
    p = prim.Prim(Xnum, y, threshold=0.5, threshold_type=">")
    box = p.find_box()

    # Plots
    try:
        fig = box.show_tradeoff()
        if fig is not None and plt is not None:
            fig.savefig(
                os.path.join(outdir, f"prim_platypus_tradeoff_no_gas_{YEAR}.png"),
                bbox_inches="tight",
                dpi=200,
            )
    except Exception:
        warnings.warn("Could not create PRIM (Platypus) tradeoff plot; continuing.")

    try:
        g = box.show_pairs_scatter()
        if g is not None:
            g.fig.savefig(
                os.path.join(outdir, f"prim_platypus_pairs_no_gas_{YEAR}.png"),
                bbox_inches="tight",
                dpi=200,
            )
    except Exception:
        warnings.warn("Could not create PRIM (Platypus) pairs scatter; continuing.")

    # Export box limits/statistics if available
    limits_path = os.path.join(outdir, f"prim_platypus_box_limits_no_gas_{YEAR}.csv")
    stats_path = os.path.join(outdir, f"prim_platypus_box_stats_no_gas_{YEAR}.csv")
    try:
        data = box.inspect(style="data")
        if isinstance(data, list) and data:
            stats, box_lims = data[-1]
            pd.DataFrame([stats]).to_csv(stats_path, index=False)
            box_lims.to_csv(limits_path, index=False)
    except Exception:
        pass

    print("PRIM (Platypus) artifacts written to:", outdir)


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified RDM script: CART / PRIM_EMA / PRIM_PLATYPUS on no-gas-2050 outcome."
    )
    parser.add_argument(
        "--algo",
        choices=["CART", "PRIM_EMA", "PRIM_PLATYPUS"],
        default="PRIM_EMA",
        help="Algorithm to run (default: CART).",
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

    feature_cols = [c for c in df.columns if c not in ("Scen_fut", "gas_cap_2050", "no_gas_2050")]

    n = len(df)
    n_no = int(df["no_gas_2050"].sum())
    n_yes = n - n_no
    print(
        f"Scenarios: {n}  —  no_gas_2050 = 1 in {n_no} ({n_no / n:.1%}); "
        f"0 in {n_yes} ({n_yes / n:.1%})"
    )
    print("Features used:", feature_cols)

    outdir = os.path.join(OUTDIR_BASE, args.algo.lower())
    if args.algo == "CART":
        run_cart(df, feature_cols, outdir)
    elif args.algo == "PRIM_EMA":
        run_prim_ema(df, feature_cols, outdir)
    elif args.algo == "PRIM_PLATYPUS":
        run_prim_platypus(df, feature_cols, outdir)
    else:
        raise SystemExit(f"Unknown algo: {args.algo}")


if __name__ == "__main__":
    main()
