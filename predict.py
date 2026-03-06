# -*- coding: utf-8 -*-
"""
Load RSF models and predict risk stratification and survival curves for OS and PFS.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

DEFAULT_OS_FEATURES = [
    "IMAT_volume_index", "IMF_ratio", "TAT_volume_index",
    "SM_SURsd", "TAT_SURmean", "IMAT_SURsd", "SM_SURmean", "TAT_density_mean",
]
DEFAULT_PFS_FEATURES = [
    "IMAT_volume_index", "IMF_ratio", "SM_SURsd", "IMAT_SURsd", "TAT_SURmean",
]

TRAINING_MEDIAN_CUTOFF_OS = 3.8762
TRAINING_MEDIAN_CUTOFF_PFS = 6.7220


def load_model(pkl_path: str) -> Tuple[object, List[str]]:
    """
    Load RSF model from pkl. Returns (model, feature_names).
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    model = obj
    feature_names = None

    if isinstance(obj, dict):
        model = obj.get("model", obj)
        feature_names = obj.get("feature_names") or obj.get("feature_names_in_")

    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    if feature_names is None:
        feature_names = DEFAULT_OS_FEATURES

    return model, feature_names


def prepare_features(X: pd.DataFrame, feature_names: List[str], fill_missing: float = 0.0) -> pd.DataFrame:
    """Build DataFrame with columns in feature_names; missing columns filled with fill_missing."""
    out = {}
    for col in feature_names:
        if col in X.columns:
            out[col] = X[col].values
        else:
            out[col] = np.full(len(X), fill_missing)
    return pd.DataFrame(out, columns=feature_names)


def predict_risk_and_survival(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
    time_max: float = 60.0,
    n_points: int = 100,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns: (risk_score, time_points, survival_probabilities).
    """
    X_use = prepare_features(X, feature_names)
    risk_score = float(model.predict(X_use)[0])
    surv_funcs = model.predict_survival_function(X_use, return_array=False)
    sf = surv_funcs[0]
    if hasattr(sf, "x") and hasattr(sf, "y"):
        t_orig = np.asarray(sf.x)
        s_orig = np.asarray(sf.y)
    else:
        t_orig = np.linspace(0, time_max, 50)
        s_orig = np.array([float(sf(t)) for t in t_orig])
    time_points = np.linspace(0, time_max, n_points)
    survival_probs = np.interp(time_points, t_orig, s_orig)
    survival_probs = np.clip(survival_probs, 0, 1)
    return risk_score, time_points, survival_probs


def predict_population_quantile_curves(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
    training_csv_path: Optional[str] = None,
    time_max: float = 60.0,
    n_points: int = 100,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute 25th/50th/75th percentile survival curves from training data
    to provide reference bands on the plot.
    """
    if training_csv_path and os.path.isfile(training_csv_path):
        try:
            train_df = pd.read_csv(training_csv_path)
            X_train = prepare_features(train_df, feature_names)
        except Exception:
            return None
    else:
        return None

    surv_funcs = model.predict_survival_function(X_train, return_array=False)
    time_points = np.linspace(0, time_max, n_points)
    all_curves = []
    for sf in surv_funcs:
        if hasattr(sf, "x") and hasattr(sf, "y"):
            curve = np.interp(time_points, sf.x, sf.y)
        else:
            curve = np.array([float(sf(t)) for t in time_points])
        all_curves.append(np.clip(curve, 0, 1))
    all_curves = np.array(all_curves)
    return {
        "time": time_points,
        "p25": np.percentile(all_curves, 25, axis=0),
        "p50": np.percentile(all_curves, 50, axis=0),
        "p75": np.percentile(all_curves, 75, axis=0),
    }


def risk_group_from_cutoff(risk_score: float, cutoff: Optional[float]) -> str:
    """Assign High / Low risk based on cutoff (training-set median by default)."""
    if cutoff is not None:
        return "High" if risk_score >= cutoff else "Low"
    return "Unknown"


def plot_survival_curve(
    time_points: np.ndarray,
    survival_probs: np.ndarray,
    title: str,
    risk_group: str,
    risk_score: float,
    outcome_label: str,
    ref_curves: Optional[Dict[str, np.ndarray]] = None,
) -> plt.Figure:
    """Plot personalized survival curve with optional population reference band."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    if ref_curves is not None:
        ax.fill_between(
            ref_curves["time"], ref_curves["p25"], ref_curves["p75"],
            alpha=0.15, color="#94a3b8", label="Population IQR (25th–75th)",
        )
        ax.plot(
            ref_curves["time"], ref_curves["p50"],
            color="#94a3b8", linewidth=1.5, linestyle="--", label="Population median",
        )

    color = "#dc2626" if risk_group == "High" else "#059669" if risk_group == "Low" else "#2E86AB"
    ax.fill_between(time_points, 0, survival_probs, alpha=0.18, color=color)
    ax.plot(
        time_points, survival_probs, color=color, linewidth=2.8,
        label=f"This patient ({risk_group} risk, score={risk_score:.2f})",
    )

    y_min_data = max(0, float(np.min(survival_probs)) - 0.10)
    ax.set_ylim(y_min_data, 1.02)

    ax.set_xlabel("Time (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Survival probability", fontsize=12, fontweight="bold")
    ax.set_title(f"{title}\n{outcome_label} — Risk group: {risk_group} (score={risk_score:.2f})", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def run_os_pfs_predictions(
    features_df: pd.DataFrame,
    model_os_path: str,
    model_pfs_path: str,
    cutoff_os: Optional[float] = None,
    cutoff_pfs: Optional[float] = None,
    time_max_os: float = 60.0,
    time_max_pfs: float = 60.0,
) -> Dict:
    """
    Run OS and PFS models; return dict with risk scores, groups, survival curves (figures), and feature names used.
    """
    if cutoff_os is None:
        cutoff_os = TRAINING_MEDIAN_CUTOFF_OS
    if cutoff_pfs is None:
        cutoff_pfs = TRAINING_MEDIAN_CUTOFF_PFS

    results = {
        "OS": {"risk_score": None, "risk_group": None, "time_points": None, "survival_probs": None, "figure": None, "feature_names": None, "features_used": None},
        "PFS": {"risk_score": None, "risk_group": None, "time_points": None, "survival_probs": None, "figure": None, "feature_names": None, "features_used": None},
        "error": None,
    }

    try:
        model_os, feat_os = load_model(model_os_path)
        results["OS"]["feature_names"] = feat_os
    except Exception as e:
        results["error"] = f"Failed to load OS model: {e}"
        return results

    try:
        model_pfs, feat_pfs = load_model(model_pfs_path)
        results["PFS"]["feature_names"] = feat_pfs
    except Exception as e:
        results["error"] = f"Failed to load PFS model: {e}"
        return results

    # OS
    try:
        X_os = prepare_features(features_df, feat_os)
        results["OS"]["features_used"] = {col: float(X_os[col].values[0]) for col in feat_os}

        risk_os, t_os, s_os = predict_risk_and_survival(model_os, features_df, feat_os, time_max=time_max_os)
        results["OS"]["risk_score"] = risk_os
        results["OS"]["risk_group"] = risk_group_from_cutoff(risk_os, cutoff_os)
        results["OS"]["time_points"] = t_os
        results["OS"]["survival_probs"] = s_os
        results["OS"]["figure"] = plot_survival_curve(
            t_os, s_os,
            title="Overall Survival (OS)",
            risk_group=results["OS"]["risk_group"],
            risk_score=risk_os,
            outcome_label="OS",
        )
    except Exception as e:
        results["OS"]["error"] = str(e)

    # PFS
    try:
        X_pfs = prepare_features(features_df, feat_pfs)
        results["PFS"]["features_used"] = {col: float(X_pfs[col].values[0]) for col in feat_pfs}

        risk_pfs, t_pfs, s_pfs = predict_risk_and_survival(model_pfs, features_df, feat_pfs, time_max=time_max_pfs)
        results["PFS"]["risk_score"] = risk_pfs
        results["PFS"]["risk_group"] = risk_group_from_cutoff(risk_pfs, cutoff_pfs)
        results["PFS"]["time_points"] = t_pfs
        results["PFS"]["survival_probs"] = s_pfs
        results["PFS"]["figure"] = plot_survival_curve(
            t_pfs, s_pfs,
            title="Progression-Free Survival (PFS)",
            risk_group=results["PFS"]["risk_group"],
            risk_score=risk_pfs,
            outcome_label="PFS",
        )
    except Exception as e:
        results["PFS"]["error"] = str(e)

    return results
