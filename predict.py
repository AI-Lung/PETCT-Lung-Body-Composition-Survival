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
    "IMAT_volume_index", "IMF_ratio", "TAT_volume_index",
    "SM_SURsd", "TAT_SURmean", "IMAT_SURsd", "SM_SURmean", "TAT_density_mean",
]


def load_model(pkl_path: str) -> Tuple[object, List[str]]:
    """Load RSF model from pkl. Returns (model, feature_names)."""
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
    """Returns: (risk_score, time_points, survival_probabilities)."""
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


def risk_group_from_cutoff(risk_score: float, cutoff: Optional[float], default_high_quantile: float = 0.5) -> str:
    """Assign High / Low risk."""
    if cutoff is not None:
        return "High" if risk_score >= cutoff else "Low"
    return "High" if risk_score >= 0 else "Low"


def plot_survival_curve(
    time_points: np.ndarray,
    survival_probs: np.ndarray,
    title: str,
    risk_group: str,
    outcome_label: str,
) -> plt.Figure:
    """Plot single survival curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(time_points, 0, survival_probs, alpha=0.3)
    ax.plot(time_points, survival_probs, color="#2E86AB", linewidth=2.5, label=f"Predicted survival ({risk_group} risk)")
    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    ax.set_title(f"{title}\n{outcome_label} — Risk group: {risk_group}", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
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
    """Run OS and PFS models; return dict with risk scores, groups, survival curves (figures)."""
    results = {
        "OS": {"risk_score": None, "risk_group": None, "time_points": None, "survival_probs": None, "figure": None, "feature_names": None},
        "PFS": {"risk_score": None, "risk_group": None, "time_points": None, "survival_probs": None, "figure": None, "feature_names": None},
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

    try:
        risk_os, t_os, s_os = predict_risk_and_survival(model_os, features_df, feat_os, time_max=time_max_os)
        results["OS"]["risk_score"] = risk_os
        results["OS"]["risk_group"] = risk_group_from_cutoff(risk_os, cutoff_os)
        results["OS"]["time_points"] = t_os
        results["OS"]["survival_probs"] = s_os
        results["OS"]["figure"] = plot_survival_curve(t_os, s_os, "Overall Survival (OS)", results["OS"]["risk_group"], "OS")
    except Exception as e:
        results["OS"]["error"] = str(e)

    try:
        risk_pfs, t_pfs, s_pfs = predict_risk_and_survival(model_pfs, features_df, feat_pfs, time_max=time_max_pfs)
        results["PFS"]["risk_score"] = risk_pfs
        results["PFS"]["risk_group"] = risk_group_from_cutoff(risk_pfs, cutoff_pfs)
        results["PFS"]["time_points"] = t_pfs
        results["PFS"]["survival_probs"] = s_pfs
        results["PFS"]["figure"] = plot_survival_curve(t_pfs, s_pfs, "Progression-Free Survival (PFS)", results["PFS"]["risk_group"], "PFS")
    except Exception as e:
        results["PFS"]["error"] = str(e)

    return results
