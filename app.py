# -*- coding: utf-8 -*-
"""
Streamlit app: Upload CT and PET NIfTI, weight and dose, run segmentation and feature extraction,
then RSF-based OS/PFS risk stratification and personalized survival curves.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extraction import extract_all_features, ALL_FEATURE_NAMES
from segmentation import run_segmentation_pipeline
from predict import run_os_pfs_predictions

# Page config
st.set_page_config(page_title="PET/CT Body Composition & Survival Risk", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ============== Custom CSS ==============
st.markdown("""
<style>
    /* Main theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%); }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3b82f6 100%);
        padding: 2rem 2rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 58, 95, 0.25);
        color: white;
    }
    .hero h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0;
        opacity: 0.95;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Section cards */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    .section-card h3 {
        color: #1e3a5f;
        font-size: 1.15rem;
        margin-top: 0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .risk-high { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; }
    .risk-low { background: linear-gradient(135deg, #059669, #10b981); color: white; }
    .risk-na { background: #64748b; color: white; }
    
    /* Step labels */
    .step-label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #3b82f6;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    [data-testid="stSidebar"] label { color: #cbd5e1 !important; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.45);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] { font-size: 1.15rem !important; font-weight: 700 !important; color: #1e3a5f !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #64748b !important; }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.4rem 0.6rem !important;
        margin-bottom: 0.3rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader { background: #f1f5f9; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <h1>🩺 PET/CT Body Composition & Survival Risk</h1>
    <p>Upload CT and PET NIfTI images with body weight and injection dose. The pipeline runs TotalSegmentator, extracts body composition and PET-derived features, then predicts OS and PFS risk groups and personalized survival curves using trained RSF models.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: model paths and options
st.sidebar.markdown("### ⚙️ Model & options")
model_dir = st.sidebar.text_input(
    "Folder containing RSF models",
    value=str(SCRIPT_DIR / "Model"),
    help="Folder with RSF_best_model_OS.pkl and RSF_best_model_PFS.pkl",
)
model_os_path = Path(model_dir) / "RSF_best_model_OS.pkl"
model_pfs_path = Path(model_dir) / "RSF_best_model_PFS.pkl"
use_gpu = st.sidebar.checkbox("Use GPU for segmentation", value=True)
device = "cuda" if use_gpu else "cpu"
st.sidebar.markdown("#### Memory optimization")
force_split = st.sidebar.checkbox(
    "Force split (low RAM mode)",
    value=True,
    help="Process image in 3 chunks to reduce RAM usage. Recommended if you have < 32 GB RAM.",
)
fast_mode = st.sidebar.checkbox(
    "Fast mode (3 mm resolution)",
    value=False,
    help="Use lower resolution model (3 mm). Much faster and uses less RAM, but slightly less accurate.",
)
nr_thr_resamp = st.sidebar.number_input("Resampling threads", min_value=1, max_value=8, value=1, help="Fewer threads = less RAM. Default 1 is safest.")
nr_thr_saving = st.sidebar.number_input("Saving threads", min_value=1, max_value=8, value=1, help="Fewer threads = less RAM. Default 1 is safest.")
st.sidebar.markdown("#### Timeout")
timeout_total_min = st.sidebar.number_input("Timeout for whole-body seg (minutes)", min_value=10, max_value=120, value=90, help="Increase if segmentation times out (e.g. on CPU).")
timeout_tissue_min = st.sidebar.number_input("Timeout for 4-tissue seg (minutes)", min_value=5, max_value=60, value=40)
skip_segmentation = st.sidebar.checkbox(
    "Skip segmentation (use existing mask folders)",
    value=False,
    help="If you already have seg_total and seg_4tissue folders, set paths in the form.",
)

# Input section in card style
st.markdown('<div class="section-card"><h3>📁 1. Input data</h3>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    ct_file = st.file_uploader("**CT image** (NIfTI .nii / .nii.gz)", type=["nii", "nii.gz"], key="ct")
with col2:
    pet_file = st.file_uploader("**PET image** (NIfTI .nii / .nii.gz)", type=["nii", "nii.gz"], key="pet")

st.markdown("**Patient & acquisition**")
r1, r2, r3 = st.columns(3)
with r1:
    weight_kg = st.number_input("Body weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1)
with r2:
    dose_mbq = st.number_input("Injected activity (MBq)", min_value=0.1, max_value=1000.0, value=300.0, step=1.0)
with r3:
    height_cm = st.number_input("Height (cm, optional)", min_value=0.0, max_value=250.0, value=0.0, step=0.5,
                                help="Set 0 to skip; used for TAT volume index (TAT/height²).")
pet_is_suv = st.checkbox("PET image is already in SUV (no conversion)", value=False)

mask_dir_total_input = None
mask_dir_4tissue_input = None
if skip_segmentation:
    st.markdown("**Existing segmentation paths**")
    mask_dir_total_input = st.text_input("Whole-body segmentation folder (bones, liver)", placeholder="/path/to/seg_total")
    mask_dir_4tissue_input = st.text_input("4-tissue segmentation folder", placeholder="/path/to/seg_4tissue")

st.markdown("</div>", unsafe_allow_html=True)
run_button = st.button("▶ Run pipeline: Segmentation → Features → Risk & Survival")

if run_button:
    if not ct_file:
        st.error("Please upload a CT NIfTI file.")
        st.stop()
    if not pet_file:
        st.error("Please upload a PET NIfTI file.")
        st.stop()
    if not Path(model_os_path).exists():
        st.error(f"OS model not found at {model_os_path}. Please place RSF_best_model_OS.pkl in the Model folder.")
        st.stop()
    if not Path(model_pfs_path).exists():
        st.error(f"PFS model not found at {model_pfs_path}. Please place RSF_best_model_PFS.pkl in the Model folder.")
        st.stop()

    work_dir = tempfile.mkdtemp(prefix="petct_streamlit_")
    ct_path = os.path.join(work_dir, "ct.nii.gz")
    pet_path = os.path.join(work_dir, "pet.nii.gz")

    with open(ct_path, "wb") as f:
        f.write(ct_file.getvalue())
    with open(pet_path, "wb") as f:
        f.write(pet_file.getvalue())

    mask_dir_total = None
    mask_dir_4tissue = None

    # Step 1: Segmentation
    if not skip_segmentation:
        st.markdown('<div class="step-label">Step 1 — Segmentation</div>', unsafe_allow_html=True)
        progress = st.progress(0)
        with st.spinner("Running TotalSegmentator (whole-body and 4-tissue). This may take 20–90 min on CPU; less on GPU."):
            mask_dir_total, mask_dir_4tissue, msg = run_segmentation_pipeline(
                ct_path, work_dir, device=device,
                timeout_total=timeout_total_min * 60,
                timeout_tissue=timeout_tissue_min * 60,
                fast=fast_mode,
                force_split=force_split,
                nr_thr_resamp=nr_thr_resamp,
                nr_thr_saving=nr_thr_saving,
            )
        progress.progress(100)
        if mask_dir_total is None or mask_dir_4tissue is None:
            st.error(msg)
            if "memoryerror" in msg.lower() or "unable to allocate" in msg.lower():
                st.info("**RAM not enough.** Try: enable **Fast mode (3mm)** and **Force split** in the sidebar. Close other programs to free memory.")
            else:
                st.info("**Tips:** Check sidebar options. Use GPU if available. If timeout, increase timeout.")
            st.stop()
        st.success(msg)
    else:
        if mask_dir_total_input and mask_dir_4tissue_input and os.path.isdir(mask_dir_total_input) and os.path.isdir(mask_dir_4tissue_input):
            mask_dir_total = mask_dir_total_input
            mask_dir_4tissue = mask_dir_4tissue_input
        else:
            st.error("Skip segmentation is selected but valid paths to mask folders were not provided.")
            st.stop()

    # Step 2: Feature extraction
    st.markdown('<div class="step-label">Step 2 — Feature extraction</div>', unsafe_allow_html=True)
    try:
        height = height_cm if height_cm > 0 else None
        features_df = extract_all_features(
            ct_path, pet_path,
            mask_dir_4tissue, mask_dir_total,
            weight_kg=weight_kg, dose_mbq=dose_mbq,
            height_cm=height, pet_is_suv=pet_is_suv,
        )
        st.success(f"Extracted {len(ALL_FEATURE_NAMES)} features.")
        with st.expander("📊 View extracted features", expanded=True):
            feat_groups = {
                "Skeletal Muscle (SM)": ["SM_volume_index", "SM_density_mean", "SM_density_sd", "SM_SURmean", "SM_SURsd"],
                "Subcutaneous Fat (SAT)": ["SAT_volume_index", "SAT_density_mean", "SAT_density_sd", "SAT_SURmean", "SAT_SURsd"],
                "Intermuscular Fat (IMAT)": ["IMAT_volume_index", "IMF_ratio", "IMAT_density_mean", "IMAT_density_sd", "IMAT_SURmean", "IMAT_SURsd"],
                "Torso Fat (TAT)": ["TAT_volume_index", "TF_SAT_ratio", "TAT_density_mean", "TAT_density_sd", "TAT_SURmean", "TAT_SURsd"],
                "Bone": ["Bone_density_mean", "Bone_density_sd"],
            }
            feat_cols = st.columns(len(feat_groups))
            for col, (group_name, keys) in zip(feat_cols, feat_groups.items()):
                with col:
                    st.markdown(f"**{group_name}**")
                    for k in keys:
                        val = features_df[k].values[0] if k in features_df.columns else 0.0
                        label = k.split("_", 1)[-1] if "_" in k else k
                        st.metric(label=label, value=f"{val:.4f}")
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    # Step 3: Risk and survival
    st.markdown('<div class="step-label">Step 3 — Risk stratification & survival curves</div>', unsafe_allow_html=True)
    try:
        results = run_os_pfs_predictions(
            features_df,
            str(model_os_path),
            str(model_pfs_path),
            cutoff_os=None,
            cutoff_pfs=None,
            time_max_os=60.0,
            time_max_pfs=60.0,
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    if results.get("error"):
        st.error(results["error"])
        st.stop()

    # Output: Risk stratification with styled badges
    st.markdown("---")
    st.markdown("### 📈 Risk stratification results")
    col_os, col_pfs = st.columns(2)
    os_res = results["OS"]
    pfs_res = results["PFS"]
    with col_os:
        rg_os = os_res.get("risk_group", "—")
        cls_os = "risk-high" if rg_os == "High" else ("risk-low" if rg_os == "Low" else "risk-na")
        st.markdown(f'<div class="section-card"><h3>Overall Survival (OS)</h3><p><span class="risk-badge {cls_os}">{rg_os}</span></p>', unsafe_allow_html=True)
        if os_res.get("risk_score") is not None:
            st.caption(f"Risk score: {os_res['risk_score']:.4f}")
        if os_res.get("error"):
            st.warning(os_res["error"])
        st.markdown("</div>", unsafe_allow_html=True)
    with col_pfs:
        rg_pfs = pfs_res.get("risk_group", "—")
        cls_pfs = "risk-high" if rg_pfs == "High" else ("risk-low" if rg_pfs == "Low" else "risk-na")
        st.markdown(f'<div class="section-card"><h3>Progression-Free Survival (PFS)</h3><p><span class="risk-badge {cls_pfs}">{rg_pfs}</span></p>', unsafe_allow_html=True)
        if pfs_res.get("risk_score") is not None:
            st.caption(f"Risk score: {pfs_res['risk_score']:.4f}")
        if pfs_res.get("error"):
            st.warning(pfs_res["error"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Survival curves in cards
    st.markdown("### 📉 Personalized survival curves")
    fig_os = results["OS"].get("figure")
    fig_pfs = results["PFS"].get("figure")
    c1, c2 = st.columns(2)
    if fig_os:
        with c1:
            st.markdown('<div class="section-card"><h3>Overall Survival (OS)</h3>', unsafe_allow_html=True)
            st.pyplot(fig_os)
            plt.close(fig_os)
            st.markdown("</div>", unsafe_allow_html=True)
    if fig_pfs:
        with c2:
            st.markdown('<div class="section-card"><h3>Progression-Free Survival (PFS)</h3>', unsafe_allow_html=True)
            st.pyplot(fig_pfs)
            plt.close(fig_pfs)
            st.markdown("</div>", unsafe_allow_html=True)

    st.success("Analysis complete. Risk groups are derived from the RSF model risk score. Survival curves show the model-predicted survival probability over time (months).")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='color:#94a3b8; font-size:0.85rem;'>PET/CT Body Composition & RSF Survival Model — Upload CT/PET to get risk stratification and survival curves.</p>",
    unsafe_allow_html=True,
)
