# PET/CT Body Composition & Survival Risk

A Streamlit app for **CT/PET NIfTI** upload, **TotalSegmentator**-based body composition segmentation, feature extraction, and **RSF**-based OS/PFS risk stratification with personalized survival curves.

## Overview

1. **Segmentation** — TotalSegmentator (whole-body + 4-tissue: SM, SAT, IMAT, TAT)
2. **Feature extraction** — 24 CT- and PET-derived features (volumes, densities, SUR)
3. **Risk & survival** — Trained RSF models predict risk groups and survival curves

## Requirements

- Python 3.9+
- Install: `pip install -r requirements.txt`
- **TotalSegmentator** is used for segmentation; on low-RAM systems the app auto-retries with `--fast` / `--force_split` / `--body_seg`.

## Model files

Place your trained RSF models in the `Model` folder:

- `RSF_best_model_OS.pkl` — Overall Survival
- `RSF_best_model_PFS.pkl` — Progression-Free Survival

See `Model/README.md` for details.

## Run locally

```bash
# Option 1: direct
streamlit run app.py

# Option 2: use dl_env if you have conda
python run_streamlit.py

# Windows: double-click run.bat (uses dl_env if found)
```

Then open the URL shown (default http://localhost:8501).

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → select your repo, branch `main`, main file path: **`app.py`**.
4. Click **Deploy**. Streamlit Cloud will install from `requirements.txt`.

**Note:** TotalSegmentator is heavy (PyTorch, nnUNet). On Streamlit Cloud you may hit memory limits for on-the-fly segmentation. For cloud use, consider **Skip segmentation** and providing pre-computed mask folder paths, or run segmentation locally and use the app for features + risk only.

## Inputs

- **CT** and **PET** NIfTI (`.nii` or `.nii.gz`)
- **Body weight (kg)** and **Injected activity (MBq)** for SUV conversion (unless PET is already SUV)
- **Height (cm)** optional; used for TAT volume index (TAT/height²)

## Outputs

- Risk stratification: OS and PFS risk groups (High/Low) and risk scores
- Personalized survival curves: predicted survival probability over time (months)

## License

Use and cite according to your project and the TotalSegmentator / scikit-survival licenses.
