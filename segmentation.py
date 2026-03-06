# -*- coding: utf-8 -*-
"""
Run TotalSegmentator for whole-body (bones, liver) and tissue_4_types (SM, SAT, IMAT, TAT).
Supports CLI executable with auto-retry on MemoryError.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional

TASK_TOTAL = "total"
TASK_TISSUE_4 = "tissue_4_types"
ENV_NAME = "dl_env"


def _get_conda_exe() -> Optional[str]:
    """If current Python is from Anaconda/Miniconda, return path to conda.exe."""
    exe = os.path.normpath(sys.executable)
    for name in ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3"):
        if name in exe:
            base = os.path.dirname(exe)
            if "envs" in exe.split(os.sep):
                idx = exe.split(os.sep).index("envs")
                base = os.sep.join(exe.split(os.sep)[:idx])
            conda_exe = os.path.join(base, "Scripts", "conda.exe")
            if os.path.isfile(conda_exe):
                return conda_exe
            conda_bat = os.path.join(base, "Scripts", "conda.bat")
            if os.path.isfile(conda_bat):
                return conda_bat
            break
    return None


def _find_dl_env_python() -> Optional[str]:
    """Return path to dl_env's Python."""
    if os.environ.get("CONDA_DEFAULT_ENV") == ENV_NAME:
        return sys.executable
    exe = os.path.normpath(sys.executable)
    if "envs" in exe.split(os.sep):
        parts = exe.split(os.sep)
        idx = parts.index("envs")
        parent = parts[idx + 1] if idx + 1 < len(parts) else ""
        if parent.lower() == ENV_NAME:
            return sys.executable

    try:
        r = subprocess.run(
            ["conda", "run", "-n", ENV_NAME, "python", "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout and os.path.isfile(r.stdout.strip()):
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    if sys.platform == "win32":
        base = os.sep.join(exe.split(os.sep)[:exe.split(os.sep).index("envs")]) if "envs" in exe.split(os.sep) else os.path.dirname(exe)
        py = os.path.join(base, "envs", ENV_NAME, "python.exe")
    else:
        base = os.sep.join(exe.split(os.sep)[:exe.split(os.sep).index("envs")]) if "envs" in exe.split(os.sep) else os.path.dirname(exe)
        py = os.path.join(base, "envs", ENV_NAME, "bin", "python")
    if os.path.isfile(py):
        return py

    app_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in [ENV_NAME, "..", os.path.join("..", ENV_NAME), os.path.join("..", "..", ENV_NAME)]:
        base = os.path.normpath(os.path.join(app_dir, rel))
        py = os.path.join(base, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(base, "bin", "python")
        if os.path.isfile(py):
            return py
    return None


def _device_to_cli(device: str) -> str:
    if device.lower() in ("cuda", "gpu"):
        return "gpu"
    return "cpu"


def _find_totalseg_exe(python_exe: str) -> Optional[str]:
    scripts_dir = os.path.join(os.path.dirname(python_exe), "Scripts")
    if not os.path.isdir(scripts_dir):
        scripts_dir = os.path.dirname(python_exe)
    for name in ("TotalSegmentator.exe", "TotalSegmentator", "totalsegmentator.exe", "totalsegmentator"):
        p = os.path.join(scripts_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _filter_error(text: str) -> str:
    lines = text.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(ch in stripped for ch in ("\u2588", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589", "it/s", "s/it")):
            continue
        if stripped.startswith("0%|") or stripped.endswith("?it/s]"):
            continue
        filtered.append(stripped)
    return "\n".join(filtered) if filtered else text[-1000:]


def _is_memory_error(err_text: str) -> bool:
    err_lower = (err_text or "").lower()
    return any(kw in err_lower for kw in ("memoryerror", "arraymemoryerror", "unable to allocate", "out of memory"))


def _find_ts_executable() -> Optional[str]:
    ts = _find_totalseg_exe(sys.executable)
    if ts:
        return ts
    dl_py = _find_dl_env_python()
    if dl_py and dl_py != sys.executable:
        ts = _find_totalseg_exe(dl_py)
        if ts:
            return ts
    return None


TASKS_NO_FAST = {"tissue_4_types", "tissue_types", "tissue_types_mr"}


def _low_mem_env() -> dict:
    env = os.environ.copy()
    env["nnUNet_def_n_proc"] = "1"
    env["nnUNet_n_proc_DA"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    return env


def _run_totalsegmentator_cli(
    input_nii: str, output_dir: str, task: str, device: str, timeout: int,
    fast: bool = False, force_split: bool = False, nr_thr_resamp: int = 1, nr_thr_saving: int = 1,
) -> Tuple[bool, str]:
    import shutil
    output_dir = os.path.abspath(output_dir)
    input_nii = os.path.abspath(input_nii)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device_cli = _device_to_cli(device)

    def _run_cmd(cmd: list, env: dict = None) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=os.path.dirname(input_nii) or ".", env=env or os.environ.copy(),
            )
            out, err = (result.stdout or "").strip(), (result.stderr or "").strip()
            if result.returncode == 0 and any(Path(output_dir).glob("*.nii*")):
                return True, ""
            return False, _filter_error(err) or _filter_error(out) or f"Exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s. Try GPU or increase timeout in sidebar."
        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _build_args(use_fast: bool = False, use_fastest: bool = False, use_split: bool = False, use_body_seg: bool = False) -> list:
        args = ["-i", input_nii, "-o", output_dir, "-ta", task, "-d", device_cli, "-nr", str(nr_thr_resamp), "-ns", str(nr_thr_saving)]
        if use_fastest and task not in TASKS_NO_FAST:
            args.append("--fastest")
        elif use_fast and task not in TASKS_NO_FAST:
            args.append("--fast")
        if use_split:
            args.append("--force_split")
        if use_body_seg:
            args.append("--body_seg")
        return args

    def _clean_output():
        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    ts_exe = _find_ts_executable()
    if not ts_exe:
        return False, "TotalSegmentator executable not found. Install: pip install TotalSegmentator"

    low_env = _low_mem_env()

    # Attempt 1: user settings + low-memory env
    cli_args = _build_args(use_fast=fast, use_split=force_split)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, ""
    if not _is_memory_error(err):
        return False, "TotalSegmentator failed. " + (err or "")[:2000]

    # Attempt 2: + force_split + body_seg
    _clean_output()
    cli_args = _build_args(use_fast=fast, use_split=True, use_body_seg=True)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, "(low-RAM: --force_split --body_seg) "
    if not _is_memory_error(err):
        return False, "TotalSegmentator failed. " + (err or "")[:2000]

    # Attempt 3: + fast (3mm) + force_split + body_seg
    _clean_output()
    cli_args = _build_args(use_fast=True, use_split=True, use_body_seg=True)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, "(low-RAM: --fast --force_split --body_seg) "
    if not _is_memory_error(err):
        return False, "TotalSegmentator failed. " + (err or "")[:2000]

    # Attempt 4: fastest (6mm) + force_split + body_seg
    _clean_output()
    cli_args = _build_args(use_fastest=True, use_split=True, use_body_seg=True)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, "(low-RAM: --fastest --force_split --body_seg) "

    return False, "TotalSegmentator failed (MemoryError after all retries). " + (err or "")[:2000]


def run_totalsegmentator(
    input_nii: str, output_dir: str, task: str = TASK_TISSUE_4, device: str = "cuda", timeout: int = 1800,
    fast: bool = False, force_split: bool = False, nr_thr_resamp: int = 1, nr_thr_saving: int = 1,
) -> Tuple[bool, str]:
    return _run_totalsegmentator_cli(
        input_nii, output_dir, task, device, timeout,
        fast=fast, force_split=force_split, nr_thr_resamp=nr_thr_resamp, nr_thr_saving=nr_thr_saving,
    )


def run_segmentation_pipeline(
    ct_nii_path: str, work_dir: str, device: str = "cuda",
    timeout_total: int = 5400, timeout_tissue: int = 2400,
    fast: bool = False, force_split: bool = False, nr_thr_resamp: int = 1, nr_thr_saving: int = 1,
) -> Tuple[Optional[str], Optional[str], str]:
    work_dir = os.path.abspath(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_total = os.path.join(work_dir, "seg_total")
    out_tissue = os.path.join(work_dir, "seg_4tissue")

    if not os.path.isfile(ct_nii_path):
        return None, None, f"CT file not found: {ct_nii_path}"

    seg_kwargs = dict(fast=fast, force_split=force_split, nr_thr_resamp=nr_thr_resamp, nr_thr_saving=nr_thr_saving)
    notes = []

    ok_total, msg_total = run_totalsegmentator(ct_nii_path, out_total, task=TASK_TOTAL, device=device, timeout=timeout_total, **seg_kwargs)
    if not ok_total:
        return None, None, "TotalSegmentator (total) failed or timed out. " + (msg_total or "")
    if msg_total:
        notes.append("total: " + msg_total)

    ok_tissue, msg_tissue = run_totalsegmentator(ct_nii_path, out_tissue, task=TASK_TISSUE_4, device=device, timeout=timeout_tissue, **seg_kwargs)
    if not ok_tissue:
        return out_total, None, "TotalSegmentator (tissue_4_types) failed or timed out. " + (msg_tissue or "")
    if msg_tissue:
        notes.append("tissue: " + msg_tissue)

    result_msg = "Segmentation completed successfully."
    if notes:
        result_msg += " " + " | ".join(notes)
    return out_total, out_tissue, result_msg
