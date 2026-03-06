# -*- coding: utf-8 -*-
"""
Launcher for the Streamlit app. Prefer running in an env where TotalSegmentator is installed (e.g. dl_env).
Usage:  python run_streamlit.py   (uses dl_env Python if found)
        Or:  streamlit run app.py
        Or double-click run.bat (Windows).
"""
import os
import sys
import subprocess

ENV_NAME = "dl_env"


def find_dl_env_python():
    """Return path to Python in conda env dl_env or venv dl_env, or None."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if os.environ.get("CONDA_DEFAULT_ENV") == ENV_NAME:
        return sys.executable
    exe = os.path.normpath(sys.executable)
    parent_dir = os.path.basename(os.path.dirname(exe))
    if parent_dir.lower() == ENV_NAME:
        return sys.executable

    if sys.platform == "win32":
        base = os.sep.join(exe.split(os.sep)[:exe.split(os.sep).index("envs")]) if "envs" in exe.split(os.sep) else os.path.dirname(exe)
        env_py = os.path.join(base, "envs", ENV_NAME, "python.exe")
    else:
        base = os.sep.join(exe.split(os.sep)[:exe.split(os.sep).index("envs")]) if "envs" in exe.split(os.sep) else os.path.dirname(exe)
        env_py = os.path.join(base, "envs", ENV_NAME, "bin", "python")
    if os.path.isfile(env_py):
        return env_py

    for rel in [ENV_NAME, "..", os.path.join("..", ENV_NAME), os.path.join("..", "..", ENV_NAME)]:
        base = os.path.normpath(os.path.join(app_dir, rel))
        py = os.path.join(base, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(base, "bin", "python")
        if os.path.isfile(py):
            return py
    return None


def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    if not os.path.isfile(os.path.join(app_dir, "app.py")):
        print("Error: app.py not found in", app_dir)
        sys.exit(1)

    python_exe = find_dl_env_python()
    if python_exe is None:
        print(f"ERROR: {ENV_NAME} not found. Install TotalSegmentator in that env, or run: streamlit run app.py")
        print("Current Python:", sys.executable)
        sys.exit(1)

    print("Using Python:", python_exe)
    for port in [8501, 8502, 8503]:
        print(f"Starting Streamlit on http://localhost:{port} ...")
        ret = subprocess.call([python_exe, "-m", "streamlit", "run", "app.py", "--server.port", str(port)])
        if ret == 0:
            sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
