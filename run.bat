@echo off
chcp 65001 >nul 2>nul
cd /d "%~dp0"

echo ========================================
echo  PET/CT Streamlit Launcher
echo ========================================
echo.

set "PORT=8501"

REM Prefer conda run -n dl_env (if conda in PATH)
where conda >nul 2>nul
if %errorlevel% equ 0 (
    echo Using: conda run -n dl_env ...
    echo Open: http://localhost:%PORT%
    echo.
    conda run -n dl_env streamlit run app.py --server.port %PORT%
    if %errorlevel% equ 0 goto :done
)

REM Fallback: current Python
echo Using current Python: streamlit run app.py
echo Open: http://localhost:%PORT%
echo.
python -m streamlit run app.py --server.port %PORT%
if %errorlevel% equ 0 goto :done

echo.
echo Failed to start. Install: pip install streamlit
echo Or create conda env: conda create -n dl_env python=3.10
echo   conda activate dl_env
echo   pip install -r requirements.txt
pause
exit /b 1

:done
echo.
pause
