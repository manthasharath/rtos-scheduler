@echo off
title Multi-Core RTOS Scheduling Simulator
REM Double-click this file to start the simulator in your web browser.
REM Keep this window open while using the app. Close it to stop the app.

cd /d "%~dp0"

REM Skip Streamlit's first-run email question
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 from the Microsoft Store first.
    pause
    exit /b 1
)

python -c "import streamlit, plotly, pandas, openpyxl, sys; v=tuple(int(x) for x in streamlit.__version__.split('.')[:2]); sys.exit(0 if v>=(1,50) else 1)" >nul 2>nul
if errorlevel 1 (
    echo Installing / updating required packages - please wait...
    python -m pip install --upgrade -r requirements.txt
)

echo.
echo Starting the simulator... your browser will open at http://localhost:8501
echo Keep this window open. Close it to stop the app.
echo.
python -m streamlit run app.py --browser.gatherUsageStats false
pause
