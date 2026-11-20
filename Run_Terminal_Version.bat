@echo off
title Multi-Core RTOS Scheduling Simulator - Terminal Version
REM Double-click to start the terminal (command-line) version of the simulator.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 from the Microsoft Store first.
    pause
    exit /b 1
)
python -c "import pandas, numpy, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages - first run only, please wait...
    python -m pip install -r requirements.txt
)
python cli.py
pause
