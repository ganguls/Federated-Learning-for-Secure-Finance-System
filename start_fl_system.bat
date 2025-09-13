@echo off
echo ========================================
echo Federated Learning System - Local Mode
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements_local.txt

REM Start the FL system
echo Starting FL system...
python run_fl_system_local.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo System stopped with errors. Check the logs.
    pause
)

