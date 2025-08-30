@echo off
echo ============================================================
echo Federated Learning System for Loan Prediction
echo ============================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Starting the system...
echo.

echo Step 1: Running data preprocessing...
cd Datapre
python complete_datapre.py
if errorlevel 1 (
    echo Error: Data preprocessing failed
    pause
    exit /b 1
)
cd ..

echo.
echo Step 2: Starting federated learning system...
python run_fl_system.py

echo.
echo System completed. Press any key to exit...
pause >nul
