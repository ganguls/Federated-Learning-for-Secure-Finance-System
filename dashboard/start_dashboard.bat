@echo off
echo ============================================================
echo FL Enterprise Dashboard Startup
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

echo Python found. Starting dashboard...
echo.

echo Step 1: Installing dashboard dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 2: Starting the dashboard...
python start_dashboard.py

echo.
echo Dashboard stopped. Press any key to exit...
pause >nul
