@echo off
echo ============================================================
echo FL Dashboard Docker Startup
echo ============================================================
echo.

echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed or not running
    echo Please install Docker Desktop and start it
    pause
    exit /b 1
)

echo Docker found. Starting dashboard...
echo.

echo Step 1: Building dashboard image...
docker-compose build
if errorlevel 1 (
    echo Error: Failed to build dashboard image
    pause
    exit /b 1
)

echo.
echo Step 2: Starting dashboard container...
docker-compose up -d
if errorlevel 1 (
    echo Error: Failed to start dashboard container
    pause
    exit /b 1
)

echo.
echo Step 3: Waiting for dashboard to start...
timeout /t 10 /nobreak >nul

echo.
echo Step 4: Testing dashboard...
python test_dashboard.py

echo.
echo Dashboard should be running at: http://localhost:5000
echo.
echo To stop the dashboard, run: docker-compose down
echo To view logs, run: docker-compose logs -f dashboard
echo.
pause

