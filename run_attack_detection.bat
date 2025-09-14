@echo off
REM Federated Learning Attack Detection - Build and Run Script (Windows)
REM ====================================================================

echo Federated Learning Attack Detection System
echo ==========================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed or not in PATH
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running
    exit /b 1
)

REM Create necessary directories
echo Creating directories...
if not exist "dataset" mkdir dataset
if not exist "results" mkdir results

REM Check if dataset exists
if not exist "dataset\smallLendingClub.csv" (
    echo Warning: Dataset not found at dataset\smallLendingClub.csv
    echo Please place your Lending Club dataset in the dataset\ directory
    echo You can download it from: https://www.kaggle.com/datasets/wordsforthewise/lending-club
    echo.
    echo For testing purposes, the system will create a dummy dataset...
)

REM Build Docker image
echo Building Docker image...
docker build -t fl-attack-detect:latest .

if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully
) else (
    echo ❌ Docker build failed
    exit /b 1
)

REM Run the attack detection system
echo.
echo Running attack detection experiment...
echo This may take several minutes depending on your system...

docker run --rm -v "%cd%\dataset:/app/dataset:ro" -v "%cd%\results:/app/results" fl-attack-detect:latest --dataset /app/dataset/smallLendingClub.csv --epochs 100 --n_train_clients 10 --n_total_clients 10 --malicious-percentages 0 10 20 --output-dir /app/results --epsilon 1.0

if %errorlevel% equ 0 (
    echo.
    echo ✅ Attack detection experiment completed successfully!
    echo.
    echo Results saved to: %cd%\results\
    echo Files created:
    dir results
    echo.
    echo To view results:
    echo   - Check results\lending_club_results.pkl for complete data
    echo   - View plots: results\*.png
    echo   - Check metadata: results\metadata.json
) else (
    echo ❌ Attack detection experiment failed
    exit /b 1
)

echo.
echo To run smoke tests:
echo   python tests\test_attack_detection.py
echo.
echo To run with different parameters:
echo   docker run --rm -v "%cd%\dataset:/app/dataset:ro" -v "%cd%\results:/app/results" fl-attack-detect:latest --help

pause



