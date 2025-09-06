@echo off
echo Starting Federated Learning System with 10 Clients
echo ================================================

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not running or not installed!
    pause
    exit /b 1
)

echo Starting all services (CA, Server, Dashboard, and 10 Clients)...
docker-compose up -d

echo.
echo All services started! You can now access:
echo - Dashboard: http://localhost:5000
echo - Server API: http://localhost:8080
echo - CA Service: http://localhost:9000
echo - Prometheus: http://localhost:9090

echo.
echo To view logs: docker-compose logs -f
echo To stop all services: docker-compose down
echo.
pause
