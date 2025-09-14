@echo off
REM Federated Learning Attack Detection - Docker Research Demo
REM Final Year Project Presentation System

echo ================================================================
echo 🎓 FEDERATED LEARNING ATTACK DETECTION - DOCKER DEMO
echo ================================================================
echo Final Year Project Presentation System
echo Data Poisoning Prevention in Federated Learning
echo ================================================================
echo.

REM Check if Docker is running
echo 🔍 Checking Docker environment...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)
echo ✅ Docker is running

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose.
    pause
    exit /b 1
)
echo ✅ Docker Compose is available

REM Clean up any existing containers
echo.
echo 🧹 Cleaning up existing containers...
docker-compose down -v >nul 2>&1
echo ✅ Cleanup completed

REM Build the images
echo.
echo 🔨 Building Docker images...
echo ℹ️  This may take a few minutes on first run...

docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo ❌ Failed to build images
    pause
    exit /b 1
)
echo ✅ Images built successfully

REM Start the services
echo.
echo 🚀 Starting Federated Learning system...
echo ℹ️  Starting all services (CA, Server, Clients, Dashboard, Research Demo)...

docker-compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start services
    pause
    exit /b 1
)
echo ✅ All services started successfully

REM Wait for services to be ready
echo.
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo 🔍 Checking service status...
docker-compose ps

REM Run the research demo
echo.
echo 🎓 Running research demonstration...
echo ℹ️  Executing comprehensive analysis...

docker-compose exec -T research-demo python demo_presentation.py
if %errorlevel% neq 0 (
    echo ⚠️  Research demo had issues, but system is still running
) else (
    echo ✅ Research demo completed successfully
)

REM Display access information
echo.
echo ================================================================
echo 🌐 ACCESS INFORMATION
echo ================================================================
echo.
echo 📊 Main Dashboard:
echo    URL: http://localhost:5000
echo    Features: System monitoring, client management, training metrics
echo.
echo 🎓 Research Demo:
echo    URL: http://localhost:5000/research
echo    Features: Interactive attack detection, privacy analysis, visualization
echo.
echo 📈 Monitoring:
echo    Prometheus: http://localhost:9090
echo    Grafana: http://localhost:3000 (if enabled)
echo.
echo 🔧 Management Commands:
echo    View logs: docker-compose logs -f [service_name]
echo    Stop system: docker-compose down
echo    Restart: docker-compose restart [service_name]
echo    Access container: docker-compose exec [service_name] bash
echo.

REM Display research demo results
echo ================================================================
echo 📊 RESEARCH DEMO RESULTS
echo ================================================================
echo.

REM Check if demo results exist
docker-compose exec -T research-demo ls -la demo_results/ 2>nul | findstr /i "\.png\|\.json\|\.md" >nul
if %errorlevel% equ 0 (
    echo ✅ Demo results generated successfully
    echo.
    echo Generated files:
    docker-compose exec -T research-demo ls -la demo_results/ | findstr /i "\.png\|\.json\|\.md"
) else (
    echo ⚠️  Demo results not found. Check research-demo container logs.
)

echo.
echo ================================================================
echo 🎉 SYSTEM READY FOR PRESENTATION!
echo ================================================================
echo.
echo 🎯 Next Steps:
echo 1. Open http://localhost:5000/research in your browser
echo 2. Follow the presentation guide in presentation_guide.md
echo 3. Demonstrate different attack scenarios
echo 4. Show privacy-detection trade-off analysis
echo 5. Export results for documentation
echo.
echo 📖 For detailed instructions, check:
echo    - presentation_guide.md
echo    - RESEARCH_PROJECT_SUMMARY.md
echo.
echo 🛑 To stop the system: docker-compose down
echo ================================================================

echo.
echo 📋 Showing dashboard logs (Press Ctrl+C to stop):
echo ================================================================
docker-compose logs -f dashboard

