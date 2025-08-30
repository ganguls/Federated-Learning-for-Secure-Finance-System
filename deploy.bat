@echo off
REM FL Enterprise System Deployment Script for Windows
REM This script deploys the entire FL system with Docker and Kubernetes

echo 🚀 Starting FL Enterprise System Deployment...

REM Check if Docker is running
echo [INFO] Checking Docker status...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)
echo [SUCCESS] Docker is running

REM Check if kubectl is available
echo [INFO] Checking kubectl availability...
kubectl version --client >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] kubectl not found. Kubernetes deployment will be skipped.
    set K8S_AVAILABLE=false
) else (
    echo [SUCCESS] kubectl found
    set K8S_AVAILABLE=true
)

REM Build Docker images
echo [INFO] Building Docker images...

echo [INFO] Building Central Authority image...
docker build -t fl-enterprise-ca:latest ./ca

echo [INFO] Building Server image...
docker build -t fl-enterprise-server:latest ./server

echo [INFO] Building Client image...
docker build -t fl-enterprise-client:latest ./clients

echo [INFO] Building Dashboard image...
docker build -t fl-enterprise-dashboard:latest ./dashboard

echo [SUCCESS] All Docker images built successfully

REM Deploy with Docker Compose
echo [INFO] Deploying with Docker Compose...

echo [INFO] Stopping existing containers...
docker-compose down --remove-orphans

echo [INFO] Starting services...
docker-compose up -d

echo [INFO] Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo [INFO] Checking service status...
docker-compose ps

echo [SUCCESS] Docker deployment completed

REM Deploy to Kubernetes if available
if "%K8S_AVAILABLE%"=="true" (
    echo [INFO] Deploying to Kubernetes...
    
    echo [INFO] Creating namespace...
    kubectl apply -f k8s/namespace.yaml
    
    echo [INFO] Setting up storage...
    kubectl apply -f k8s/persistent-volumes.yaml
    
    echo [INFO] Creating configuration...
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    
    echo [INFO] Deploying services...
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/services.yaml
    
    echo [INFO] Deploying ingress...
    kubectl apply -f k8s/ingress.yaml
    
    echo [SUCCESS] Kubernetes deployment completed
) else (
    echo [WARNING] Skipping Kubernetes deployment
)

REM Initialize CA
echo [INFO] Initializing Central Authority...
docker exec fl-ca python ca.py --action init 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] CA initialized successfully
) else (
    echo [WARNING] CA initialization failed or container not running
)

REM Show deployment information
echo.
echo 🎉 FL Enterprise System Deployment Complete!
echo.
echo 📊 Services Status:
echo   - Dashboard: http://localhost:5000
echo   - Server: http://localhost:8080
echo   - CA Service: http://localhost:9000
echo.

if "%K8S_AVAILABLE%"=="true" (
    echo ☸️  Kubernetes Services:
    echo   - Dashboard: http://dashboard.fl-enterprise.local
    echo   - CA: http://ca.fl-enterprise.local
    echo.
    echo 📋 To check Kubernetes status:
    echo   kubectl get all -n fl-enterprise
    echo   kubectl get pods -n fl-enterprise
    echo.
)

echo 🔧 To manage the system:
echo   - Start: docker-compose up -d
echo   - Stop: docker-compose down
echo   - Logs: docker-compose logs -f
echo.

echo [SUCCESS] Deployment completed successfully!
pause
