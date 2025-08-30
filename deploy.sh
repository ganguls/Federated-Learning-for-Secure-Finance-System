#!/bin/bash

# FL Enterprise System Deployment Script
# This script deploys the entire FL system with Docker and Kubernetes

set -e

echo "🚀 Starting FL Enterprise System Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker status..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if kubectl is available
check_kubectl() {
    print_status "Checking kubectl availability..."
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl not found. Kubernetes deployment will be skipped."
        K8S_AVAILABLE=false
    else
        print_success "kubectl found"
        K8S_AVAILABLE=true
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build CA image
    print_status "Building Central Authority image..."
    docker build -t fl-enterprise-ca:latest ./ca
    
    # Build Server image
    print_status "Building Server image..."
    docker build -t fl-enterprise-server:latest ./server
    
    # Build Client image
    print_status "Building Client image..."
    docker build -t fl-enterprise-client:latest ./clients
    
    # Build Dashboard image
    print_status "Building Dashboard image..."
    docker build -t fl-enterprise-dashboard:latest ./dashboard
    
    print_success "All Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    print_status "Deploying with Docker Compose..."
    
    # Stop existing containers
    print_status "Stopping existing containers..."
    docker-compose down --remove-orphans
    
    # Start services
    print_status "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service status
    print_status "Checking service status..."
    docker-compose ps
    
    print_success "Docker deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    if [ "$K8S_AVAILABLE" = false ]; then
        print_warning "Skipping Kubernetes deployment"
        return
    fi
    
    print_status "Deploying to Kubernetes..."
    
    # Create namespace
    print_status "Creating namespace..."
    kubectl apply -f k8s/namespace.yaml
    
    # Create storage class and persistent volumes
    print_status "Setting up storage..."
    kubectl apply -f k8s/persistent-volumes.yaml
    
    # Create configmap and secrets
    print_status "Creating configuration..."
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    
    # Deploy services
    print_status "Deploying services..."
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/services.yaml
    
    # Deploy ingress
    print_status "Deploying ingress..."
    kubectl apply -f k8s/ingress.yaml
    
    # Wait for deployments to be ready
    print_status "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/fl-ca -n fl-enterprise
    kubectl wait --for=condition=available --timeout=300s deployment/fl-server -n fl-enterprise
    kubectl wait --for=condition=available --timeout=300s deployment/fl-dashboard -n fl-enterprise
    
    print_success "Kubernetes deployment completed"
}

# Initialize CA
initialize_ca() {
    print_status "Initializing Central Authority..."
    
    # Check if CA is running
    if docker ps | grep -q "fl-ca"; then
        print_status "CA is running, initializing..."
        docker exec fl-ca python ca.py --action init
        print_success "CA initialized successfully"
    else
        print_warning "CA container not running. Please start the system first."
    fi
}

# Show deployment information
show_info() {
    echo ""
    echo "🎉 FL Enterprise System Deployment Complete!"
    echo ""
    echo "📊 Services Status:"
    echo "  - Dashboard: http://localhost:5000"
    echo "  - Server: http://localhost:8080"
    echo "  - CA Service: http://localhost:9000"
    echo ""
    
    if [ "$K8S_AVAILABLE" = true ]; then
        echo "☸️  Kubernetes Services:"
        echo "  - Dashboard: http://dashboard.fl-enterprise.local"
        echo "  - CA: http://ca.fl-enterprise.local"
        echo ""
        echo "📋 To check Kubernetes status:"
        echo "  kubectl get all -n fl-enterprise"
        echo "  kubectl get pods -n fl-enterprise"
    fi
    
    echo ""
    echo "🔧 To manage the system:"
    echo "  - Start: docker-compose up -d"
    echo "  - Stop: docker-compose down"
    echo "  - Logs: docker-compose logs -f"
    echo ""
}

# Main deployment flow
main() {
    print_status "Starting FL Enterprise System deployment..."
    
    # Pre-deployment checks
    check_docker
    check_kubectl
    
    # Build images
    build_images
    
    # Deploy with Docker Compose
    deploy_docker
    
    # Deploy to Kubernetes if available
    deploy_kubernetes
    
    # Initialize CA
    initialize_ca
    
    # Show deployment information
    show_info
    
    print_success "Deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    --docker-only)
        print_status "Docker-only deployment mode"
        check_docker
        build_images
        deploy_docker
        initialize_ca
        show_info
        ;;
    --k8s-only)
        print_status "Kubernetes-only deployment mode"
        check_kubectl
        if [ "$K8S_AVAILABLE" = true ]; then
            build_images
            deploy_kubernetes
            show_info
        else
            print_error "Kubernetes not available"
            exit 1
        fi
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo "Options:"
        echo "  --docker-only    Deploy only with Docker Compose"
        echo "  --k8s-only       Deploy only to Kubernetes"
        echo "  --help, -h       Show this help message"
        echo "  (no option)      Deploy with both Docker and Kubernetes"
        exit 0
        ;;
    *)
        main
        ;;
esac
