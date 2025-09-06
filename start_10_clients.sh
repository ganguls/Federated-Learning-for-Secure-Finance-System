#!/bin/bash

echo "Starting Federated Learning System with 10 Clients"
echo "================================================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed or not in PATH!"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Docker is not running! Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed or not in PATH!"
    exit 1
fi

echo "Starting all services (CA, Server, Dashboard, and 10 Clients)..."
docker-compose up -d

echo
echo "All services started! You can now access:"
echo "- Dashboard: http://localhost:5000"
echo "- Server API: http://localhost:8080"
echo "- CA Service: http://localhost:9000"
echo "- Prometheus: http://localhost:9090"

echo
echo "To view logs: docker-compose logs -f"
echo "To stop all services: docker-compose down"
echo

# Show running containers
echo "Running containers:"
docker-compose ps
