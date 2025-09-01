#!/bin/bash

echo "============================================================"
echo "FL Dashboard Docker Startup"
echo "============================================================"
echo

echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Please install Docker and start the daemon"
    exit 1
fi

echo "Docker found. Starting dashboard..."
echo

echo "Step 1: Building dashboard image..."
docker-compose build
if [ $? -ne 0 ]; then
    echo "Error: Failed to build dashboard image"
    exit 1
fi

echo
echo "Step 2: Starting dashboard container..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "Error: Failed to start dashboard container"
    exit 1
fi

echo
echo "Step 3: Waiting for dashboard to start..."
sleep 10

echo
echo "Step 4: Testing dashboard..."
python3 test_dashboard.py

echo
echo "Dashboard should be running at: http://localhost:5000"
echo
echo "To stop the dashboard, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f dashboard"
echo

