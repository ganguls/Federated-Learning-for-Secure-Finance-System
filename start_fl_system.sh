#!/bin/bash

echo "========================================"
echo "Federated Learning System - Local Mode"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.9+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements_local.txt

# Start the FL system
echo "Starting FL system..."
python3 run_fl_system_local.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo "System stopped with errors. Check the logs."
    read -p "Press Enter to continue..."
fi

