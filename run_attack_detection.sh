#!/bin/bash
# Federated Learning Attack Detection - Build and Run Script
# =========================================================

set -e  # Exit on any error

echo "Federated Learning Attack Detection System"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p dataset results

# Check if dataset exists
if [ ! -f "dataset/smallLendingClub.csv" ]; then
    echo "Warning: Dataset not found at dataset/smallLendingClub.csv"
    echo "Please place your Lending Club dataset in the dataset/ directory"
    echo "You can download it from: https://www.kaggle.com/datasets/wordsforthewise/lending-club"
    echo ""
    echo "For testing purposes, the system will create a dummy dataset..."
fi

# Build Docker image
echo "Building Docker image..."
docker build -t fl-attack-detect:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Run the attack detection system
echo ""
echo "Running attack detection experiment..."
echo "This may take several minutes depending on your system..."

docker run --rm \
    -v "$(pwd)/dataset:/app/dataset:ro" \
    -v "$(pwd)/results:/app/results" \
    fl-attack-detect:latest \
    --dataset /app/dataset/smallLendingClub.csv \
    --epochs 100 \
    --n_train_clients 10 \
    --n_total_clients 10 \
    --malicious-percentages 0 10 20 \
    --output-dir /app/results \
    --epsilon 1.0

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Attack detection experiment completed successfully!"
    echo ""
    echo "Results saved to: $(pwd)/results/"
    echo "Files created:"
    ls -la results/
    echo ""
    echo "To view results:"
    echo "  - Check results/lending_club_results.pkl for complete data"
    echo "  - View plots: results/*.png"
    echo "  - Check metadata: results/metadata.json"
else
    echo "❌ Attack detection experiment failed"
    exit 1
fi

echo ""
echo "To run smoke tests:"
echo "  python tests/test_attack_detection.py"
echo ""
echo "To run with different parameters:"
echo "  docker run --rm -v \$(pwd)/dataset:/app/dataset:ro -v \$(pwd)/results:/app/results fl-attack-detect:latest --help"



