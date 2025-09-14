# Federated Learning Attack Detection System

This repository implements a comprehensive malicious data poisoning client detection mechanism for federated learning using Local Differential Privacy (LDP) and K-means clustering.

## Overview

The system detects malicious clients in federated learning by:
1. **Local Differential Privacy (LDP)**: Adds calibrated noise to client losses to preserve privacy while maintaining detection utility
2. **K-means Clustering**: Identifies malicious clients by clustering noisy losses and selecting the cluster with highest mean loss
3. **Real-time Detection**: Removes detected malicious clients from aggregation during training

## Quick Start

### Prerequisites

- Docker (version 20.10 or higher)
- At least 4GB RAM
- 2GB free disk space

### Building the Docker Image

```bash
# Build the attack detection image
docker build -t fl-attack-detect:latest .
```

### Running the Attack Detection System

#### Basic Usage

```bash
# Run with default parameters
docker run -v /local/dataset:/app/dataset -v /local/results:/app/results fl-attack-detect:latest

# Run with custom parameters
docker run -v /local/dataset:/app/dataset -v /local/results:/app/results fl-attack-detect:latest \
  --dataset /app/dataset/smallLendingClub.csv \
  --epochs 100 \
  --n_train_clients 10 \
  --n_total_clients 10 \
  --malicious-percentages 0 10 20 \
  --output-dir /app/results \
  --epsilon 1.0
```

#### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `/app/dataset/smallLendingClub.csv` | Path to the dataset CSV file |
| `--epochs` | `100` | Number of training epochs |
| `--n_train_clients` | `10` | Number of clients participating in training |
| `--n_total_clients` | `10` | Total number of clients |
| `--malicious-percentages` | `[0.0, 10.0]` | List of malicious client percentages to test |
| `--output-dir` | `/app/results` | Output directory for results |
| `--epsilon` | `1.0` | LDP privacy parameter |
| `--verbose` | `False` | Enable verbose logging |

### Expected Output

After running, you should find the following files in your results directory:

```
results/
├── lending_club_results.pkl      # Complete experiment results
├── metadata.json                 # Experiment metadata and parameters
├── test_accuracy.png            # Test accuracy over epochs plot
├── attack_detection.png         # Detection accuracy over epochs plot
└── detection_f1.png             # Detection F1 score over epochs plot
```

## How It Works

### 1. Local Differential Privacy (LDP)

The system applies LDP noise to client losses using the Laplace mechanism:

```python
# Add calibrated noise to preserve privacy
noise_scale = sensitivity / epsilon
noise = np.random.laplace(0, noise_scale, size=losses.shape)
noisy_losses = losses + noise
```

**Key Parameters:**
- `epsilon`: Privacy parameter (smaller = more private, larger = more utility)
- `sensitivity`: Sensitivity parameter (default: 1e-4 for normalized losses)

### 2. K-means Clustering Detection

Malicious clients are identified by clustering noisy losses:

```python
# Cluster clients based on noisy losses
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(noisy_losses)

# Select cluster with highest mean loss as malicious
malicious_cluster_id = np.argmax(cluster_means)
```

**Detection Logic:**
- Assumes malicious clients have higher losses due to data poisoning
- Uses 2-cluster K-means by default (benign vs malicious)
- Selects cluster with highest mean loss as malicious

### 3. Attack Types Supported

- **Label Flipping**: Malicious clients flip training labels (0→1, 1→0)
- **Data Poisoning**: Malicious clients inject corrupted data
- **Model Poisoning**: Malicious clients send corrupted model updates

## Testing

### Smoke Tests

Run the included smoke tests to verify the system works correctly:

```bash
# Run smoke tests
python tests/test_attack_detection.py

# Or with pytest
pytest tests/test_attack_detection.py -v
```

### Test Cases

The smoke tests verify:

1. **Zero Malicious Clients**: System completes without errors when no attacks are present
2. **Ten Percent Malicious**: System detects and handles 10% malicious clients
3. **API Functionality**: All detection functions work correctly
4. **Result Generation**: Output files are created properly

### Expected Test Output

```
Running Federated Learning Attack Detection Smoke Tests
============================================================
test_apply_ldp (__main__.TestAttackDetection) ... ok
test_eliminate_kmeans (__main__.TestAttackDetection) ... ok
test_calculate_accuracy (__main__.TestAttackDetection) ... ok
test_calculate_f1_score (__main__.TestAttackDetection) ... ok
test_detect_malicious_clients (__main__.TestAttackDetection) ... ok
test_zero_malicious_percentage (__main__.TestSmokeTests) ... ok
test_ten_malicious_percentage (__main__.TestSmokeTests) ... ok
test_script_help (__main__.TestSmokeTests) ... ok

======================================================================
SMOKE TEST SUMMARY
======================================================================
Tests run: 8
Failures: 0
Errors: 0

Overall result: PASSED
```

## Performance Considerations

### Privacy vs Utility Trade-off

- **Lower epsilon (0.1-0.5)**: Stronger privacy, may reduce detection accuracy
- **Higher epsilon (1.0-2.0)**: Better detection accuracy, weaker privacy
- **Default epsilon (1.0)**: Balanced privacy-utility trade-off

### Computational Requirements

- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ RAM for 10 clients, 100 epochs
- **Storage**: 1GB+ for results and plots
- **Time**: ~5-10 minutes for 100 epochs with 10 clients

### Scaling Considerations

- **More Clients**: Detection accuracy may decrease with very large client pools
- **More Epochs**: Longer training improves model performance but increases runtime
- **Larger Datasets**: Memory usage scales with dataset size

## Troubleshooting

### Common Issues

1. **"Dataset file not found"**
   - Ensure dataset path is correct and file exists
   - Check Docker volume mounting: `-v /local/dataset:/app/dataset`

2. **"Port already in use"**
   - Stop other Docker containers using the same port
   - Use `docker ps` to check running containers

3. **"Out of memory"**
   - Reduce number of clients or epochs
   - Increase Docker memory limit
   - Use smaller dataset

4. **"Detection accuracy is low"**
   - Try different epsilon values (0.5-2.0)
   - Check if attacks are too subtle
   - Verify data preprocessing is correct

### Debug Mode

Enable verbose logging for debugging:

```bash
docker run -v /local/dataset:/app/dataset -v /local/results:/app/results fl-attack-detect:latest --verbose
```

## File Structure

```
fl_defenses/
├── detector.py              # Core detection algorithms (LDP, K-means)
scripts/
├── run_attack_detection.py  # Main experiment script
tests/
├── test_attack_detection.py # Unit tests and smoke tests
Dockerfile                   # Docker configuration
requirements.txt            # Python dependencies
README_ATTACK_DETECTION.md  # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{fl_attack_detection,
  title={Federated Learning Attack Detection System},
  author={FL Defense System},
  year={2025},
  url={https://github.com/your-repo/fl-attack-detection}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the test cases for usage examples



