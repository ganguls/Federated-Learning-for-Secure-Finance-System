#!/usr/bin/env python3
"""
Example Usage of Federated Learning Attack Detection System
==========================================================

This script demonstrates how to use the attack detection system
with a simple example.

Usage:
    python example_usage.py

Author: FL Defense System
Date: 2025
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fl_defenses.detector import detect_malicious_clients, apply_ldp, eliminate_kmeans


def create_sample_data():
    """Create sample data for demonstration."""
    # Create a simple dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    X = np.random.randn(n_samples, 10)
    
    # Generate labels with some pattern
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create DataFrame
    data = {}
    for i in range(10):
        data[f'feature_{i}'] = X[:, i]
    data['loan_status_binary'] = y
    
    df = pd.DataFrame(data)
    
    return df


def demonstrate_ldp():
    """Demonstrate LDP noise application."""
    print("=" * 60)
    print("DEMONSTRATING LOCAL DIFFERENTIAL PRIVACY (LDP)")
    print("=" * 60)
    
    # Sample losses from different clients
    original_losses = [0.1, 0.12, 0.11, 0.8, 0.9, 0.85]  # First 3 benign, last 3 malicious
    
    print(f"Original losses: {original_losses}")
    print(f"Loss range: [{min(original_losses):.3f}, {max(original_losses):.3f}]")
    
    # Apply LDP with different epsilon values
    for epsilon in [0.5, 1.0, 2.0]:
        noisy_losses = apply_ldp(original_losses, epsilon=epsilon, sensitivity=1e-4)
        print(f"\nEpsilon = {epsilon}:")
        print(f"  Noisy losses: {[f'{x:.3f}' for x in noisy_losses]}")
        print(f"  Noise range: [{min(noisy_losses):.3f}, {max(noisy_losses):.3f}]")


def demonstrate_detection():
    """Demonstrate attack detection."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING ATTACK DETECTION")
    print("=" * 60)
    
    # Simulate losses from 10 clients (first 7 benign, last 3 malicious)
    true_malicious = [7, 8, 9]
    original_losses = [0.1, 0.12, 0.11, 0.13, 0.14, 0.15, 0.16, 0.8, 0.9, 0.85]
    
    print(f"True malicious clients: {true_malicious}")
    print(f"Original losses: {[f'{x:.3f}' for x in original_losses]}")
    
    # Run detection
    results = detect_malicious_clients(
        original_losses,
        true_malicious=true_malicious,
        epsilon=1.0,
        sensitivity=1e-4
    )
    
    print(f"\nDetection Results:")
    print(f"  Detected malicious: {results['detected_malicious']}")
    print(f"  Detection accuracy: {results['accuracy']:.3f}")
    print(f"  Detection F1 score: {results['f1_score']:.3f}")
    
    # Show detection metrics
    metrics = results['detection_metrics']
    print(f"\nDetection Metrics:")
    print(f"  Cluster means: {[f'{x:.3f}' for x in metrics['cluster_means']]}")
    print(f"  Malicious cluster ID: {metrics['malicious_cluster_id']}")
    print(f"  Malicious cluster mean: {metrics['malicious_cluster_mean']:.3f}")


def demonstrate_clustering():
    """Demonstrate K-means clustering."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING K-MEANS CLUSTERING")
    print("=" * 60)
    
    # Test with different loss patterns
    test_cases = [
        {
            "name": "Clear separation",
            "losses": [0.1, 0.12, 0.11, 0.8, 0.9, 0.85],
            "expected": "Should detect last 3 as malicious"
        },
        {
            "name": "Noisy separation",
            "losses": [0.1, 0.12, 0.11, 0.15, 0.8, 0.9, 0.85, 0.16],
            "expected": "May have false positives/negatives"
        },
        {
            "name": "Uniform losses",
            "losses": [0.5, 0.51, 0.49, 0.52, 0.48, 0.50],
            "expected": "May not detect any malicious clients"
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  Losses: {[f'{x:.3f}' for x in case['losses']]}")
        print(f"  Expected: {case['expected']}")
        
        malicious_indices, metrics = eliminate_kmeans(case['losses'], n_clusters=2)
        print(f"  Detected malicious: {malicious_indices}")
        print(f"  Cluster means: {[f'{x:.3f}' for x in metrics['cluster_means']]}")


def main():
    """Main demonstration function."""
    print("Federated Learning Attack Detection System - Example Usage")
    print("=" * 70)
    
    # Demonstrate LDP
    demonstrate_ldp()
    
    # Demonstrate detection
    demonstrate_detection()
    
    # Demonstrate clustering
    demonstrate_clustering()
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED")
    print("=" * 70)
    print("\nTo run the full experiment:")
    print("1. Build Docker image: docker build -t fl-attack-detect:latest .")
    print("2. Run experiment: docker run -v /local/dataset:/app/dataset -v /local/results:/app/results fl-attack-detect:latest")
    print("3. Check results in /local/results/ directory")
    
    print("\nTo run smoke tests:")
    print("python tests/test_attack_detection.py")


if __name__ == "__main__":
    main()



