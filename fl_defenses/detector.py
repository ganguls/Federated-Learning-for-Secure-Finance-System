"""
Federated Learning Defense Mechanisms
====================================

This module implements Local Differential Privacy (LDP) and clustering-based
detection mechanisms for identifying malicious clients in federated learning.

Author: FL Defense System
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_ldp(local_losses: List[float], epsilon: float = 1.0, sensitivity: float = 1e-4) -> List[float]:
    """
    Apply Local Differential Privacy (LDP) noise to local losses.
    
    Uses the Laplace mechanism to add calibrated noise to preserve privacy
    while maintaining utility for attack detection.
    
    Args:
        local_losses: List of local training losses from clients
        epsilon: Privacy parameter (smaller = more private, larger = more utility)
        sensitivity: Sensitivity parameter for the Laplace mechanism
        
    Returns:
        List of noisy losses with LDP applied
        
    Note:
        - Lower epsilon values provide stronger privacy guarantees
        - Sensitivity should be tuned based on the expected range of losses
        - Default sensitivity=1e-4 is conservative for normalized losses
    """
    if not local_losses:
        return []
    
    # Convert to numpy array for easier manipulation
    losses = np.array(local_losses)
    
    # Calculate noise scale based on epsilon and sensitivity
    noise_scale = sensitivity / epsilon
    
    # Generate Laplace noise
    noise = np.random.laplace(0, noise_scale, size=losses.shape)
    
    # Add noise to losses
    noisy_losses = losses + noise
    
    logger.info(f"Applied LDP with epsilon={epsilon}, sensitivity={sensitivity}")
    logger.info(f"Original loss range: [{losses.min():.6f}, {losses.max():.6f}]")
    logger.info(f"Noisy loss range: [{noisy_losses.min():.6f}, {noisy_losses.max():.6f}]")
    
    return noisy_losses.tolist()


def eliminate_kmeans(noisy_losses: List[float], n_clusters: int = 2, 
                    random_state: int = 42) -> Tuple[List[int], Dict]:
    """
    Use K-means clustering to identify malicious clients based on noisy losses.
    
    Assumes that malicious clients will have higher losses and will cluster
    together, while benign clients will have lower, more consistent losses.
    
    Args:
        noisy_losses: List of LDP-noisy losses from clients
        n_clusters: Number of clusters for K-means (default: 2 for benign/malicious)
        random_state: Random seed for reproducible results
        
    Returns:
        Tuple of (malicious_client_indices, detection_metrics)
        
    Note:
        - The cluster with the highest mean loss is considered malicious
        - This is a heuristic that works well for data poisoning attacks
        - May produce false positives with high noise levels
    """
    if not noisy_losses or len(noisy_losses) < n_clusters:
        return [], {"error": "Insufficient data for clustering"}
    
    # Convert to numpy array and reshape for K-means
    losses = np.array(noisy_losses).reshape(-1, 1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(losses)
    
    # Calculate cluster statistics
    cluster_means = []
    cluster_sizes = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_losses = losses[cluster_mask]
        cluster_means.append(np.mean(cluster_losses))
        cluster_sizes.append(np.sum(cluster_mask))
    
    # Identify malicious cluster (highest mean loss)
    malicious_cluster_id = np.argmax(cluster_means)
    malicious_client_indices = np.where(cluster_labels == malicious_cluster_id)[0].tolist()
    
    # Calculate detection metrics
    detection_metrics = {
        "cluster_means": cluster_means,
        "cluster_sizes": cluster_sizes,
        "malicious_cluster_id": malicious_cluster_id,
        "malicious_cluster_mean": cluster_means[malicious_cluster_id],
        "benign_cluster_mean": np.mean([cluster_means[i] for i in range(n_clusters) if i != malicious_cluster_id]),
        "total_clients": len(noisy_losses),
        "detected_malicious": len(malicious_client_indices)
    }
    
    logger.info(f"K-means clustering completed: {len(malicious_client_indices)}/{len(noisy_losses)} clients detected as malicious")
    logger.info(f"Malicious cluster mean loss: {cluster_means[malicious_cluster_id]:.6f}")
    
    return malicious_client_indices, detection_metrics


def calculate_accuracy(true_malicious: List[int], detected_malicious: List[int], 
                      total_clients: int) -> float:
    """
    Calculate detection accuracy for malicious client identification.
    
    Args:
        true_malicious: List of indices of actually malicious clients
        detected_malicious: List of indices of detected malicious clients
        total_clients: Total number of clients in the system
        
    Returns:
        Detection accuracy as a float between 0.0 and 1.0
    """
    if total_clients == 0:
        return 0.0
    
    # Convert to sets for easier comparison
    true_set = set(true_malicious)
    detected_set = set(detected_malicious)
    
    # Calculate true positives, false positives, true negatives
    true_positives = len(true_set.intersection(detected_set))
    false_positives = len(detected_set - true_set)
    true_negatives = total_clients - len(true_set) - false_positives
    
    # Accuracy = (TP + TN) / Total
    accuracy = (true_positives + true_negatives) / total_clients
    
    return accuracy


def calculate_f1_score(true_malicious: List[int], detected_malicious: List[int]) -> float:
    """
    Calculate F1 score for malicious client detection.
    
    Args:
        true_malicious: List of indices of actually malicious clients
        detected_malicious: List of indices of detected malicious clients
        
    Returns:
        F1 score as a float between 0.0 and 1.0
    """
    if not true_malicious and not detected_malicious:
        return 1.0  # Perfect score when no malicious clients exist
    
    if not true_malicious or not detected_malicious:
        return 0.0  # No true positives possible
    
    # Convert to sets
    true_set = set(true_malicious)
    detected_set = set(detected_malicious)
    
    # Calculate precision and recall
    true_positives = len(true_set.intersection(detected_set))
    false_positives = len(detected_set - true_set)
    false_negatives = len(true_set - detected_set)
    
    # Handle edge cases
    if true_positives == 0:
        return 0.0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def detect_malicious_clients(local_losses: List[float], true_malicious: List[int] = None,
                           epsilon: float = 1.0, sensitivity: float = 1e-4,
                           n_clusters: int = 2) -> Dict:
    """
    Complete pipeline for detecting malicious clients using LDP and K-means.
    
    Args:
        local_losses: List of local training losses from clients
        true_malicious: List of indices of actually malicious clients (for evaluation)
        epsilon: LDP privacy parameter
        sensitivity: LDP sensitivity parameter
        n_clusters: Number of clusters for K-means
        
    Returns:
        Dictionary containing detection results and metrics
    """
    if not local_losses:
        return {"error": "No losses provided"}
    
    # Normalize losses if they're too large/small for LDP
    losses = np.array(local_losses)
    loss_mean = np.mean(losses)
    
    # Apply normalization if losses are orders of magnitude larger than sensitivity
    if loss_mean > sensitivity * 1000:
        normalized_losses = losses / loss_mean
        logger.info(f"Normalized losses by mean: {loss_mean:.6f}")
    else:
        normalized_losses = losses
    
    # Apply LDP
    noisy_losses = apply_ldp(normalized_losses.tolist(), epsilon, sensitivity)
    
    # Apply K-means detection
    malicious_indices, detection_metrics = eliminate_kmeans(noisy_losses, n_clusters)
    
    # Calculate evaluation metrics if ground truth is available
    results = {
        "detected_malicious": malicious_indices,
        "detection_metrics": detection_metrics,
        "original_losses": local_losses,
        "noisy_losses": noisy_losses,
        "normalization_factor": loss_mean if loss_mean > sensitivity * 1000 else 1.0
    }
    
    if true_malicious is not None:
        accuracy = calculate_accuracy(true_malicious, malicious_indices, len(local_losses))
        f1 = calculate_f1_score(true_malicious, malicious_indices)
        
        results.update({
            "accuracy": accuracy,
            "f1_score": f1,
            "true_malicious": true_malicious
        })
        
        logger.info(f"Detection accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
    
    return results


# Unit tests in comments for reference
"""
Test cases for the detector module:

def test_apply_ldp():
    # Test basic LDP functionality
    losses = [0.1, 0.2, 0.15, 0.3, 0.25]
    noisy = apply_ldp(losses, epsilon=1.0, sensitivity=1e-4)
    assert len(noisy) == len(losses)
    assert all(isinstance(x, float) for x in noisy)

def test_eliminate_kmeans():
    # Test clustering with clear separation
    losses = [0.1, 0.12, 0.11, 0.8, 0.9, 0.85]  # First 3 benign, last 3 malicious
    malicious, metrics = eliminate_kmeans(losses, n_clusters=2)
    assert len(malicious) > 0
    assert "cluster_means" in metrics

def test_calculate_accuracy():
    # Test accuracy calculation
    true_malicious = [3, 4, 5]
    detected_malicious = [3, 4, 6]  # One false positive, one false negative
    accuracy = calculate_accuracy(true_malicious, detected_malicious, 6)
    assert 0.0 <= accuracy <= 1.0

def test_calculate_f1_score():
    # Test F1 score calculation
    true_malicious = [0, 1]
    detected_malicious = [0, 2]  # One true positive, one false positive, one false negative
    f1 = calculate_f1_score(true_malicious, detected_malicious)
    assert 0.0 <= f1 <= 1.0

def test_detect_malicious_clients():
    # Test complete pipeline
    losses = [0.1, 0.12, 0.11, 0.8, 0.9, 0.85]
    true_malicious = [3, 4, 5]
    results = detect_malicious_clients(losses, true_malicious)
    assert "detected_malicious" in results
    assert "accuracy" in results
    assert "f1_score" in results
"""



