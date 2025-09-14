"""
Data Poisoning Detection Utilities for Tabular Data
Extracted from the original project for integration into target FL systems
"""

import copy
import torch
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

def apply_ldp(losses, epsilon, sensitivity=0.001):
    """
    Apply Local Differential Privacy to loss values
    
    Args:
        losses: List of loss values from clients
        epsilon: Privacy parameter (smaller = more private)
        sensitivity: Sensitivity parameter for LDP
    
    Returns:
        noisy_losses: Loss values with added noise
    """
    # Compute the scale of the Laplace noise
    b = sensitivity / epsilon
    
    # Generate Laplace noise for each data point
    noise = np.random.laplace(0, b, len(losses))
    
    # Add the noise to the original data
    noisy_data = losses + noise
    
    return noisy_data

def eliminate_kmeans(info):
    """
    Detect malicious clients using K-means clustering on loss values
    
    Args:
        info: Tuple of (local_losses, local_weights, selected_users)
    
    Returns:
        selected_users_new: List of clean client IDs
        attackers: List of detected attacker IDs
    """
    (local_losses, local_weights, selected_users) = info
    data = np.array(local_losses).reshape(-1, 1)
    
    # Apply KMeans clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(data)
    
    # Get cluster assignments and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # We assume the cluster with the higher centroid value corresponds to attackers
    # This is because attackers might have higher loss values than honest clients
    attacker_cluster = np.argmax(centroids)
    
    # Get the indices of the attacker cluster
    attackers = np.where(labels == attacker_cluster)[0]
    selected_users_new = [selected_users[i] for i in range(len(selected_users)) if i not in attackers]
    
    return selected_users_new, attackers

def eliminate_fixed_percentage(info, n_train_clients, percentage):
    """
    Eliminate a fixed percentage of clients with highest losses
    
    Args:
        info: Tuple of (local_losses, local_weights, selected_users)
        n_train_clients: Total number of training clients
        percentage: Percentage of clients to eliminate (0.0 to 1.0)
    
    Returns:
        selected_users: List of remaining client IDs
        attackers: List of eliminated client IDs
    """
    (local_losses, local_weights, selected_users) = info
    
    sorted_indices = sorted(range(len(selected_users)), key=lambda k: local_losses[k])
    
    local_losses = [local_losses[i] for i in sorted_indices]
    local_weights = [local_weights[i] for i in sorted_indices]
    selected_users = [selected_users[i] for i in sorted_indices]
    
    idx = int((1 - percentage) * n_train_clients)
    
    attackers = selected_users[idx:]
    selected_users = selected_users[:idx]
    
    return selected_users, attackers

def eliminate_with_z_score(info, threshold):
    """
    Eliminate clients with loss values beyond z-score threshold
    
    Args:
        info: Tuple of (local_losses, local_weights, selected_users)
        threshold: Z-score threshold for elimination
    
    Returns:
        selected_users_new: List of remaining client IDs
        attackers: List of eliminated client IDs
    """
    (local_losses, local_weights, selected_users) = info
    
    sorted_indices = sorted(range(len(selected_users)), key=lambda k: local_losses[k])
    
    local_losses = [local_losses[i] for i in sorted_indices]
    local_weights = [local_weights[i] for i in sorted_indices]
    selected_users = [selected_users[i] for i in sorted_indices]
    
    mean = np.mean(local_losses)
    std = np.std(local_losses)
    z_scores = [(loss - mean) / std for loss in local_losses]
    
    attackers = np.where(np.abs(z_scores) > threshold)[0]
    
    selected_users_new = [selected_users[i] for i in range(len(selected_users)) if i not in attackers]
    
    return selected_users_new, attackers

def calculate_accuracy(actual, predicted, total_clients):
    """
    Calculate detection accuracy
    
    Args:
        actual: Set of actual attacker IDs
        predicted: Set of predicted attacker IDs
        total_clients: Set of all client IDs
    
    Returns:
        accuracy: Detection accuracy (0.0 to 1.0)
    """
    TP = len(set(actual) & set(predicted))
    FP = len(set(predicted) - set(actual))
    FN = len(set(actual) - set(predicted))
    TN = len(set(total_clients) - set(actual) - set(predicted))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    return accuracy

def calculate_f1_score(actual, predicted):
    """
    Calculate F1 score for detection
    
    Args:
        actual: Set of actual attacker IDs
        predicted: Set of predicted attacker IDs
    
    Returns:
        f1_score: F1 score (0.0 to 1.0)
    """
    TP = len(set(actual) & set(predicted))
    FP = len(set(predicted) - set(actual))
    FN = len(set(actual) - set(predicted))
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

def calculate_precision(actual, predicted):
    """
    Calculate precision for detection
    
    Args:
        actual: Set of actual attacker IDs
        predicted: Set of predicted attacker IDs
    
    Returns:
        precision: Precision score (0.0 to 1.0)
    """
    TP = len(set(actual) & set(predicted))
    FP = len(set(predicted) - set(actual))
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    return precision

def calculate_recall(actual, predicted):
    """
    Calculate recall for detection
    
    Args:
        actual: Set of actual attacker IDs
        predicted: Set of predicted attacker IDs
    
    Returns:
        recall: Recall score (0.0 to 1.0)
    """
    TP = len(set(actual) & set(predicted))
    FN = len(set(actual) - set(predicted))
    
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    return recall

def average_weights(weight_list):
    """
    Average weights from multiple clients using FedAvg
    
    Args:
        weight_list: List of model state dictionaries
    
    Returns:
        w_avg: Averaged model state dictionary
    """
    w_avg = copy.deepcopy(weight_list[0])
    
    for key in w_avg.keys():
        for i in range(1, len(weight_list)):
            w_avg[key] += weight_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weight_list))
    
    return w_avg

def detect_malicious_clients(client_losses, client_weights, client_ids, 
                           method='kmeans', epsilon=1.0, sensitivity=0.0001):
    """
    Main function to detect malicious clients
    
    Args:
        client_losses: List of loss values from clients
        client_weights: List of model weights from clients
        client_ids: List of client IDs
        method: Detection method ('kmeans', 'fixed_percentage', 'z_score')
        epsilon: LDP privacy parameter
        sensitivity: LDP sensitivity parameter
    
    Returns:
        clean_clients: List of clean client IDs
        attackers: List of detected attacker IDs
    """
    # Apply LDP for privacy protection
    noisy_losses = apply_ldp(client_losses, epsilon, sensitivity)
    
    # Prepare info tuple
    info = (noisy_losses, client_weights, client_ids)
    
    # Apply detection method
    if method == 'kmeans':
        clean_clients, attackers = eliminate_kmeans(info)
    elif method == 'fixed_percentage':
        clean_clients, attackers = eliminate_fixed_percentage(info, len(client_ids), 0.2)
    elif method == 'z_score':
        clean_clients, attackers = eliminate_with_z_score(info, threshold=2.0)
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    return clean_clients, attackers
