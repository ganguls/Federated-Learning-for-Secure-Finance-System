#!/usr/bin/env python3
"""
Standalone attack detection service that runs independently of Flask
"""

import json
import sys
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

def run_attack_detection(malicious_percentage=20, attack_type='label_flipping', epsilon=1.0):
    """Run attack detection with all parameters"""
    try:
        # Run detection logic
        n_clients = 10
        n_features = 5
        n_samples_per_client = 100
        
        # Generate random data
        X = np.random.randn(n_clients * n_samples_per_client, n_features)
        y = np.random.randint(0, 2, n_clients * n_samples_per_client)
        
        # Split data among clients
        client_data = []
        for i in range(n_clients):
            start_idx = i * n_samples_per_client
            end_idx = (i + 1) * n_samples_per_client
            X_client = X[start_idx:end_idx]
            y_client = y[start_idx:end_idx]
            client_data.append((X_client, y_client))
        
        # Simulate attack
        n_malicious = int(n_clients * malicious_percentage / 100)
        true_malicious = random.sample(range(n_clients), n_malicious)
        
        # Generate client losses
        client_losses = []
        for i, (X_client, y_client) in enumerate(client_data):
            # Train a simple model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_client, y_client)
            
            # Calculate loss
            y_pred = model.predict(X_client)
            loss = 1 - accuracy_score(y_client, y_pred)
            
            # Apply attack if client is malicious
            if i in true_malicious:
                if attack_type == 'label_flipping':
                    loss = loss * 2.0  # Increase loss
                elif attack_type == 'gradient_poisoning':
                    loss = loss * 1.5
                elif attack_type == 'backdoor':
                    loss = loss * 3.0
            
            client_losses.append(loss)
        
        # Apply LDP
        noise_scale = 1e-4 / epsilon
        noisy_losses = []
        for loss in client_losses:
            noise = np.random.laplace(0, noise_scale)
            noisy_loss = loss + noise
            noisy_losses.append(max(0, noisy_loss))
        
        # Simple threshold-based detection
        mean_loss = np.mean(noisy_losses)
        std_loss = np.std(noisy_losses)
        threshold = mean_loss + 0.5 * std_loss
        detected_malicious = [i for i, loss in enumerate(noisy_losses) if loss > threshold]
        
        # Calculate metrics
        true_malicious_set = set(true_malicious)
        detected_malicious_set = set(detected_malicious)
        
        true_positives = len(true_malicious_set & detected_malicious_set)
        false_positives = len(detected_malicious_set - true_malicious_set)
        false_negatives = len(true_malicious_set - detected_malicious_set)
        
        accuracy = (true_positives + n_clients - len(true_malicious) - false_positives) / n_clients
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(1, precision + recall)
        
        # Create result with all Python native types
        result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'malicious_percentage': int(malicious_percentage),
            'attack_type': attack_type,
            'epsilon': float(epsilon),
            'true_malicious': [int(x) for x in true_malicious],
            'detected_malicious': [int(x) for x in detected_malicious],
            'client_losses': [float(x) for x in client_losses],
            'client_details': [
                {
                    'client_id': int(i),
                    'loss': float(loss),
                    'is_malicious': bool(i in true_malicious),
                    'attack_type': attack_type if i in true_malicious else 'none'
                }
                for i, loss in enumerate(client_losses)
            ],
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            },
            'detection_summary': {
                'total_clients': int(n_clients),
                'true_malicious_count': int(len(true_malicious)),
                'detected_malicious_count': int(len(detected_malicious)),
                'false_positive_rate': float(false_positives / max(1, n_clients - len(true_malicious))),
                'false_negative_rate': float(false_negatives / max(1, len(true_malicious))),
                'detection_confidence': float(accuracy)
            }
        }
        
        return {'success': True, 'result': result}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    # Get parameters from command line
    if len(sys.argv) >= 4:
        malicious_percentage = int(sys.argv[1])
        attack_type = sys.argv[2]
        epsilon = float(sys.argv[3])
    else:
        malicious_percentage = 20
        attack_type = 'label_flipping'
        epsilon = 1.0
    
    # Run detection
    result = run_attack_detection(malicious_percentage, attack_type, epsilon)
    
    # Output JSON
    print(json.dumps(result))

