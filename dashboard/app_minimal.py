#!/usr/bin/env python3
"""
Minimal working version that avoids the JSON serialization issue
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
import requests
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fl-dashboard-secret-key-2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class FLDashboard:
    def __init__(self):
        self.clients = {}
        self.server_process = None
        self.attack_detection_enabled = True
        self.detection_history = []
        self.detection_metrics = {
            'total_detections': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

    def apply_ldp(self, losses, epsilon=1.0, sensitivity=1e-4):
        """Apply Local Differential Privacy to client losses"""
        if not losses:
            return []
        
        # Add Laplace noise
        noise_scale = sensitivity / epsilon
        noisy_losses = []
        
        for loss in losses:
            noise = np.random.laplace(0, noise_scale)
            noisy_loss = loss + noise
            noisy_losses.append(max(0, noisy_loss))  # Ensure non-negative
        
        return noisy_losses

    def detect_malicious_clients(self, client_losses, true_malicious=None, epsilon=1.0):
        """Detect malicious clients using LDP and threshold-based detection"""
        if not client_losses or len(client_losses) < 2:
            return [], {}
        
        # Apply LDP
        noisy_losses = self.apply_ldp(client_losses, epsilon=epsilon, sensitivity=1e-4)
        
        # Simple threshold-based detection
        mean_loss = np.mean(noisy_losses)
        std_loss = np.std(noisy_losses)
        threshold = mean_loss + 0.5 * std_loss  # Threshold for malicious detection
        
        detected_malicious = [i for i, loss in enumerate(noisy_losses) if loss > threshold]
        
        # Calculate metrics if true labels are provided
        metrics = {}
        if true_malicious is not None:
            true_malicious_set = set(true_malicious)
            detected_malicious_set = set(detected_malicious)
            
            true_positives = len(true_malicious_set & detected_malicious_set)
            false_positives = len(detected_malicious_set - true_malicious_set)
            false_negatives = len(true_malicious_set - detected_malicious_set)
            
            accuracy = (true_positives + len(client_losses) - len(true_malicious) - false_positives) / len(client_losses)
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1 = 2 * precision * recall / max(1, precision + recall)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            }
        
        return detected_malicious, metrics

    def run_attack_detection_demo(self, malicious_percentage=20, attack_type='label_flipping', epsilon=1.0):
        """Run attack detection demonstration - MINIMAL WORKING VERSION"""
        try:
            # Generate synthetic data
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
            
            # Run detection
            detected_malicious, metrics = self.detect_malicious_clients(
                client_losses, true_malicious, epsilon
            )
            
            # Create client details - CONVERT ALL TO PYTHON TYPES IMMEDIATELY
            client_details = []
            for i, loss in enumerate(client_losses):
                client_details.append({
                    'client_id': int(i),
                    'loss': float(loss),
                    'is_malicious': bool(i in true_malicious),
                    'attack_type': attack_type if i in true_malicious else 'none'
                })
            
            # Create detection result - CONVERT ALL TO PYTHON TYPES IMMEDIATELY
            detection_result = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'malicious_percentage': int(malicious_percentage),
                'attack_type': attack_type,
                'epsilon': float(epsilon),
                'true_malicious': [int(x) for x in true_malicious],
                'detected_malicious': [int(x) for x in detected_malicious],
                'client_losses': [float(x) for x in client_losses],
                'client_details': client_details,
                'metrics': metrics,
                'detection_summary': {
                    'total_clients': int(n_clients),
                    'true_malicious_count': int(len(true_malicious)),
                    'detected_malicious_count': int(len(detected_malicious)),
                    'false_positive_rate': float(metrics.get('false_positives', 0) / max(1, n_clients - len(true_malicious))),
                    'false_negative_rate': float(metrics.get('false_negatives', 0) / max(1, len(true_malicious))),
                    'detection_confidence': float(metrics.get('accuracy', 0))
                }
            }
            
            # Store the result (all types are already Python native)
            self.detection_history.append(detection_result)
            
            # Update overall metrics
            if 'accuracy' in metrics:
                self.detection_metrics['total_detections'] += 1
                self.detection_metrics['accuracy'] = float(metrics['accuracy'])
                self.detection_metrics['precision'] = float(metrics['precision'])
                self.detection_metrics['recall'] = float(metrics['recall'])
                self.detection_metrics['f1_score'] = float(metrics['f1_score'])
                self.detection_metrics['true_positives'] = int(metrics['true_positives'])
                self.detection_metrics['false_positives'] = int(metrics['false_positives'])
                self.detection_metrics['false_negatives'] = int(metrics['false_negatives'])
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error running attack detection demo: {e}")
            return {'error': str(e)}

    def get_detection_status(self):
        """Get current attack detection status"""
        return {
            'enabled': self.attack_detection_enabled,
            'total_detections': self.detection_metrics['total_detections'],
            'accuracy': self.detection_metrics['accuracy'],
            'precision': self.detection_metrics['precision'],
            'recall': self.detection_metrics['recall'],
            'f1_score': self.detection_metrics['f1_score']
        }

# Initialize dashboard
dashboard = FLDashboard()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/research')
def research():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/api/detection/run_demo', methods=['POST'])
def run_detection_demo():
    """Run attack detection demo - MINIMAL WORKING VERSION"""
    try:
        data = request.get_json() or {}
        malicious_percentage = data.get('malicious_percentage', 20)
        attack_type = data.get('attack_type', 'label_flipping')
        epsilon = data.get('epsilon', 1.0)
        
        result = dashboard.run_attack_detection_demo(malicious_percentage, attack_type, epsilon)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 500
        
        # All data is already Python native types, so jsonify should work
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        logger.error(f"Error running detection demo: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detection/history')
def get_detection_history():
    """Get attack detection history"""
    return jsonify({
        'success': True,
        'history': dashboard.detection_history[-10:]  # Last 10 detections
    })

@app.route('/api/detection/status')
def get_detection_status():
    """Get attack detection status"""
    return jsonify({
        'success': True,
        'status': dashboard.get_detection_status()
    })

@app.route('/api/demo/clients')
def get_demo_clients():
    """Get demo clients for attack simulation"""
    clients = []
    for i in range(10):
        clients.append({
            'id': i,
            'name': f'Client {i+1}',
            'status': 'active',
            'last_seen': time.time()
        })
    return jsonify({'success': True, 'clients': clients})

@app.route('/api/system/status')
def get_system_status():
    """Get system status"""
    return jsonify({
        'success': True,
        'status': 'running',
        'clients_count': len(dashboard.clients),
        'timestamp': time.time()
    })

@app.route('/api/metrics/training')
def get_training_metrics():
    """Get training metrics"""
    return jsonify({
        'success': True,
        'metrics': {
            'round': 1,
            'accuracy': 0.85,
            'loss': 0.15,
            'clients': 10
        }
    })

@app.route('/api/metrics/clients')
def get_client_metrics():
    """Get client metrics"""
    return jsonify({
        'success': True,
        'clients': []
    })

@app.route('/api/metrics/system')
def get_system_metrics():
    """Get system metrics"""
    return jsonify({
        'success': True,
        'metrics': {
            'cpu_usage': 25.5,
            'memory_usage': 60.2,
            'disk_usage': 45.8
        }
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

