#!/usr/bin/env python3
"""
Pure Flask app without SocketIO to avoid the conflict
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
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
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

# PURE FLASK: No SocketIO
@app.route('/api/detection/run_demo', methods=['POST'])
def run_detection_demo():
    """Run attack detection demo - PURE FLASK APPROACH"""
    try:
        data = request.get_json() or {}
        malicious_percentage = data.get('malicious_percentage', 20)
        attack_type = data.get('attack_type', 'label_flipping')
        epsilon = data.get('epsilon', 1.0)
        
        # Call external Python script
        script_path = os.path.join(os.path.dirname(__file__), '..', 'attack_detection_service.py')
        
        # Run the external script
        result = subprocess.run([
            sys.executable, 
            script_path, 
            str(malicious_percentage), 
            attack_type, 
            str(epsilon)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Parse the JSON output
            response_data = json.loads(result.stdout.strip())
            
            # Store in history if successful
            if response_data.get('success') and 'result' in response_data:
                dashboard.detection_history.append(response_data['result'])
                
                # Update metrics
                metrics = response_data['result'].get('metrics', {})
                if 'accuracy' in metrics:
                    dashboard.detection_metrics['total_detections'] += 1
                    dashboard.detection_metrics['accuracy'] = float(metrics['accuracy'])
                    dashboard.detection_metrics['precision'] = float(metrics['precision'])
                    dashboard.detection_metrics['recall'] = float(metrics['recall'])
                    dashboard.detection_metrics['f1_score'] = float(metrics['f1_score'])
                    dashboard.detection_metrics['true_positives'] = int(metrics['true_positives'])
                    dashboard.detection_metrics['false_positives'] = int(metrics['false_positives'])
                    dashboard.detection_metrics['false_negatives'] = int(metrics['false_negatives'])
            
            return jsonify(response_data)
        else:
            return jsonify({
                'success': False, 
                'error': f'External service failed: {result.stderr}'
            }), 500
        
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
    app.run(host='0.0.0.0', port=5000, debug=False)

