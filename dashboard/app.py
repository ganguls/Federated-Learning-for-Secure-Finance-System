#!/usr/bin/env python3
"""
Pure Flask app without SocketIO to avoid the conflict
Enhanced with real FL detection integration
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

# Import detection API
from detection_api import detection_bp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fl-dashboard-secret-key-2024'
CORS(app)

# Register detection blueprint
app.register_blueprint(detection_bp)

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

@app.route('/logout')
def logout():
    """Logout route"""
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

# Enhanced detection endpoint for real FL integration
@app.route('/api/detection/run_enhanced', methods=['POST'])
def run_enhanced_detection():
    """Run enhanced detection on real FL results"""
    try:
        data = request.get_json() or {}
        use_cached = data.get('use_cached', True)
        
        # This will use the detection API manager
        from detection_api import api_manager
        
        result = api_manager.run_detection(use_cached=use_cached)
        
        if result.get('success'):
            # Update dashboard metrics
            detection_metrics = result.get('detection_metrics', {})
            if detection_metrics:
                dashboard.detection_metrics['total_detections'] += 1
                dashboard.detection_metrics['accuracy'] = float(detection_metrics.get('accuracy', 0))
                dashboard.detection_metrics['precision'] = float(detection_metrics.get('precision', 0))
                dashboard.detection_metrics['recall'] = float(detection_metrics.get('recall', 0))
                dashboard.detection_metrics['f1_score'] = float(detection_metrics.get('f1_score', 0))
            
            # Save detection results to Docker volume
            try:
                detection_results_file = "/app/detection_results/latest_detection.json"
                os.makedirs(os.path.dirname(detection_results_file), exist_ok=True)
                with open(detection_results_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"Detection results saved to {detection_results_file}")
            except Exception as e:
                logger.warning(f"Could not save detection results: {e}")
            
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in enhanced detection: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

# PURE FLASK: No SocketIO
@app.route('/api/detection/run_demo', methods=['POST'])
def run_detection_demo():
    """Run comprehensive attack detection demo with all 3 methods"""
    try:
        data = request.get_json() or {}
        malicious_percentage = data.get('malicious_percentage', 20)
        attack_type = data.get('attack_type', 'label_flipping')
        epsilon = data.get('epsilon', 1.0)
        
        # Import comprehensive detection pipeline
        try:
            sys.path.append('/app')
            from comprehensive_attack_detection import ComprehensiveDetectionPipeline
            
            # Initialize and run comprehensive detection
            pipeline = ComprehensiveDetectionPipeline()
            results = pipeline.run_comprehensive_detection(
                malicious_percentage=malicious_percentage,
                attack_type=attack_type,
                epsilon=epsilon
            )
            
            if 'error' in results:
                return jsonify({
                    'success': False,
                    'error': results['error']
                }), 500
            
            # Store in history if successful
            dashboard.detection_history.append(results)
            
            # Update metrics with ensemble results
            ensemble_metrics = results.get('fusion_results', {}).get('ensemble_metrics', {})
            if ensemble_metrics:
                dashboard.detection_metrics['total_detections'] += 1
                dashboard.detection_metrics['accuracy'] = float(ensemble_metrics.get('accuracy', 0))
                dashboard.detection_metrics['f1_score'] = float(ensemble_metrics.get('f1_score', 0))
                
                # Calculate precision and recall from individual methods
                method_results = results.get('detection_methods', {})
                avg_precision = np.mean([
                    method.get('metrics', {}).get('precision', 0) 
                    for method in method_results.values()
                ])
                avg_recall = np.mean([
                    method.get('metrics', {}).get('recall', 0) 
                    for method in method_results.values()
                ])
                
                dashboard.detection_metrics['precision'] = float(avg_precision)
                dashboard.detection_metrics['recall'] = float(avg_recall)
            
            # Return comprehensive results
            return jsonify({
                'success': True,
                'result': results
            })
            
        except ImportError as e:
            logger.error(f"Could not import comprehensive detection: {e}")
            # Fallback to basic detection
            return run_basic_detection_fallback(malicious_percentage, attack_type, epsilon)
        
    except Exception as e:
        logger.error(f"Error running comprehensive detection demo: {e}")
        response = jsonify({'success': False, 'error': str(e)})
        response.status_code = 500
        return response

def run_basic_detection_fallback(malicious_percentage, attack_type, epsilon):
    """Fallback to basic detection if comprehensive detection fails"""
    try:
        # Call external Python script - use absolute path in Docker container
        script_path = '/app/attack_detection_service.py'
        
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
            response = jsonify({
                'success': False, 
                'error': f'External service failed: {result.stderr}'
            })
            response.status_code = 500
            return response
        
    except Exception as e:
        logger.error(f"Error in fallback detection: {e}")
        response = jsonify({'success': False, 'error': str(e)})
        response.status_code = 500
        return response

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
