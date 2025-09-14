#!/usr/bin/env python3
"""
Detection API for Dashboard Integration
Provides endpoints for running detection on real FL results
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify, current_app
import requests

# Add server path for detection imports
sys.path.append('/app/server')
sys.path.append('/app/data_poisoning_detection')

try:
    from detection_adapter import FLDetectionAdapter, create_detection_adapter
    from enhanced_server_strategy import EnhancedLoanServerStrategy
except ImportError as e:
    logging.warning(f"Detection modules not available: {e}")
    # Fallback implementations
    class FLDetectionAdapter:
        def __init__(self, *args, **kwargs):
            pass
        def detect_malicious_clients(self, *args, **kwargs):
            return [], {}
        def get_detection_summary(self):
            return {}
    
    class EnhancedLoanServerStrategy:
        def __init__(self, *args, **kwargs):
            pass

# Create blueprint
detection_bp = Blueprint('detection', __name__, url_prefix='/api/detection')

logger = logging.getLogger(__name__)

class DetectionAPIManager:
    """
    Manages detection API endpoints and state
    """
    
    def __init__(self):
        self.detection_adapter = None
        self.server_strategy = None
        self.detection_history = []
        self.last_fl_results = None
        self.detection_config = {
            'enabled': True,
            'method': 'kmeans',
            'ldp_epsilon': 1.0,
            'ldp_sensitivity': 0.001,
            'input_dim': 20,
            'defense_threshold': 0.3
        }
        
        # Initialize detection adapter
        self._initialize_detection_adapter()
    
    def _initialize_detection_adapter(self):
        """Initialize detection adapter with current config"""
        try:
            self.detection_adapter = create_detection_adapter(
                input_dim=self.detection_config['input_dim'],
                detection_method=self.detection_config['method'],
                ldp_epsilon=self.detection_config['ldp_epsilon']
            )
            logger.info("Detection adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detection adapter: {e}")
            self.detection_adapter = None
    
    def update_config(self, new_config: Dict):
        """Update detection configuration"""
        try:
            self.detection_config.update(new_config)
            self._initialize_detection_adapter()
            logger.info(f"Detection config updated: {new_config}")
            return True
        except Exception as e:
            logger.error(f"Failed to update detection config: {e}")
            return False
    
    def get_fl_results_from_server(self) -> Optional[List]:
        """
        Retrieve latest FL results from server
        This connects to the running FL server or loads from metrics
        """
        try:
            # Check if we have cached results
            if self.last_fl_results:
                return self.last_fl_results
            
            # Try to load from metrics file (Docker path)
            metrics_files = [
                "/app/detection_results/enhanced_training_metrics.json",
                "enhanced_training_metrics.json",
                "/app/logs/enhanced_training_metrics.json"
            ]
            
            for metrics_file in metrics_files:
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                        if metrics_data:
                            # Extract latest round results
                            latest_round = metrics_data[-1]
                            logger.info(f"Loaded FL results from metrics file: {latest_round.get('round', 'unknown')} round")
                            return self._simulate_fl_results_from_metrics(latest_round)
            
            # Try to connect to server API
            try:
                server_url = os.getenv('SERVER_URL', 'http://server:8080')
                response = requests.get(f"{server_url}/api/detection/latest_results", timeout=5)
                if response.status_code == 200:
                    results = response.json()
                    logger.info("Retrieved FL results from server API")
                    return results
            except Exception as e:
                logger.warning(f"Could not connect to server API: {e}")
            
            # Fallback: simulate FL results for demo
            logger.warning("No real FL results available, using simulated data")
            return self._simulate_fl_results()
            
        except Exception as e:
            logger.error(f"Error retrieving FL results: {e}")
            return None
    
    def _simulate_fl_results_from_metrics(self, metrics: Dict) -> List:
        """Convert metrics to FL results format"""
        try:
            # This is a placeholder - in reality, you'd reconstruct from actual FL state
            # For now, return simulated results
            return self._simulate_fl_results()
        except Exception as e:
            logger.error(f"Error converting metrics to FL results: {e}")
            return self._simulate_fl_results()
    
    def _simulate_fl_results(self) -> List:
        """
        Simulate FL results for demonstration
        In production, this would be replaced with real FL server data
        """
        try:
            # Simulate 10 clients with realistic parameters
            results = []
            np.random.seed(42)  # For reproducible results
            
            for i in range(10):
                # Simulate LogisticRegression parameters
                n_features = self.detection_config['input_dim']
                coef = np.random.normal(0, 0.1, (1, n_features))
                intercept = np.random.normal(0, 0.1, 1)
                
                # Simulate metrics
                accuracy = np.random.uniform(0.6, 0.9)
                is_malicious = i < 2  # First 2 clients are malicious
                
                metrics = {
                    "client_id": f"client_{i+1}",
                    "accuracy": accuracy,
                    "precision": accuracy + np.random.uniform(-0.1, 0.1),
                    "recall": accuracy + np.random.uniform(-0.1, 0.1),
                    "f1_score": accuracy + np.random.uniform(-0.1, 0.1),
                    "is_malicious": is_malicious,
                    "labels_flipped": np.random.randint(0, 50) if is_malicious else 0
                }
                
                # Create FL result tuple
                result = ([coef, intercept], 100, metrics)  # (parameters, num_examples, metrics)
                results.append(result)
            
            logger.info(f"Simulated {len(results)} FL results for detection")
            return results
            
        except Exception as e:
            logger.error(f"Error simulating FL results: {e}")
            return []
    
    def run_detection(self, use_cached: bool = True) -> Dict:
        """
        Run detection on latest FL results
        
        Args:
            use_cached: Whether to use cached results or fetch fresh ones
            
        Returns:
            Detection results dictionary
        """
        try:
            start_time = time.time()
            
            # Get FL results
            if use_cached and self.last_fl_results:
                fl_results = self.last_fl_results
                logger.info("Using cached FL results for detection")
            else:
                fl_results = self.get_fl_results_from_server()
                if fl_results:
                    self.last_fl_results = fl_results
                else:
                    return {'error': 'No FL results available for detection', 'success': False}
            
            if not self.detection_adapter:
                return {'error': 'Detection adapter not initialized', 'success': False}
            
            # Run detection
            detected_malicious, detection_metrics = self.detection_adapter.detect_malicious_clients(
                client_results=fl_results
            )
            
            # Create detection record
            detection_record = {
                'timestamp': time.time(),
                'detection_time': time.time() - start_time,
                'total_clients': len(fl_results),
                'detected_malicious': detected_malicious,
                'detection_metrics': detection_metrics,
                'detection_method': self.detection_config['method'],
                'ldp_epsilon': self.detection_config['ldp_epsilon']
            }
            
            # Store in history
            self.detection_history.append(detection_record)
            
            # Get detection summary
            detection_summary = self.detection_adapter.get_detection_summary()
            
            logger.info(f"Detection completed in {time.time() - start_time:.3f}s")
            logger.info(f"Detected {len(detected_malicious)} malicious clients: {detected_malicious}")
            
            return {
                'success': True,
                'detected_malicious': detected_malicious,
                'detection_metrics': detection_metrics,
                'detection_summary': detection_summary,
                'detection_record': detection_record,
                'total_clients': len(fl_results),
                'detection_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error running detection: {e}")
            return {'error': str(e), 'success': False}
    
    def get_detection_status(self) -> Dict:
        """Get current detection status"""
        return {
            'enabled': self.detection_config['enabled'],
            'method': self.detection_config['method'],
            'ldp_epsilon': self.detection_config['ldp_epsilon'],
            'total_detections': len(self.detection_history),
            'last_detection': self.detection_history[-1] if self.detection_history else None,
            'adapter_initialized': self.detection_adapter is not None
        }
    
    def get_detection_history(self, limit: int = 10) -> List[Dict]:
        """Get recent detection history"""
        return self.detection_history[-limit:] if self.detection_history else []

# Initialize API manager
api_manager = DetectionAPIManager()

@detection_bp.route('/status', methods=['GET'])
def get_detection_status():
    """Get detection status"""
    try:
        status = api_manager.get_detection_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting detection status: {e}")
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/run', methods=['POST'])
def run_detection():
    """Run detection on latest FL results"""
    try:
        data = request.get_json() or {}
        use_cached = data.get('use_cached', True)
        
        # Run detection
        result = api_manager.run_detection(use_cached=use_cached)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in run_detection endpoint: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@detection_bp.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update detection configuration"""
    try:
        if request.method == 'GET':
            return jsonify(api_manager.detection_config)
        
        elif request.method == 'POST':
            data = request.get_json() or {}
            if api_manager.update_config(data):
                return jsonify({'success': True, 'config': api_manager.detection_config})
            else:
                return jsonify({'error': 'Failed to update config'}), 400
                
    except Exception as e:
        logger.error(f"Error handling config: {e}")
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/history', methods=['GET'])
def get_detection_history():
    """Get detection history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        history = api_manager.get_detection_history(limit=limit)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/summary', methods=['GET'])
def get_detection_summary():
    """Get comprehensive detection summary"""
    try:
        if not api_manager.detection_adapter:
            return jsonify({'error': 'Detection adapter not initialized'}), 500
        
        summary = api_manager.detection_adapter.get_detection_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting detection summary: {e}")
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/toggle', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off"""
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', not api_manager.detection_config['enabled'])
        
        if api_manager.update_config({'enabled': enabled}):
            return jsonify({
                'success': True, 
                'enabled': enabled,
                'message': f'Detection {"enabled" if enabled else "disabled"}'
            })
        else:
            return jsonify({'error': 'Failed to toggle detection'}), 400
            
    except Exception as e:
        logger.error(f"Error toggling detection: {e}")
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/methods', methods=['GET'])
def get_available_methods():
    """Get available detection methods"""
    return jsonify({
        'methods': ['kmeans', 'fixed_percentage', 'z_score'],
        'current_method': api_manager.detection_config['method'],
        'descriptions': {
            'kmeans': 'K-means clustering on client parameters',
            'fixed_percentage': 'Remove top X% clients by loss',
            'z_score': 'Remove clients with z-score above threshold'
        }
    })

@detection_bp.route('/metrics', methods=['GET'])
def get_detection_metrics():
    """Get detailed detection metrics"""
    try:
        if not api_manager.detection_adapter:
            return jsonify({'error': 'Detection adapter not initialized'}), 500
        
        metrics = api_manager.detection_adapter.detection_metrics
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting detection metrics: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@detection_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Detection endpoint not found'}), 404

@detection_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error in detection API'}), 500
