#!/usr/bin/env python3
"""
Enhanced LoanServerStrategy with Advanced Data Poisoning Detection
Docker-compatible version for production deployment
"""

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import ssl
import requests
from typing import Dict, List, Tuple, Optional
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import json
from pathlib import Path
import logging
import time
import sys

# Add detection system to path
sys.path.append('/app/data_poisoning_detection')

# Import detection adapter
try:
    from detection_adapter import FLDetectionAdapter, create_detection_adapter
except ImportError as e:
    logging.warning(f"Detection adapter not available: {e}")
    # Fallback implementation
    class FLDetectionAdapter:
        def __init__(self, *args, **kwargs):
            pass
        def detect_malicious_clients(self, *args, **kwargs):
            return [], {}
        def get_detection_summary(self):
            return {}

logger = logging.getLogger(__name__)

class EnhancedLoanServerStrategy(FedAvg):
    """
    Enhanced federated averaging strategy with comprehensive data poisoning detection
    Docker-compatible version
    """
    
    def __init__(self, 
                 ca_url: str = None,
                 detection_config: Optional[Dict] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Original server components
        self.round_metrics = []
        self.client_metrics = {}
        self.ca_url = ca_url or "http://ca:9000"
        self.certificate_validation_enabled = bool(ca_url)
        
        # Enhanced detection components
        self.detection_enabled = os.getenv('DETECTION_ENABLED', 'true').lower() == 'true'
        self.detection_config = detection_config or {
            'enabled': self.detection_enabled,
            'method': os.getenv('DETECTION_METHOD', 'kmeans'),
            'ldp_epsilon': float(os.getenv('LDP_EPSILON', '1.0')),
            'ldp_sensitivity': 0.001,
            'input_dim': 20,
            'defense_threshold': 0.3
        }
        
        # Initialize detection adapter
        try:
            self.detection_adapter = create_detection_adapter(
                input_dim=self.detection_config['input_dim'],
                detection_method=self.detection_config['method'],
                ldp_epsilon=self.detection_config['ldp_epsilon']
            )
            logger.info("Detection adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detection adapter: {e}")
            self.detection_adapter = FLDetectionAdapter()
        
        # Detection state
        self.malicious_clients = set()
        self.detection_history = []
        self.client_update_history = {}
        self.defense_threshold = self.detection_config.get('defense_threshold', 0.3)
        
        # Performance tracking
        self.detection_times = []
        self.aggregation_times = []
        
        logger.info(f"Enhanced server strategy initialized: detection={self.detection_enabled}")
        logger.info(f"Detection method: {self.detection_config.get('method', 'kmeans')}")
        logger.info(f"LDP epsilon: {self.detection_config.get('ldp_epsilon', 1.0)}")
    
    def validate_client_certificate(self, client_id: str) -> bool:
        """Validate client certificate with CA service"""
        if not self.certificate_validation_enabled:
            return True
        
        try:
            response = requests.get(
                f"{self.ca_url}/certificates/{client_id}/validate",
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("valid", False)
            else:
                logger.warning(f"Certificate validation failed for client {client_id}: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error validating certificate for client {client_id}: {e}")
            return False
    
    def detect_malicious_clients_enhanced(self, results: List[Tuple]) -> Tuple[set, Dict]:
        """
        Enhanced malicious client detection using PyTorch-based detection system
        
        Args:
            results: List of Flower client results (parameters, num_examples, metrics)
            
        Returns:
            Tuple of (detected_malicious_set, detection_metrics)
        """
        if not self.detection_enabled or not results or len(results) < 2:
            return set(), {}
        
        try:
            start_time = time.time()
            
            # Extract known malicious clients from metrics (for evaluation)
            true_malicious = []
            for result in results:
                if len(result) >= 3:
                    metrics = result[2]
                    if isinstance(metrics, dict) and metrics.get("is_malicious", False):
                        true_malicious.append(str(metrics.get("client_id", "unknown")))
            
            # Run detection using adapter
            detected_malicious, detection_metrics = self.detection_adapter.detect_malicious_clients(
                client_results=results,
                true_malicious=true_malicious if true_malicious else None
            )
            
            # Convert to set for compatibility
            detected_set = set(detected_malicious)
            
            # Update detection history
            detection_record = {
                'round': len(self.detection_history),
                'timestamp': time.time(),
                'total_clients': len(results),
                'detected_malicious': detected_malicious,
                'true_malicious': true_malicious,
                'detection_metrics': detection_metrics,
                'detection_time': time.time() - start_time
            }
            self.detection_history.append(detection_record)
            self.detection_times.append(time.time() - start_time)
            
            # Log detection results
            logger.info(f"Enhanced detection completed in {time.time() - start_time:.3f}s")
            logger.info(f"Detected {len(detected_malicious)} malicious clients: {detected_malicious}")
            if detection_metrics:
                logger.info(f"Detection accuracy: {detection_metrics.get('accuracy', 0):.3f}")
                logger.info(f"Detection precision: {detection_metrics.get('precision', 0):.3f}")
            
            return detected_set, detection_metrics
            
        except Exception as e:
            logger.error(f"Enhanced detection failed: {e}")
            return set(), {}
    
    def detect_malicious_clients_legacy(self, results: List[Tuple]) -> set:
        """
        Legacy detection method (original K-means clustering)
        Kept for backward compatibility and fallback
        """
        if not results or len(results) < 3:
            return set()
        
        try:
            # Extract client updates and metrics
            client_updates = []
            client_ids = []
            
            for result in results:
                if len(result) >= 3:
                    parameters, metrics, num_examples = result[0], result[1], result[2]
                    
                    # Extract client info from metrics
                    if isinstance(metrics, dict):
                        client_id = metrics.get("client_id", "unknown")
                        is_malicious = metrics.get("is_malicious", False)
                        labels_flipped = metrics.get("labels_flipped", 0)
                        
                        # Store client information
                        client_ids.append(client_id)
                        
                        # Track malicious behavior
                        if is_malicious or labels_flipped > 0:
                            logger.info(f"Legacy detection: Client {client_id} flagged as malicious")
                            self.malicious_clients.add(client_id)
                        
                        # Flatten parameters for clustering
                        if isinstance(parameters, (list, tuple)) and len(parameters) >= 2:
                            coef, intercept = parameters[0], parameters[1]
                            if hasattr(coef, 'flatten'):
                                flat_params = np.concatenate([coef.flatten(), intercept.flatten()])
                            else:
                                flat_params = np.concatenate([np.array(coef).flatten(), np.array(intercept).flatten()])
                            client_updates.append(flat_params)
            
            # Perform K-means clustering if we have enough clients
            if len(client_updates) >= 3:
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    # Normalize the updates
                    scaler = StandardScaler()
                    normalized_updates = scaler.fit_transform(client_updates)
                    
                    # Perform clustering (2 clusters: normal vs malicious)
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(normalized_updates)
                    
                    # Identify the smaller cluster as potentially malicious
                    cluster_0_count = np.sum(cluster_labels == 0)
                    cluster_1_count = np.sum(cluster_labels == 1)
                    
                    malicious_cluster = 0 if cluster_0_count < cluster_1_count else 1
                    
                    # Mark clients in the malicious cluster
                    for i, (client_id, cluster_label) in enumerate(zip(client_ids, cluster_labels)):
                        if cluster_label == malicious_cluster and client_id != "unknown":
                            logger.info(f"Legacy clustering detected potential malicious client: {client_id}")
                            self.malicious_clients.add(client_id)
                    
                except Exception as e:
                    logger.error(f"Legacy clustering defense failed: {e}")
            
            return self.malicious_clients
            
        except Exception as e:
            logger.error(f"Legacy detection failed: {e}")
            return set()
    
    def apply_defense_mechanism(self, results: List[Tuple], detected_malicious: set) -> List[Tuple]:
        """
        Apply defense mechanism to filter out malicious clients
        
        Args:
            results: List of Flower client results
            detected_malicious: Set of detected malicious client IDs
            
        Returns:
            Filtered list of client results
        """
        if not detected_malicious:
            return results
        
        filtered_results = []
        for result in results:
            if len(result) >= 3:
                metrics = result[2]
                if isinstance(metrics, dict):
                    client_id = str(metrics.get("client_id", "unknown"))
                    if client_id not in detected_malicious:
                        filtered_results.append(result)
                    else:
                        logger.info(f"Filtered out malicious client: {client_id}")
                else:
                    # Keep result if we can't determine client ID
                    filtered_results.append(result)
            else:
                # Keep result if format is unexpected
                filtered_results.append(result)
        
        return filtered_results
    
    def aggregate_fit(self, server_round: int, results: List[Tuple], failures: List) -> Tuple[Optional[Dict], Dict]:
        """
        Enhanced aggregation with comprehensive detection
        
        Args:
            server_round: Current round number
            results: List of client results
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        logger.info(f"🔄 Round {server_round}: Aggregating {len(results)} client updates...")
        
        start_time = time.time()
        
        # Step 1: Detect malicious clients
        if self.detection_enabled:
            detected_malicious, detection_metrics = self.detect_malicious_clients_enhanced(results)
            if detected_malicious:
                logger.info(f"🚨 Enhanced detection: {len(detected_malicious)} malicious clients detected")
                self.malicious_clients.update(detected_malicious)
        else:
            # Fallback to legacy detection
            detected_malicious = self.detect_malicious_clients_legacy(results)
            detection_metrics = {}
        
        # Step 2: Apply defense mechanism
        filtered_results = self.apply_defense_mechanism(results, detected_malicious)
        
        if len(filtered_results) != len(results):
            logger.info(f"🛡️ Defense applied: Using {len(filtered_results)}/{len(results)} clients for aggregation")
        
        # Step 3: Perform standard FedAvg aggregation
        aggregation_start = time.time()
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, filtered_results, failures)
        self.aggregation_times.append(time.time() - aggregation_start)
        
        # Step 4: Update round metrics
        self._update_round_metrics(server_round, results, filtered_results, detection_metrics)
        
        # Step 5: Save metrics
        self.save_metrics()
        
        total_time = time.time() - start_time
        logger.info(f"Round {server_round} completed in {total_time:.3f}s (detection: {self.detection_times[-1] if self.detection_times else 0:.3f}s)")
        
        return aggregated_params, aggregated_metrics
    
    def _update_round_metrics(self, 
                            server_round: int, 
                            original_results: List[Tuple], 
                            filtered_results: List[Tuple],
                            detection_metrics: Dict):
        """Update round metrics with detection information"""
        try:
            # Calculate accuracy metrics
            accuracies = []
            for result in filtered_results:
                if len(result) >= 3:
                    metrics = result[2]
                    if isinstance(metrics, dict) and "accuracy" in metrics:
                        accuracies.append(float(metrics["accuracy"]))
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            # Create round metrics
            round_metrics = {
                "round": server_round,
                "total_clients": len(original_results),
                "filtered_clients": len(filtered_results),
                "malicious_clients": len(original_results) - len(filtered_results),
                "avg_accuracy": avg_accuracy,
                "detection_metrics": detection_metrics,
                "timestamp": time.time()
            }
            
            self.round_metrics.append(round_metrics)
            
            logger.info(f"Round {server_round} - Average Accuracy: {avg_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating round metrics: {e}")
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple], failures: List) -> Optional[float]:
        """
        Enhanced evaluation aggregation with detection metrics
        """
        if not results:
            return None
        
        try:
            # Calculate average accuracy
            accuracies = []
            for result in results:
                if len(result) >= 3:
                    metrics = result[2]
                    if isinstance(metrics, dict) and "accuracy" in metrics:
                        accuracies.append(float(metrics["accuracy"]))
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            # Update client metrics
            for result in results:
                if len(result) >= 3:
                    metrics = result[2]
                    if isinstance(metrics, dict):
                        client_id = metrics.get("client_id", "unknown")
                        self.client_metrics[client_id] = metrics
            
            # Create serializable metrics
            serializable_metrics = {
                "round": server_round,
                "avg_accuracy": avg_accuracy,
                "client_metrics": self.client_metrics
            }
            
            self.round_metrics.append(serializable_metrics)
            self.save_metrics()
            
            logger.info(f"Round {server_round} - Average Accuracy: {avg_accuracy:.4f}")
            return avg_accuracy
            
        except Exception as e:
            logger.error(f"Error in aggregate_evaluate: {e}")
            return None
    
    def save_metrics(self):
        """Save training metrics to file"""
        metrics_file = "/app/detection_results/enhanced_training_metrics.json"
        try:
            # Ensure all data is JSON serializable
            serializable_metrics = []
            for metric in self.round_metrics:
                if isinstance(metric, dict):
                    clean_metric = {}
                    for key, value in metric.items():
                        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            clean_metric[key] = value
                        else:
                            clean_metric[key] = str(value)
                    serializable_metrics.append(clean_metric)
                else:
                    serializable_metrics.append(str(metric))
            
            with open(metrics_file, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Enhanced metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Warning: Could not save enhanced metrics: {e}")
    
    def get_detection_summary(self) -> Dict:
        """Get comprehensive detection summary"""
        return {
            'detection_enabled': self.detection_enabled,
            'detection_config': self.detection_config,
            'malicious_clients': list(self.malicious_clients),
            'detection_history': self.detection_history[-10:],  # Last 10 rounds
            'detection_adapter_summary': self.detection_adapter.get_detection_summary(),
            'performance_metrics': {
                'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
                'avg_aggregation_time': np.mean(self.aggregation_times) if self.aggregation_times else 0,
                'total_rounds': len(self.round_metrics)
            }
        }
    
    def run_detection_on_demand(self, results: List[Tuple]) -> Dict:
        """
        Run detection on demand (for dashboard button)
        
        Args:
            results: List of client results to analyze
            
        Returns:
            Detection results dictionary
        """
        try:
            if not results:
                return {'error': 'No results provided for detection'}
            
            # Run detection
            detected_malicious, detection_metrics = self.detect_malicious_clients_enhanced(results)
            
            # Get detection summary
            detection_summary = self.get_detection_summary()
            
            return {
                'success': True,
                'detected_malicious': list(detected_malicious),
                'detection_metrics': detection_metrics,
                'detection_summary': detection_summary,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in on-demand detection: {e}")
            return {'error': str(e), 'success': False}

def main():
    """Start the Enhanced Flower federated learning server with detection"""
    print("Starting Enhanced Federated Learning Server with Detection...")
    print("=" * 70)
    
    # Get configuration from environment variables
    num_rounds = int(os.getenv("NUM_ROUNDS", "10"))
    min_clients = int(os.getenv("MIN_CLIENTS", "10"))
    server_port = int(os.getenv("SERVER_PORT", "8080"))
    ca_url = os.getenv("CA_URL", "http://ca:9000")
    enable_certificates = os.getenv("ENABLE_CERTIFICATES", "true").lower() == "true"
    
    # Detection configuration
    detection_config = {
        'enabled': os.getenv('DETECTION_ENABLED', 'true').lower() == 'true',
        'method': os.getenv('DETECTION_METHOD', 'kmeans'),
        'ldp_epsilon': float(os.getenv('LDP_EPSILON', '1.0')),
        'ldp_sensitivity': 0.001,
        'input_dim': 20,
        'defense_threshold': 0.3
    }
    
    # Server configuration
    config = ServerConfig(
        num_rounds=num_rounds,
    )
    
    # Strategy configuration with detection integration
    strategy = EnhancedLoanServerStrategy(
        ca_url=ca_url if enable_certificates else None,
        detection_config=detection_config,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
    )
    
    # Start the server
    print("Enhanced Server configuration:")
    print(f"  - Number of rounds: {config.num_rounds}")
    print(f"  - Min fit clients: {strategy.min_fit_clients}")
    print(f"  - Min evaluate clients: {strategy.min_evaluate_clients}")
    print(f"  - Min available clients: {strategy.min_available_clients}")
    print(f"  - Certificate validation: {'Enabled' if enable_certificates else 'Disabled'}")
    print(f"  - Detection enabled: {strategy.detection_enabled}")
    print(f"  - Detection method: {strategy.detection_config['method']}")
    print(f"  - LDP epsilon: {strategy.detection_config['ldp_epsilon']}")
    if enable_certificates:
        print(f"  - CA Service URL: {ca_url}")
    print("=" * 70)
    
    # Generate certificates for all expected clients if CA is enabled
    if enable_certificates:
        print("Generating certificates for clients...")
        try:
            for client_id in range(1, min_clients + 1):
                response = requests.post(
                    f"{ca_url}/certificates/generate",
                    json={"client_id": str(client_id), "permissions": "standard"},
                    timeout=10
                )
                if response.status_code == 200:
                    print(f"  ✓ Certificate generated for client {client_id}")
                else:
                    print(f"  ⚠ Failed to generate certificate for client {client_id}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not connect to CA service: {e}")
            print("  Continuing without certificate pre-generation...")
    
    # Start the enhanced server
    fl.server.start_server(
        server_address=f"0.0.0.0:{server_port}",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()