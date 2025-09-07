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

class LoanServerStrategy(FedAvg):
    """Custom federated averaging strategy with malicious client detection and defense"""
    
    def __init__(self, ca_url: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []
        self.client_metrics = {}
        self.ca_url = ca_url or "http://ca:9000"
        self.certificate_validation_enabled = bool(ca_url)
        
        # Defense mechanism components
        self.malicious_clients = set()  # Detected malicious clients
        self.client_update_history = {}  # Track client update patterns
        self.defense_threshold = 0.3  # Threshold for detecting anomalies
        self.clustering_enabled = True  # Enable K-means clustering defense
    
    def validate_client_certificate(self, client_id: str) -> bool:
        """Validate client certificate with CA service"""
        if not self.certificate_validation_enabled:
            return True  # Skip validation if CA not configured
        
        try:
            response = requests.get(
                f"{self.ca_url}/certificates/{client_id}/validate",
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("valid", False)
            else:
                print(f"Certificate validation failed for client {client_id}: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"Error validating certificate for client {client_id}: {e}")
            return False
    
    def detect_malicious_clients(self, results):
        """Detect malicious clients using clustering and statistical analysis"""
        if not results or len(results) < 3:
            return set()
        
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
                        print(f"🚨 Detected malicious behavior from Client {client_id}: {labels_flipped} labels flipped")
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
        if self.clustering_enabled and len(client_updates) >= 3:
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
                        print(f"🔍 Clustering detected potential malicious client: {client_id}")
                        self.malicious_clients.add(client_id)
                
            except Exception as e:
                print(f"⚠️ Clustering defense failed: {e}")
        
        return self.malicious_clients
    
    def apply_defense_mechanism(self, results):
        """Apply defense mechanism to filter out malicious clients"""
        if not self.malicious_clients:
            return results
        
        # Filter out results from detected malicious clients
        filtered_results = []
        excluded_count = 0
        
        for result in results:
            if len(result) >= 2 and isinstance(result[1], dict):
                client_id = result[1].get("client_id", "unknown")
                if client_id not in self.malicious_clients:
                    filtered_results.append(result)
                else:
                    excluded_count += 1
                    print(f"🛡️ Excluded malicious client {client_id} from aggregation")
            else:
                # Include results without client identification
                filtered_results.append(result)
        
        if excluded_count > 0:
            print(f"🛡️ Defense mechanism: Excluded {excluded_count} malicious clients from aggregation")
        
        return filtered_results
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Optional[float]:
        """Aggregate evaluation metrics from all clients"""
        if not results:
            return None
        
        # Aggregate accuracy - results is a list of tuples (loss, metrics, num_examples)
        accuracies = []
        round_metrics = {
            "round": server_round,
            "avg_accuracy": 0.0,
            "client_metrics": {}
        }
        
        for result in results:
            # result is a tuple (loss, metrics, num_examples)
            if len(result) == 3:
                loss, metrics, num_examples = result
            elif len(result) == 2:
                loss, metrics = result
                num_examples = 0
            else:
                continue
                
            # metrics is a dictionary, extract values safely
            accuracy = metrics.get("accuracy", 0.0) if isinstance(metrics, dict) else 0.0
            client_id = metrics.get("client_id", "unknown") if isinstance(metrics, dict) else "unknown"
            
            # Validate client certificate before processing metrics
            if self.certificate_validation_enabled and client_id != "unknown":
                if not self.validate_client_certificate(client_id):
                    print(f"Rejecting metrics from client {client_id}: Invalid certificate")
                    continue
            
            accuracies.append(accuracy)
            round_metrics["client_metrics"][client_id] = {
                "accuracy": accuracy,
                "precision": metrics.get("precision", 0.0) if isinstance(metrics, dict) else 0.0,
                "recall": metrics.get("recall", 0.0) if isinstance(metrics, dict) else 0.0,
                "f1_score": metrics.get("f1_score", 0.0) if isinstance(metrics, dict) else 0.0,
                "loss": loss,
                "num_examples": num_examples,
                "certificate_valid": True
            }
        
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            round_metrics["avg_accuracy"] = avg_accuracy
        
        # Only save serializable data
        serializable_metrics = {
            "round": server_round,
            "avg_accuracy": avg_accuracy if accuracies else 0.0,
            "client_metrics": round_metrics["client_metrics"]
        }
        
        self.round_metrics.append(serializable_metrics)
        
        # Save metrics to file
        self.save_metrics()
        
        print(f"Round {server_round} - Average Accuracy: {avg_accuracy:.4f}")
        return avg_accuracy
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights with defense mechanism"""
        print(f"\n🔄 Round {server_round}: Aggregating {len(results)} client updates...")
        
        # Detect malicious clients before aggregation
        detected_malicious = self.detect_malicious_clients(results)
        if detected_malicious:
            print(f"🚨 Detected {len(detected_malicious)} malicious clients: {detected_malicious}")
        
        # Apply defense mechanism to filter malicious clients
        filtered_results = self.apply_defense_mechanism(results)
        
        if len(filtered_results) != len(results):
            print(f"🛡️ Defense applied: Using {len(filtered_results)}/{len(results)} clients for aggregation")
        
        # Call parent aggregation with filtered results
        return super().aggregate_fit(server_round, filtered_results, failures)
    
    def save_metrics(self):
        """Save training metrics to file"""
        metrics_file = "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.round_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")

def main():
    """Start the Flower federated learning server"""
    print("Starting Federated Learning Server for Loan Prediction...")
    print("=" * 60)
    
    # Get configuration from environment variables
    num_rounds = int(os.getenv("NUM_ROUNDS", "5"))
    min_clients = int(os.getenv("MIN_CLIENTS", "10"))
    server_port = int(os.getenv("SERVER_PORT", "8080"))
    ca_url = os.getenv("CA_URL", "http://ca:9000")
    enable_certificates = os.getenv("ENABLE_CERTIFICATES", "true").lower() == "true"
    
    # Server configuration
    config = ServerConfig(
        num_rounds=num_rounds,
    )
    
    # Strategy configuration with CA integration
    strategy = LoanServerStrategy(
        ca_url=ca_url if enable_certificates else None,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
    )
    
    # Start the server
    print("Server configuration:")
    print(f"  - Number of rounds: {config.num_rounds}")
    print(f"  - Min fit clients: {strategy.min_fit_clients}")
    print(f"  - Min evaluate clients: {strategy.min_evaluate_clients}")
    print(f"  - Min available clients: {strategy.min_available_clients}")
    print(f"  - Certificate validation: {'Enabled' if enable_certificates else 'Disabled'}")
    if enable_certificates:
        print(f"  - CA Service URL: {ca_url}")
    print("=" * 60)
    
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
    
    fl.server.start_server(
        server_address=f"0.0.0.0:{server_port}",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
