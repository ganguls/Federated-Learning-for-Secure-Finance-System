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
    """Custom federated averaging strategy for loan prediction with certificate validation"""
    
    def __init__(self, ca_url: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []
        self.client_metrics = {}
        self.ca_url = ca_url or "http://ca:9000"
        self.certificate_validation_enabled = bool(ca_url)
    
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
