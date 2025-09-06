import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import json

class LoanServerStrategy(FedAvg):
    """Custom federated averaging strategy for loan prediction"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []
        self.client_metrics = {}
    
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
            accuracies.append(accuracy)
            
            client_id = metrics.get("client_id", "unknown") if isinstance(metrics, dict) else "unknown"
            round_metrics["client_metrics"][client_id] = {
                "accuracy": accuracy,
                "precision": metrics.get("precision", 0.0) if isinstance(metrics, dict) else 0.0,
                "recall": metrics.get("recall", 0.0) if isinstance(metrics, dict) else 0.0,
                "f1_score": metrics.get("f1_score", 0.0) if isinstance(metrics, dict) else 0.0,
                "loss": loss,
                "num_examples": num_examples
            }
        
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            round_metrics["avg_accuracy"] = avg_accuracy
        
        self.round_metrics.append(round_metrics)
        
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
    
    # Server configuration
    config = ServerConfig(
        num_rounds=5,
    )
    
    # Strategy configuration
    strategy = LoanServerStrategy(
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
    )
    
    # Start the server
    print("Server configuration:")
    print(f"  - Number of rounds: {config.num_rounds}")
    print(f"  - Min fit clients: {strategy.min_fit_clients}")
    print(f"  - Min evaluate clients: {strategy.min_evaluate_clients}")
    print(f"  - Min available clients: {strategy.min_available_clients}")
    print("=" * 60)
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
