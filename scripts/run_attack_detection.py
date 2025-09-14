#!/usr/bin/env python3
"""
Federated Learning Attack Detection Script
=========================================

This script implements a comprehensive attack detection system for federated learning
using Local Differential Privacy (LDP) and K-means clustering to identify malicious clients.

Usage:
    python run_attack_detection.py --dataset /app/dataset/smallLendingClub.csv --epochs 100 --n_train_clients 10 --malicious-percentages 10

Author: FL Defense System
Date: 2025
"""

import argparse
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_defenses.detector import detect_malicious_clients, apply_ldp, eliminate_kmeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalUpdate:
    """
    Local update class for federated learning clients.
    
    This class handles local model training and evaluation for individual clients
    in the federated learning system.
    """
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, is_malicious: bool = False):
        """
        Initialize the local update client.
        
        Args:
            client_id: Unique identifier for the client
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            is_malicious: Whether this client is malicious
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_malicious = is_malicious
        
        # Initialize model
        self.model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
            warm_start=True
        )
        
        # Initialize with zeros
        n_features = X_train.shape[1]
        self.model.coef_ = np.zeros((1, n_features))
        self.model.intercept_ = np.zeros(1)
        self.model.classes_ = np.array([0, 1])
        
        logger.debug(f"Initialized client {client_id} with {len(X_train)} training samples")
    
    def apply_label_flipping_attack(self, y_data: np.ndarray) -> np.ndarray:
        """
        Apply label flipping attack for malicious clients.
        
        Args:
            y_data: Original labels
            
        Returns:
            Modified labels (flipped if malicious, original otherwise)
        """
        if not self.is_malicious:
            return y_data
        
        # Create a copy to avoid modifying original data
        y_attacked = y_data.copy()
        
        # Label flipping attack: flip all labels
        y_attacked = 1 - y_attacked
        
        logger.debug(f"Client {self.client_id}: Applied label flipping attack - {np.sum(y_data != y_attacked)} labels flipped")
        return y_attacked
    
    def update_weights(self, global_weights: List[np.ndarray], fake: bool = False) -> Tuple[List[np.ndarray], float]:
        """
        Update local model weights and return loss.
        
        Args:
            global_weights: Global model weights from server
            fake: If True, use fake data for attack detection
            
        Returns:
            Tuple of (updated_weights, loss)
        """
        # Set global weights
        if global_weights:
            coef, intercept = global_weights
            self.model.coef_ = coef
            self.model.intercept_ = intercept
        
        # Apply attack if malicious and not fake
        if self.is_malicious and not fake:
            y_train_modified = self.apply_label_flipping_attack(self.y_train)
        else:
            y_train_modified = self.y_train
        
        # Train the model
        self.model.fit(self.X_train, y_train_modified)
        
        # Calculate loss
        y_pred = self.model.predict(self.X_test)
        loss = 1.0 - accuracy_score(self.y_test, y_pred)
        
        # Return updated weights and loss
        return [self.model.coef_, self.model.intercept_], loss


class CreditRiskNet:
    """
    Credit Risk Neural Network model.
    
    This is a placeholder class that adapts to the existing LogisticRegression
    implementation in the FL system.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize the Credit Risk Network.
        
        Args:
            input_dim: Input dimension (number of features)
        """
        self.input_dim = input_dim
        self.model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
            warm_start=True
        )
        
        # Initialize with zeros
        self.model.coef_ = np.zeros((1, input_dim))
        self.model.intercept_ = np.zeros(1)
        self.model.classes_ = np.array([0, 1])
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights."""
        return [self.model.coef_, self.model.intercept_]
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights."""
        coef, intercept = weights
        self.model.coef_ = coef
        self.model.intercept_ = intercept
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """
        Evaluate the model on given data.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        y_pred = self.model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        loss = 1.0 - accuracy
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss": loss
        }
        
        return loss, metrics


def preprocess_lending_club_data(file_path: str, n_clients: int = 10) -> Tuple[List[LocalUpdate], np.ndarray, np.ndarray]:
    """
    Preprocess Lending Club data for federated learning.
    
    Args:
        file_path: Path to the CSV file
        n_clients: Number of clients to create
        
    Returns:
        Tuple of (clients_list, global_X_test, global_y_test)
    """
    logger.info(f"Loading data from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path, low_memory=False)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Basic preprocessing
    # Drop irrelevant columns
    irrelevant_cols = ['id', 'member_id', 'url', 'desc', 'title', 'emp_title', 'zip_code']
    df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns])
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    
    # Create target variable
    df['loan_status_binary'] = (df['loan_status'] == 'Fully Paid').astype(int)
    df = df.drop(columns=['loan_status'])
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    ordinal_cols = ['emp_length', 'sub_grade']
    for col in ordinal_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # One-hot encoding for nominal columns
    nominal_cols = ['grade', 'home_ownership', 'purpose', 'addr_state']
    nominal_cols = [col for col in nominal_cols if col in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Remove columns with too many unique values or low variance
    for col in df.columns:
        if col != 'loan_status_binary':
            if df[col].nunique() > 100 or df[col].var() < 1e-6:
                df = df.drop(columns=[col])
    
    # Prepare features and target
    X = df.drop(columns=['loan_status_binary']).values
    y = df['loan_status_binary'].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Create clients with distributed data
    clients = []
    samples_per_client = len(X_train) // n_clients
    
    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < n_clients - 1 else len(X_train)
        
        client_X_train = X_train[start_idx:end_idx]
        client_y_train = y_train[start_idx:end_idx]
        
        # Create client
        client = LocalUpdate(
            client_id=i,
            X_train=client_X_train,
            y_train=client_y_train,
            X_test=X_test,
            y_test=y_test,
            is_malicious=False  # Will be set later
        )
        clients.append(client)
    
    logger.info(f"Created {len(clients)} clients with {samples_per_client} samples each")
    return clients, X_test, y_test


def run_attack_detection_experiment(
    dataset_path: str,
    epochs: int = 100,
    n_train_clients: int = 10,
    n_total_clients: int = 10,
    malicious_percentages: List[float] = [0.0, 10.0],
    output_dir: str = "/app/results",
    epsilon: float = 1.0
) -> Dict:
    """
    Run the attack detection experiment.
    
    Args:
        dataset_path: Path to the dataset
        epochs: Number of training epochs
        n_train_clients: Number of clients participating in training
        n_total_clients: Total number of clients
        malicious_percentages: List of malicious client percentages to test
        output_dir: Output directory for results
        epsilon: LDP privacy parameter
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting attack detection experiment")
    logger.info(f"Parameters: epochs={epochs}, n_train_clients={n_train_clients}, "
               f"n_total_clients={n_total_clients}, malicious_percentages={malicious_percentages}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    clients, X_test, y_test = preprocess_lending_club_data(dataset_path, n_total_clients)
    
    # Initialize global model
    input_dim = clients[0].X_train.shape[1]
    global_model = CreditRiskNet(input_dim)
    
    # Results storage
    all_results = {}
    
    for malicious_pct in malicious_percentages:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment with {malicious_pct}% malicious clients")
        logger.info(f"{'='*50}")
        
        # Set malicious clients
        n_malicious = int(n_total_clients * malicious_pct / 100)
        true_malicious = list(range(n_malicious))
        
        for i in range(n_total_clients):
            clients[i].is_malicious = i < n_malicious
        
        logger.info(f"Set {n_malicious} clients as malicious: {true_malicious}")
        
        # Initialize results for this experiment
        experiment_results = {
            "malicious_percentage": malicious_pct,
            "true_malicious": true_malicious,
            "epoch_results": [],
            "detection_results": [],
            "final_metrics": {}
        }
        
        # Training loop
        for epoch in tqdm(range(epochs), desc=f"Training (malicious_pct={malicious_pct}%)"):
            # Select clients for this round
            selected_clients = np.random.choice(n_total_clients, n_train_clients, replace=False)
            
            # Get fake updates for attack detection
            fake_losses = []
            fake_weights = []
            
            for client_idx in selected_clients:
                client = clients[client_idx]
                weights, loss = client.update_weights(global_model.get_weights(), fake=True)
                fake_losses.append(loss)
                fake_weights.append(weights)
            
            # Apply attack detection
            detection_results = detect_malicious_clients(
                fake_losses, 
                true_malicious=[i for i in selected_clients if i in true_malicious],
                epsilon=epsilon
            )
            
            # Remove detected malicious clients from aggregation
            detected_malicious = detection_results.get("detected_malicious", [])
            benign_clients = [i for i in selected_clients if i not in detected_malicious]
            
            logger.debug(f"Epoch {epoch}: Detected {len(detected_malicious)} malicious clients: {detected_malicious}")
            
            # Aggregate updates from benign clients only
            if benign_clients:
                # Get real updates from benign clients
                benign_weights = []
                for client_idx in benign_clients:
                    client = clients[client_idx]
                    weights, _ = client.update_weights(global_model.get_weights(), fake=False)
                    benign_weights.append(weights)
                
                # Simple averaging (FedAvg)
                if benign_weights:
                    avg_coef = np.mean([w[0] for w in benign_weights], axis=0)
                    avg_intercept = np.mean([w[1] for w in benign_weights], axis=0)
                    global_model.set_weights([avg_coef, avg_intercept])
            
            # Evaluate global model
            test_loss, test_metrics = global_model.evaluate(X_test, y_test)
            
            # Store epoch results
            epoch_result = {
                "epoch": epoch,
                "test_loss": test_loss,
                "test_accuracy": test_metrics["accuracy"],
                "detected_malicious": detected_malicious,
                "detection_accuracy": detection_results.get("accuracy", 0.0),
                "detection_f1": detection_results.get("f1_score", 0.0)
            }
            experiment_results["epoch_results"].append(epoch_result)
            
            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch:3d}: Test Acc={test_metrics['accuracy']:.4f}, "
                           f"Test Loss={test_loss:.4f}, Detected Malicious={len(detected_malicious)}, "
                           f"Detection Acc={detection_results.get('accuracy', 0.0):.4f}")
        
        # Calculate final metrics
        final_epoch = experiment_results["epoch_results"][-1]
        experiment_results["final_metrics"] = {
            "final_test_accuracy": final_epoch["test_accuracy"],
            "final_test_loss": final_epoch["test_loss"],
            "avg_detection_accuracy": np.mean([r["detection_accuracy"] for r in experiment_results["epoch_results"]]),
            "avg_detection_f1": np.mean([r["detection_f1"] for r in experiment_results["epoch_results"]]),
            "total_detections": sum(len(r["detected_malicious"]) for r in experiment_results["epoch_results"])
        }
        
        all_results[f"malicious_{malicious_pct}pct"] = experiment_results
        
        logger.info(f"Experiment completed for {malicious_pct}% malicious clients")
        logger.info(f"Final Test Accuracy: {final_epoch['test_accuracy']:.4f}")
        logger.info(f"Average Detection Accuracy: {experiment_results['final_metrics']['avg_detection_accuracy']:.4f}")
        logger.info(f"Average Detection F1: {experiment_results['final_metrics']['avg_detection_f1']:.4f}")
    
    # Save results
    results_file = os.path.join(output_dir, "lending_club_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create plots
    create_plots(all_results, output_dir)
    
    # Save metadata
    metadata = {
        "experiment_date": datetime.now().isoformat(),
        "parameters": {
            "epochs": epochs,
            "n_train_clients": n_train_clients,
            "n_total_clients": n_total_clients,
            "malicious_percentages": malicious_percentages,
            "epsilon": epsilon
        },
        "results_file": results_file
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    return all_results


def create_plots(results: Dict, output_dir: str):
    """
    Create visualization plots for the experiment results.
    
    Args:
        results: Experiment results dictionary
        output_dir: Output directory for plots
    """
    logger.info("Creating visualization plots")
    
    # Plot 1: Test Accuracy over Epochs
    plt.figure(figsize=(12, 8))
    
    for exp_name, exp_results in results.items():
        epochs = [r["epoch"] for r in exp_results["epoch_results"]]
        test_acc = [r["test_accuracy"] for r in exp_results["epoch_results"]]
        plt.plot(epochs, test_acc, label=f"Test Accuracy ({exp_name})", linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Over Training Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    accuracy_plot = os.path.join(output_dir, "test_accuracy.png")
    plt.savefig(accuracy_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Detection Accuracy over Epochs
    plt.figure(figsize=(12, 8))
    
    for exp_name, exp_results in results.items():
        epochs = [r["epoch"] for r in exp_results["epoch_results"]]
        det_acc = [r["detection_accuracy"] for r in exp_results["epoch_results"]]
        plt.plot(epochs, det_acc, label=f"Detection Accuracy ({exp_name})", linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Detection Accuracy")
    plt.title("Attack Detection Accuracy Over Training Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    detection_plot = os.path.join(output_dir, "attack_detection.png")
    plt.savefig(detection_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Detection F1 Score over Epochs
    plt.figure(figsize=(12, 8))
    
    for exp_name, exp_results in results.items():
        epochs = [r["epoch"] for r in exp_results["epoch_results"]]
        det_f1 = [r["detection_f1"] for r in exp_results["epoch_results"]]
        plt.plot(epochs, det_f1, label=f"Detection F1 ({exp_name})", linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Detection F1 Score")
    plt.title("Attack Detection F1 Score Over Training Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    f1_plot = os.path.join(output_dir, "detection_f1.png")
    plt.savefig(f1_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Federated Learning Attack Detection")
    
    parser.add_argument("--dataset", type=str, default="/app/dataset/smallLendingClub.csv",
                       help="Path to the dataset CSV file")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--n_train_clients", type=int, default=10,
                       help="Number of clients participating in training")
    parser.add_argument("--n_total_clients", type=int, default=10,
                       help="Total number of clients")
    parser.add_argument("--malicious-percentages", type=float, nargs='+', default=[0.0, 10.0],
                       help="List of malicious client percentages to test")
    parser.add_argument("--output-dir", type=str, default="/app/results",
                       help="Output directory for results")
    parser.add_argument("--epsilon", type=float, default=1.0,
                       help="LDP privacy parameter")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    if args.n_train_clients > args.n_total_clients:
        logger.error("n_train_clients cannot be greater than n_total_clients")
        sys.exit(1)
    
    if any(pct < 0 or pct > 100 for pct in args.malicious_percentages):
        logger.error("Malicious percentages must be between 0 and 100")
        sys.exit(1)
    
    # Run experiment
    try:
        results = run_attack_detection_experiment(
            dataset_path=args.dataset,
            epochs=args.epochs,
            n_train_clients=args.n_train_clients,
            n_total_clients=args.n_total_clients,
            malicious_percentages=args.malicious_percentages,
            output_dir=args.output_dir,
            epsilon=args.epsilon
        )
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)
        
        for exp_name, exp_results in results.items():
            metrics = exp_results["final_metrics"]
            logger.info(f"\n{exp_name}:")
            logger.info(f"  Final Test Accuracy: {metrics['final_test_accuracy']:.4f}")
            logger.info(f"  Average Detection Accuracy: {metrics['avg_detection_accuracy']:.4f}")
            logger.info(f"  Average Detection F1: {metrics['avg_detection_f1']:.4f}")
            logger.info(f"  Total Detections: {metrics['total_detections']}")
        
        logger.info(f"\nResults saved to: {args.output_dir}")
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()



