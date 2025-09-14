"""
Federated Learning Data Poisoning Detection Module
Main module for integrating detection into existing FL systems
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import copy
import warnings
warnings.filterwarnings("ignore")

from detection_utils import (
    detect_malicious_clients, 
    apply_ldp, 
    calculate_accuracy, 
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    average_weights
)
from tabular_model import TabularNet, create_tabular_model
from tabular_data_utils import TabularDataset, create_client_dataloader

class FederatedDetectionSystem:
    """
    Main class for federated learning data poisoning detection
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int = 2,
                 model_type: str = 'credit_risk',
                 detection_method: str = 'kmeans',
                 ldp_epsilon: float = 1.0,
                 ldp_sensitivity: float = 0.0001,
                 device: str = 'auto'):
        """
        Initialize the detection system
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            model_type: Type of model ('credit_risk', 'default', 'custom')
            detection_method: Detection method ('kmeans', 'fixed_percentage', 'z_score')
            ldp_epsilon: LDP privacy parameter
            ldp_sensitivity: LDP sensitivity parameter
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_type = model_type
        self.detection_method = detection_method
        self.ldp_epsilon = ldp_epsilon
        self.ldp_sensitivity = ldp_sensitivity
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize global model
        self.global_model = create_tabular_model(
            input_dim=input_dim,
            num_classes=num_classes,
            model_type=model_type
        ).to(self.device)
        
        # Detection history
        self.detection_history = []
        self.performance_history = []
        
        print(f"Detection system initialized on {self.device}")
        print(f"Model: {model_type}, Detection: {detection_method}")
    
    def train_client_fake(self, client_data: List[Tuple], client_id: int) -> Tuple[Dict, float]:
        """
        Perform fake training on client data (for detection)
        
        Args:
            client_data: List of (features, labels) tuples
            client_id: Client identifier
        
        Returns:
            weights: Model state dictionary
            loss: Average loss value
        """
        # Create a copy of the global model
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        
        # Set up optimizer and criterion
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.001)
        criterion = nn.NLLLoss()
        
        # Calculate loss without updating weights
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in client_data:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                log_probs = local_model(features)
                loss = criterion(log_probs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return local_model.state_dict(), avg_loss
    
    def train_client_real(self, client_data: List[Tuple], client_id: int, 
                         local_epochs: int = 5, learning_rate: float = 0.001) -> Tuple[Dict, float]:
        """
        Perform real training on client data
        
        Args:
            client_data: List of (features, labels) tuples
            client_id: Client identifier
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
        
        Returns:
            weights: Updated model state dictionary
            loss: Average loss value
        """
        # Create a copy of the global model
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        
        # Set up optimizer and criterion
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
        
        # Training loop
        epoch_losses = []
        
        for epoch in range(local_epochs):
            batch_losses = []
            
            for features, labels in client_data:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                log_probs = local_model(features)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
        
        avg_loss = np.mean(epoch_losses)
        
        return local_model.state_dict(), avg_loss
    
    def detect_attackers(self, client_data_dict: Dict[int, List[Tuple]], 
                        selected_clients: List[int]) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using the configured detection method
        
        Args:
            client_data_dict: Dictionary mapping client IDs to their data
            selected_clients: List of selected client IDs for this round
        
        Returns:
            clean_clients: List of clean client IDs
            attackers: List of detected attacker IDs
        """
        # Collect fake updates for detection
        client_weights = []
        client_losses = []
        
        for client_id in selected_clients:
            if client_id in client_data_dict:
                weights, loss = self.train_client_fake(client_data_dict[client_id], client_id)
                client_weights.append(weights)
                client_losses.append(loss)
            else:
                print(f"Warning: Client {client_id} not found in data dictionary")
                continue
        
        # Detect malicious clients
        clean_clients, attackers = detect_malicious_clients(
            client_losses=client_losses,
            client_weights=client_weights,
            client_ids=selected_clients,
            method=self.detection_method,
            epsilon=self.ldp_epsilon,
            sensitivity=self.ldp_sensitivity
        )
        
        # Record detection results
        detection_record = {
            'round': len(self.detection_history),
            'selected_clients': selected_clients,
            'clean_clients': clean_clients,
            'attackers': attackers,
            'client_losses': client_losses,
            'detection_method': self.detection_method
        }
        self.detection_history.append(detection_record)
        
        return clean_clients, attackers
    
    def federated_training_round(self, client_data_dict: Dict[int, List[Tuple]], 
                               selected_clients: List[int], 
                               local_epochs: int = 5,
                               learning_rate: float = 0.001) -> Dict:
        """
        Perform one round of federated training with detection
        
        Args:
            client_data_dict: Dictionary mapping client IDs to their data
            selected_clients: List of selected client IDs for this round
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for training
        
        Returns:
            round_results: Dictionary with round results
        """
        print(f"Starting federated training round with {len(selected_clients)} clients")
        
        # Step 1: Detect malicious clients
        clean_clients, attackers = self.detect_attackers(client_data_dict, selected_clients)
        
        print(f"Detected {len(attackers)} attackers: {attackers}")
        print(f"Proceeding with {len(clean_clients)} clean clients")
        
        # Step 2: Train on clean clients
        client_weights = []
        client_losses = []
        
        for client_id in clean_clients:
            if client_id in client_data_dict:
                weights, loss = self.train_client_real(
                    client_data_dict[client_id], 
                    client_id, 
                    local_epochs, 
                    learning_rate
                )
                client_weights.append(weights)
                client_losses.append(loss)
        
        # Step 3: Update global model
        if client_weights:
            global_weights = average_weights(client_weights)
            self.global_model.load_state_dict(global_weights)
        
        # Step 4: Record performance
        round_results = {
            'round': len(self.performance_history),
            'selected_clients': selected_clients,
            'clean_clients': clean_clients,
            'attackers': attackers,
            'avg_loss': np.mean(client_losses) if client_losses else 0.0,
            'num_clean_clients': len(clean_clients),
            'num_attackers': len(attackers)
        }
        self.performance_history.append(round_results)
        
        return round_results
    
    def evaluate_model(self, test_data: List[Tuple]) -> Dict:
        """
        Evaluate the global model on test data
        
        Args:
            test_data: List of (features, labels) tuples
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        self.global_model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.NLLLoss()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_data:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                log_probs = self.global_model(features)
                loss = criterion(log_probs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(log_probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_data)
        
        # Calculate additional metrics
        precision = calculate_precision(all_labels, all_predictions)
        recall = calculate_recall(all_labels, all_predictions)
        f1 = calculate_f1_score(all_labels, all_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': total
        }
        
        return metrics
    
    def get_detection_metrics(self, true_attackers: List[int]) -> Dict:
        """
        Calculate detection performance metrics
        
        Args:
            true_attackers: List of actual attacker IDs
        
        Returns:
            metrics: Dictionary with detection metrics
        """
        if not self.detection_history:
            return {}
        
        # Get latest detection results
        latest_detection = self.detection_history[-1]
        detected_attackers = latest_detection['attackers']
        all_clients = latest_detection['selected_clients']
        
        # Calculate metrics
        accuracy = calculate_accuracy(true_attackers, detected_attackers, all_clients)
        precision = calculate_precision(true_attackers, detected_attackers)
        recall = calculate_recall(true_attackers, detected_attackers)
        f1 = calculate_f1_score(true_attackers, detected_attackers)
        
        metrics = {
            'detection_accuracy': accuracy,
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1,
            'true_attackers': true_attackers,
            'detected_attackers': detected_attackers,
            'false_positives': len(set(detected_attackers) - set(true_attackers)),
            'false_negatives': len(set(true_attackers) - set(detected_attackers))
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the global model"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'model_type': self.model_type
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
    
    def get_detection_summary(self) -> Dict:
        """Get summary of detection results"""
        if not self.detection_history:
            return {}
        
        total_rounds = len(self.detection_history)
        total_detected = sum(len(record['attackers']) for record in self.detection_history)
        avg_attackers_per_round = total_detected / total_rounds
        
        return {
            'total_rounds': total_rounds,
            'total_attackers_detected': total_detected,
            'avg_attackers_per_round': avg_attackers_per_round,
            'detection_method': self.detection_method,
            'ldp_epsilon': self.ldp_epsilon
        }
