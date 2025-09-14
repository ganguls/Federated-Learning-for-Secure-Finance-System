#!/usr/bin/env python3
"""
Detection Adapter for Flower FL System
Converts scikit-learn LogisticRegression results to PyTorch format for detection
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import sys
import os

# Add detection system to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_poisoning_detection'))

try:
    from detection_utils import detect_malicious_clients, apply_ldp, eliminate_kmeans
    from tabular_model import create_tabular_model
    from tabular_data_utils import TabularDataset
except ImportError as e:
    logging.warning(f"Detection modules not available: {e}")
    # Fallback implementations
    def detect_malicious_clients(*args, **kwargs):
        return [], []
    def apply_ldp(*args, **kwargs):
        return args[0] if args else []
    def eliminate_kmeans(*args, **kwargs):
        return [], []

logger = logging.getLogger(__name__)

class FLDetectionAdapter:
    """
    Adapter class to integrate PyTorch-based detection with Flower scikit-learn clients
    """
    
    def __init__(self, 
                 input_dim: int = 20,
                 num_classes: int = 2,
                 detection_method: str = 'kmeans',
                 ldp_epsilon: float = 1.0,
                 ldp_sensitivity: float = 0.001,
                 enable_logging: bool = True):
        """
        Initialize detection adapter
        
        Args:
            input_dim: Number of input features (inferred from client data)
            num_classes: Number of output classes (2 for binary classification)
            detection_method: Detection method ('kmeans', 'fixed_percentage', 'z_score')
            ldp_epsilon: LDP privacy parameter
            ldp_sensitivity: LDP sensitivity parameter
            enable_logging: Enable detailed logging
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.detection_method = detection_method
        self.ldp_epsilon = ldp_epsilon
        self.ldp_sensitivity = ldp_sensitivity
        self.enable_logging = enable_logging
        
        # Detection history
        self.detection_history = []
        self.detection_metrics = {
            'total_detections': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Initialize PyTorch model for detection (if needed)
        self.detection_model = None
        if detection_method in ['kmeans', 'z_score', 'fixed_percentage']:
            # Lightweight detection - no PyTorch model needed
            pass
        else:
            # Advanced detection methods might need PyTorch model
            try:
                self.detection_model = create_tabular_model(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    model_type='credit_risk'
                )
            except Exception as e:
                logger.warning(f"Could not initialize PyTorch model: {e}")
                self.detection_model = None
        
        logger.info(f"Detection adapter initialized: method={detection_method}, epsilon={ldp_epsilon}")
    
    def convert_sklearn_to_detection_format(self, 
                                          client_results: List[Tuple],
                                          client_data_dict: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[str], List[float], List[Dict]]:
        """
        Convert Flower client results to detection format
        
        Args:
            client_results: List of (parameters, num_examples, metrics) from Flower clients
            client_data_dict: Optional dict mapping client_id to actual data for advanced detection
            
        Returns:
            Tuple of (client_vectors, client_ids, client_losses, client_metrics)
        """
        client_vectors = []
        client_ids = []
        client_losses = []
        client_metrics = []
        
        for i, result in enumerate(client_results):
            try:
                # Parse Flower result format
                if len(result) >= 3:
                    parameters, num_examples, metrics = result[0], result[1], result[2]
                else:
                    logger.warning(f"Unexpected result format for client {i}: {len(result)} elements")
                    continue
                
                # Extract client ID
                client_id = str(metrics.get("client_id", f"client_{i}"))
                client_ids.append(client_id)
                
                # Convert scikit-learn parameters to detection format
                if isinstance(parameters, (list, tuple)) and len(parameters) >= 2:
                    coef, intercept = parameters[0], parameters[1]
                    
                    # Flatten parameters for detection
                    coef_flat = np.array(coef).flatten()
                    intercept_flat = np.array(intercept).flatten()
                    param_vector = np.concatenate([coef_flat, intercept_flat])
                    
                    client_vectors.append(param_vector)
                else:
                    logger.warning(f"Unexpected parameter format for client {client_id}")
                    continue
                
                # Extract loss/accuracy metrics
                loss = 1.0 - float(metrics.get("accuracy", 0.0))  # Convert accuracy to loss
                client_losses.append(loss)
                
                # Store full metrics
                client_metrics.append(metrics)
                
                if self.enable_logging:
                    logger.debug(f"Client {client_id}: loss={loss:.4f}, params_shape={param_vector.shape}")
                    
            except Exception as e:
                logger.error(f"Error processing client {i} result: {e}")
                continue
        
        logger.info(f"Converted {len(client_vectors)} client results for detection")
        return client_vectors, client_ids, client_losses, client_metrics
    
    def detect_malicious_clients(self, 
                               client_results: List[Tuple],
                               client_data_dict: Optional[Dict] = None,
                               true_malicious: Optional[List[str]] = None) -> Tuple[List[str], Dict]:
        """
        Detect malicious clients using the configured detection method
        
        Args:
            client_results: List of Flower client results
            client_data_dict: Optional client data for advanced detection
            true_malicious: Optional list of known malicious client IDs for evaluation
            
        Returns:
            Tuple of (detected_malicious_ids, detection_metrics)
        """
        try:
            # Convert client results to detection format
            client_vectors, client_ids, client_losses, client_metrics = self.convert_sklearn_to_detection_format(
                client_results, client_data_dict
            )
            
            if len(client_vectors) < 2:
                logger.warning("Not enough clients for detection (need at least 2)")
                return [], self.detection_metrics
            
            # Apply LDP if enabled
            if self.ldp_epsilon > 0:
                client_vectors = [apply_ldp(vec, self.ldp_epsilon, self.ldp_sensitivity) for vec in client_vectors]
                logger.info(f"Applied LDP with epsilon={self.ldp_epsilon}")
            
            # Run detection based on method
            if self.detection_method == 'kmeans':
                detected_indices = self._detect_by_kmeans(client_vectors, client_losses)
            elif self.detection_method == 'fixed_percentage':
                detected_indices = self._detect_by_fixed_percentage(client_vectors, client_losses)
            elif self.detection_method == 'z_score':
                detected_indices = self._detect_by_zscore(client_vectors, client_losses)
            else:
                logger.error(f"Unknown detection method: {self.detection_method}")
                return [], self.detection_metrics
            
            # Convert indices to client IDs
            detected_malicious = [client_ids[i] for i in detected_indices]
            
            # Calculate detection metrics
            detection_metrics = self._calculate_detection_metrics(
                true_malicious or [], detected_malicious, client_ids
            )
            
            # Update detection history
            detection_record = {
                'round': len(self.detection_history),
                'timestamp': pd.Timestamp.now().isoformat(),
                'method': self.detection_method,
                'total_clients': len(client_ids),
                'detected_malicious': detected_malicious,
                'detection_metrics': detection_metrics,
                'ldp_epsilon': self.ldp_epsilon
            }
            self.detection_history.append(detection_record)
            
            # Update cumulative metrics
            self._update_cumulative_metrics(detection_metrics)
            
            logger.info(f"Detection completed: {len(detected_malicious)}/{len(client_ids)} clients flagged as malicious")
            logger.info(f"Detection metrics: accuracy={detection_metrics['accuracy']:.3f}, precision={detection_metrics['precision']:.3f}")
            
            return detected_malicious, detection_metrics
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return [], self.detection_metrics
    
    def _detect_by_kmeans(self, client_vectors: List[np.ndarray], client_losses: List[float]) -> List[int]:
        """Detect malicious clients using K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            # Stack vectors for clustering
            X = np.stack(client_vectors)
            
            # Apply K-means with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Identify malicious cluster (higher mean loss)
            cluster_0_losses = [client_losses[i] for i, label in enumerate(cluster_labels) if label == 0]
            cluster_1_losses = [client_losses[i] for i, label in enumerate(cluster_labels) if label == 1]
            
            malicious_cluster = 0 if np.mean(cluster_0_losses) > np.mean(cluster_1_losses) else 1
            detected_indices = [i for i, label in enumerate(cluster_labels) if label == malicious_cluster]
            
            logger.info(f"K-means detection: cluster {malicious_cluster} flagged as malicious ({len(detected_indices)} clients)")
            return detected_indices
            
        except Exception as e:
            logger.error(f"K-means detection failed: {e}")
            return []
    
    def _detect_by_fixed_percentage(self, client_vectors: List[np.ndarray], client_losses: List[float]) -> List[int]:
        """Detect malicious clients by removing top percentage by loss"""
        try:
            # Sort by loss (descending) and take top 20%
            percentage = 0.2
            n_remove = max(1, int(len(client_losses) * percentage))
            
            # Get indices of clients with highest losses
            sorted_indices = np.argsort(client_losses)[::-1]  # Descending order
            detected_indices = sorted_indices[:n_remove].tolist()
            
            logger.info(f"Fixed percentage detection: removed top {n_remove} clients by loss")
            return detected_indices
            
        except Exception as e:
            logger.error(f"Fixed percentage detection failed: {e}")
            return []
    
    def _detect_by_zscore(self, client_vectors: List[np.ndarray], client_losses: List[float]) -> List[int]:
        """Detect malicious clients using z-score thresholding"""
        try:
            from scipy import stats
            
            # Calculate z-scores for losses
            z_scores = np.abs(stats.zscore(client_losses, nan_policy='omit'))
            threshold = 2.0  # 2 standard deviations
            
            # Flag clients with z-score above threshold
            detected_indices = [i for i, z in enumerate(z_scores) if z > threshold]
            
            logger.info(f"Z-score detection: {len(detected_indices)} clients flagged (threshold={threshold})")
            return detected_indices
            
        except Exception as e:
            logger.error(f"Z-score detection failed: {e}")
            return []
    
    def _calculate_detection_metrics(self, true_malicious: List[str], detected_malicious: List[str], all_clients: List[str]) -> Dict:
        """Calculate detection performance metrics"""
        try:
            true_set = set(true_malicious)
            detected_set = set(detected_malicious)
            all_set = set(all_clients)
            
            # Calculate confusion matrix components
            true_positives = len(true_set & detected_set)
            false_positives = len(detected_set - true_set)
            false_negatives = len(true_set - detected_set)
            true_negatives = len(all_set - true_set - detected_set)
            
            # Calculate metrics
            total = len(all_clients)
            accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives,
                'total_clients': total
            }
            
        except Exception as e:
            logger.error(f"Error calculating detection metrics: {e}")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, 'true_negatives': 0
            }
    
    def _update_cumulative_metrics(self, detection_metrics: Dict):
        """Update cumulative detection metrics"""
        self.detection_metrics['total_detections'] += 1
        
        # Update running averages
        n = self.detection_metrics['total_detections']
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            current_avg = self.detection_metrics[metric]
            new_value = detection_metrics[metric]
            self.detection_metrics[metric] = (current_avg * (n - 1) + new_value) / n
        
        # Update counts
        self.detection_metrics['false_positives'] += detection_metrics['false_positives']
        self.detection_metrics['false_negatives'] += detection_metrics['false_negatives']
    
    def get_detection_summary(self) -> Dict:
        """Get summary of detection results"""
        return {
            'total_rounds': len(self.detection_history),
            'cumulative_metrics': self.detection_metrics,
            'detection_method': self.detection_method,
            'ldp_epsilon': self.ldp_epsilon,
            'recent_detections': self.detection_history[-5:] if self.detection_history else []
        }
    
    def save_detection_logs(self, filepath: str = "detection_logs.json"):
        """Save detection history to file"""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.detection_history, f, indent=2, default=str)
            logger.info(f"Detection logs saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving detection logs: {e}")

# Utility functions for backward compatibility
def create_detection_adapter(input_dim: int = 20, 
                           detection_method: str = 'kmeans',
                           ldp_epsilon: float = 1.0) -> FLDetectionAdapter:
    """Factory function to create detection adapter"""
    return FLDetectionAdapter(
        input_dim=input_dim,
        detection_method=detection_method,
        ldp_epsilon=ldp_epsilon
    )
