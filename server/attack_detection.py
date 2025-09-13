"""
Advanced Data Poisoning Attack Detection System
Implements multiple detection mechanisms for federated learning security
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from scipy import stats
import logging
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataPoisoningDetector:
    """
    Comprehensive data poisoning attack detection system
    """
    
    def __init__(self, 
                 anomaly_threshold: float = 0.1,
                 clustering_threshold: float = 0.3,
                 statistical_threshold: float = 2.0,
                 history_window: int = 10):
        """
        Initialize the data poisoning detector
        
        Args:
            anomaly_threshold: Threshold for anomaly detection (0-1)
            clustering_threshold: Threshold for clustering-based detection (0-1)
            statistical_threshold: Z-score threshold for statistical detection
            history_window: Number of rounds to keep in history
        """
        self.anomaly_threshold = anomaly_threshold
        self.clustering_threshold = clustering_threshold
        self.statistical_threshold = statistical_threshold
        self.history_window = history_window
        
        # Detection state
        self.client_history = defaultdict(lambda: deque(maxlen=history_window))
        self.parameter_history = defaultdict(lambda: deque(maxlen=history_window))
        self.performance_history = defaultdict(lambda: deque(maxlen=history_window))
        self.detected_attacks = defaultdict(list)
        self.attack_scores = defaultdict(float)
        
        # Detection methods
        self.detection_methods = {
            'statistical': self._statistical_detection,
            'clustering': self._clustering_detection,
            'anomaly': self._anomaly_detection,
            'gradient': self._gradient_analysis,
            'performance': self._performance_analysis,
            'consensus': self._consensus_analysis
        }
        
        logger.info("Data poisoning detector initialized")
    
    def detect_attacks(self, 
                      client_updates: List[Tuple], 
                      round_number: int) -> Dict[str, Set[str]]:
        """
        Main detection function that runs all detection methods
        
        Args:
            client_updates: List of (parameters, metrics, num_examples) tuples
            round_number: Current training round
            
        Returns:
            Dictionary mapping detection method to set of malicious client IDs
        """
        logger.info(f"Running attack detection for round {round_number}")
        
        # Extract client data
        client_data = self._extract_client_data(client_updates)
        
        # Update history
        self._update_history(client_data, round_number)
        
        # Run all detection methods
        detection_results = {}
        
        for method_name, detection_func in self.detection_methods.items():
            try:
                malicious_clients = detection_func(client_data, round_number)
                detection_results[method_name] = malicious_clients
                logger.info(f"{method_name} detection found {len(malicious_clients)} malicious clients")
            except Exception as e:
                logger.error(f"Error in {method_name} detection: {e}")
                detection_results[method_name] = set()
        
        # Combine results using ensemble approach
        final_malicious = self._ensemble_detection(detection_results)
        
        # Update attack scores
        self._update_attack_scores(final_malicious, round_number)
        
        return detection_results
    
    def _extract_client_data(self, client_updates: List[Tuple]) -> Dict[str, Dict]:
        """Extract and structure client data from updates"""
        client_data = {}
        
        for update in client_updates:
            if len(update) >= 3:
                parameters, metrics, num_examples = update[0], update[1], update[2]
                
                if isinstance(metrics, dict):
                    client_id = metrics.get("client_id", "unknown")
                    
                    # Extract parameters
                    if isinstance(parameters, (list, tuple)) and len(parameters) >= 2:
                        coef, intercept = parameters[0], parameters[1]
                        if hasattr(coef, 'flatten'):
                            flat_params = np.concatenate([coef.flatten(), intercept.flatten()])
                        else:
                            flat_params = np.concatenate([np.array(coef).flatten(), np.array(intercept).flatten()])
                    else:
                        flat_params = np.array([])
                    
                    client_data[client_id] = {
                        'parameters': flat_params,
                        'metrics': metrics,
                        'num_examples': num_examples,
                        'accuracy': metrics.get('accuracy', 0.0),
                        'loss': metrics.get('loss', 1.0),
                        'is_malicious': metrics.get('is_malicious', False),
                        'labels_flipped': metrics.get('labels_flipped', 0)
                    }
        
        return client_data
    
    def _update_history(self, client_data: Dict[str, Dict], round_number: int):
        """Update client history with new data"""
        for client_id, data in client_data.items():
            self.client_history[client_id].append({
                'round': round_number,
                'accuracy': data['accuracy'],
                'loss': data['loss'],
                'num_examples': data['num_examples']
            })
            
            if len(data['parameters']) > 0:
                self.parameter_history[client_id].append(data['parameters'])
            
            self.performance_history[client_id].append({
                'round': round_number,
                'accuracy': data['accuracy'],
                'loss': data['loss']
            })
    
    def _statistical_detection(self, client_data: Dict[str, Dict], round_number: int) -> Set[str]:
        """Statistical anomaly detection using Z-score analysis"""
        malicious_clients = set()
        
        if len(client_data) < 3:
            return malicious_clients
        
        # Analyze parameter distributions
        parameters_list = []
        client_ids = []
        
        for client_id, data in client_data.items():
            if len(data['parameters']) > 0:
                parameters_list.append(data['parameters'])
                client_ids.append(client_id)
        
        if len(parameters_list) < 3:
            return malicious_clients
        
        try:
            # Calculate Z-scores for each parameter dimension
            parameters_array = np.array(parameters_list)
            z_scores = np.abs(stats.zscore(parameters_array, axis=0))
            
            # Find clients with extreme parameter values
            max_z_scores = np.max(z_scores, axis=1)
            
            for i, (client_id, max_z) in enumerate(zip(client_ids, max_z_scores)):
                if max_z > self.statistical_threshold:
                    malicious_clients.add(client_id)
                    logger.warning(f"Statistical detection: Client {client_id} has extreme parameters (Z-score: {max_z:.2f})")
        
        except Exception as e:
            logger.error(f"Statistical detection error: {e}")
        
        return malicious_clients
    
    def _clustering_detection(self, client_data: Dict[str, Dict], round_number: int) -> Set[str]:
        """Clustering-based detection using K-means and DBSCAN"""
        malicious_clients = set()
        
        if len(client_data) < 3:
            return malicious_clients
        
        # Extract parameters for clustering
        parameters_list = []
        client_ids = []
        
        for client_id, data in client_data.items():
            if len(data['parameters']) > 0:
                parameters_list.append(data['parameters'])
                client_ids.append(client_id)
        
        if len(parameters_list) < 3:
            return malicious_clients
        
        try:
            parameters_array = np.array(parameters_list)
            
            # Normalize parameters
            scaler = StandardScaler()
            normalized_params = scaler.fit_transform(parameters_array)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(normalized_params)
            
            # Identify smaller cluster as potentially malicious
            cluster_0_count = np.sum(kmeans_labels == 0)
            cluster_1_count = np.sum(kmeans_labels == 1)
            malicious_cluster = 0 if cluster_0_count < cluster_1_count else 1
            
            # DBSCAN clustering for additional validation
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            dbscan_labels = dbscan.fit_predict(normalized_params)
            
            # Combine results
            for i, (client_id, kmeans_label, dbscan_label) in enumerate(zip(client_ids, kmeans_labels, dbscan_labels)):
                is_kmeans_outlier = kmeans_label == malicious_cluster
                is_dbscan_outlier = dbscan_label == -1  # -1 indicates outlier in DBSCAN
                
                if is_kmeans_outlier or is_dbscan_outlier:
                    malicious_clients.add(client_id)
                    logger.warning(f"Clustering detection: Client {client_id} identified as outlier")
        
        except Exception as e:
            logger.error(f"Clustering detection error: {e}")
        
        return malicious_clients
    
    def _anomaly_detection(self, client_data: Dict[str, Dict], round_number: int) -> Set[str]:
        """Isolation Forest-based anomaly detection"""
        malicious_clients = set()
        
        if len(client_data) < 3:
            return malicious_clients
        
        # Extract parameters for anomaly detection
        parameters_list = []
        client_ids = []
        
        for client_id, data in client_data.items():
            if len(data['parameters']) > 0:
                parameters_list.append(data['parameters'])
                client_ids.append(client_id)
        
        if len(parameters_list) < 3:
            return malicious_clients
        
        try:
            parameters_array = np.array(parameters_list)
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=self.anomaly_threshold, random_state=42)
            anomaly_labels = iso_forest.fit_predict(parameters_array)
            
            # Identify anomalies
            for i, (client_id, label) in enumerate(zip(client_ids, anomaly_labels)):
                if label == -1:  # -1 indicates anomaly
                    malicious_clients.add(client_id)
                    logger.warning(f"Anomaly detection: Client {client_id} identified as anomaly")
        
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
        
        return malicious_clients
    
    def _gradient_analysis(self, client_data: Dict[str, Dict], round_number: int) -> Set[str]:
        """Analyze gradient patterns for suspicious behavior"""
        malicious_clients = set()
        
        for client_id, data in client_data.items():
            if client_id not in self.parameter_history or len(self.parameter_history[client_id]) < 2:
                continue
            
            try:
                # Calculate gradient magnitude
                current_params = self.parameter_history[client_id][-1]
                previous_params = self.parameter_history[client_id][-2]
                
                gradient = current_params - previous_params
                gradient_magnitude = np.linalg.norm(gradient)
                
                # Check for extreme gradient values
                if gradient_magnitude > np.percentile([np.linalg.norm(p) for p in self.parameter_history[client_id]], 95):
                    malicious_clients.add(client_id)
                    logger.warning(f"Gradient analysis: Client {client_id} has extreme gradient magnitude")
                
                # Check for gradient direction anomalies
                if len(self.parameter_history[client_id]) >= 3:
                    prev_gradient = previous_params - self.parameter_history[client_id][-3]
                    gradient_similarity = np.dot(gradient, prev_gradient) / (np.linalg.norm(gradient) * np.linalg.norm(prev_gradient) + 1e-8)
                    
                    if gradient_similarity < -0.5:  # Sudden direction change
                        malicious_clients.add(client_id)
                        logger.warning(f"Gradient analysis: Client {client_id} has suspicious gradient direction change")
            
            except Exception as e:
                logger.error(f"Gradient analysis error for client {client_id}: {e}")
        
        return malicious_clients
    
    def _performance_analysis(self, client_data: Dict[str, Dict], round_number: int) -> Set[str]:
        """Analyze performance patterns for suspicious behavior"""
        malicious_clients = set()
        
        for client_id, data in client_data.items():
            if client_id not in self.performance_history or len(self.performance_history[client_id]) < 3:
                continue
            
            try:
                # Analyze accuracy trends
                accuracies = [p['accuracy'] for p in self.performance_history[client_id]]
                
                # Check for sudden performance drops
                if len(accuracies) >= 2:
                    accuracy_change = accuracies[-1] - accuracies[-2]
                    if accuracy_change < -0.1:  # Sudden 10% drop
                        malicious_clients.add(client_id)
                        logger.warning(f"Performance analysis: Client {client_id} has sudden accuracy drop")
                
                # Check for consistently poor performance
                if len(accuracies) >= 3:
                    avg_accuracy = np.mean(accuracies[-3:])
                    if avg_accuracy < 0.3:  # Consistently poor performance
                        malicious_clients.add(client_id)
                        logger.warning(f"Performance analysis: Client {client_id} has consistently poor performance")
                
                # Check for performance variance
                if len(accuracies) >= 4:
                    accuracy_variance = np.var(accuracies[-4:])
                    if accuracy_variance > 0.1:  # High variance
                        malicious_clients.add(client_id)
                        logger.warning(f"Performance analysis: Client {client_id} has high performance variance")
            
            except Exception as e:
                logger.error(f"Performance analysis error for client {client_id}: {e}")
        
        return malicious_clients
    
    def _consensus_analysis(self, client_data: Dict[str, Dict], round_number: int) -> Set[str]:
        """Consensus-based detection using majority voting"""
        malicious_clients = set()
        
        if len(client_data) < 3:
            return malicious_clients
        
        # Calculate consensus metrics
        accuracies = [data['accuracy'] for data in client_data.values()]
        losses = [data['loss'] for data in client_data.values()]
        
        median_accuracy = np.median(accuracies)
        median_loss = np.median(losses)
        
        # Find clients that deviate significantly from consensus
        for client_id, data in client_data.items():
            accuracy_deviation = abs(data['accuracy'] - median_accuracy)
            loss_deviation = abs(data['loss'] - median_loss)
            
            if accuracy_deviation > 0.2 or loss_deviation > 0.3:  # Significant deviation
                malicious_clients.add(client_id)
                logger.warning(f"Consensus analysis: Client {client_id} deviates from consensus")
        
        return malicious_clients
    
    def _ensemble_detection(self, detection_results: Dict[str, Set[str]]) -> Set[str]:
        """Combine results from multiple detection methods using ensemble approach"""
        if not detection_results:
            return set()
        
        # Count votes for each client
        client_votes = defaultdict(int)
        total_methods = len(detection_results)
        
        for method_name, malicious_clients in detection_results.items():
            for client_id in malicious_clients:
                client_votes[client_id] += 1
        
        # Determine malicious clients based on majority vote
        malicious_clients = set()
        threshold = max(1, total_methods // 2)  # Majority threshold
        
        for client_id, votes in client_votes.items():
            if votes >= threshold:
                malicious_clients.add(client_id)
                logger.info(f"Ensemble detection: Client {client_id} flagged by {votes}/{total_methods} methods")
        
        return malicious_clients
    
    def _update_attack_scores(self, malicious_clients: Set[str], round_number: int):
        """Update attack scores for clients"""
        for client_id in malicious_clients:
            self.attack_scores[client_id] += 1
            self.detected_attacks[client_id].append({
                'round': round_number,
                'timestamp': datetime.now().isoformat(),
                'score': self.attack_scores[client_id]
            })
    
    def get_attack_summary(self) -> Dict:
        """Get summary of detected attacks"""
        summary = {
            'total_attacks': sum(len(attacks) for attacks in self.detected_attacks.values()),
            'malicious_clients': list(self.attack_scores.keys()),
            'attack_scores': dict(self.attack_scores),
            'recent_attacks': {}
        }
        
        # Get recent attacks (last 5 rounds)
        for client_id, attacks in self.detected_attacks.items():
            recent_attacks = [a for a in attacks if a['round'] >= max(0, max(a['round'] for a in attacks) - 5)]
            summary['recent_attacks'][client_id] = recent_attacks
        
        return summary
    
    def reset_detection(self):
        """Reset detection state"""
        self.client_history.clear()
        self.parameter_history.clear()
        self.performance_history.clear()
        self.detected_attacks.clear()
        self.attack_scores.clear()
        logger.info("Attack detection state reset")

