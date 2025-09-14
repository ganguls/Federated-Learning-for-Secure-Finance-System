#!/usr/bin/env python3
"""
Comprehensive Federated Learning Attack Detection System
========================================================

This module orchestrates all 3 detection methods into a unified pipeline:
1. Basic Detection (Threshold + LDP)
2. Server-Side Detection (K-means clustering)
3. Advanced Detection (LDP + K-means + Ensemble)

Author: FL Defense System
Date: 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import random

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fl_defenses'))

# Import detection modules
try:
    from attack_detection_service import run_attack_detection
    from fl_defenses.detector import detect_malicious_clients, apply_ldp, eliminate_kmeans
except ImportError as e:
    print(f"Warning: Could not import detection modules: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveDetectionPipeline:
    """
    Master orchestrator for all attack detection methods.
    Coordinates 3 different detection approaches and fuses results.
    """
    
    def __init__(self):
        self.detection_results = {
            'basic_detection': {},
            'server_detection': {},
            'advanced_detection': {},
            'combined_results': {},
            'final_verdict': {},
            'execution_time': {},
            'confidence_scores': {}
        }
        
        # Detection method weights (can be adjusted based on performance)
        self.method_weights = {
            'basic_detection': 0.3,
            'server_detection': 0.4,
            'advanced_detection': 0.3
        }
        
        # Attack type mappings
        self.attack_types = {
            'label_flipping': 'Data Poisoning',
            'gradient_poisoning': 'Gradient Attack',
            'backdoor': 'Backdoor Attack',
            'model_poisoning': 'Model Poisoning'
        }
    
    def prepare_real_attack_scenario(self, malicious_percentage=20, attack_type='label_flipping', 
                                   epsilon=1.0, n_clients=10):
        """
        Prepare realistic attack scenario using real FL client data patterns.
        """
        try:
            # Load real client data patterns
            data_paths = [
                f"Datapre/FL_clients/client_{i}.csv" for i in range(1, 11)
            ]
            
            # Check if real data exists, otherwise use enhanced synthetic data
            real_data_available = any(os.path.exists(path) for path in data_paths)
            
            if real_data_available:
                logger.info("Using real FL client data for attack scenario")
                return self._load_real_client_data(data_paths, malicious_percentage, attack_type)
            else:
                logger.info("Using enhanced synthetic data for attack scenario")
                return self._generate_enhanced_synthetic_data(n_clients, malicious_percentage, attack_type)
                
        except Exception as e:
            logger.error(f"Error preparing attack scenario: {e}")
            return self._generate_enhanced_synthetic_data(n_clients, malicious_percentage, attack_type)
    
    def _load_real_client_data(self, data_paths, malicious_percentage, attack_type):
        """Load and process real client data for attack simulation."""
        client_data = []
        client_losses = []
        true_malicious = []
        
        n_malicious = int(len(data_paths) * malicious_percentage / 100)
        malicious_indices = random.sample(range(len(data_paths)), n_malicious)
        
        for i, path in enumerate(data_paths):
            if os.path.exists(path):
                try:
                    # Load client data
                    df = pd.read_csv(path)
                    
                    # Prepare features and labels
                    if 'target' in df.columns:
                        y = df['target'].values
                    else:
                        y = df.iloc[:, -1].values  # Last column as target
                    
                    X = df.drop(columns=['target'] if 'target' in df.columns else [df.columns[-1]]).values
                    
                    # Train model and calculate loss
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    loss = 1 - accuracy_score(y, y_pred)
                    
                    # Apply attack if malicious
                    if i in malicious_indices:
                        if attack_type == 'label_flipping':
                            loss = loss * 2.0
                        elif attack_type == 'gradient_poisoning':
                            loss = loss * 1.5
                        elif attack_type == 'backdoor':
                            loss = loss * 3.0
                        true_malicious.append(i)
                    
                    client_data.append((X, y))
                    client_losses.append(loss)
                    
                except Exception as e:
                    logger.warning(f"Error loading client data from {path}: {e}")
                    # Fallback to synthetic data for this client
                    X, y = self._generate_synthetic_client_data()
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    loss = 1 - accuracy_score(y, y_pred)
                    
                    if i in malicious_indices:
                        loss = loss * 2.0
                        true_malicious.append(i)
                    
                    client_data.append((X, y))
                    client_losses.append(loss)
        
        return {
            'client_data': client_data,
            'client_losses': client_losses,
            'true_malicious': true_malicious,
            'attack_type': attack_type,
            'malicious_percentage': malicious_percentage
        }
    
    def _generate_enhanced_synthetic_data(self, n_clients, malicious_percentage, attack_type):
        """Generate enhanced synthetic data that mimics real FL patterns."""
        n_features = 10  # More realistic feature count
        n_samples_per_client = 200  # More realistic sample size
        
        # Generate correlated data (more realistic than random)
        X = np.random.randn(n_clients * n_samples_per_client, n_features)
        # Add some correlation between features
        for i in range(1, n_features):
            X[:, i] = X[:, i] + 0.3 * X[:, i-1]
        
        y = np.random.randint(0, 2, n_clients * n_samples_per_client)
        
        # Split data among clients
        client_data = []
        client_losses = []
        true_malicious = []
        
        n_malicious = int(n_clients * malicious_percentage / 100)
        malicious_indices = random.sample(range(n_clients), n_malicious)
        
        for i in range(n_clients):
            start_idx = i * n_samples_per_client
            end_idx = (i + 1) * n_samples_per_client
            X_client = X[start_idx:end_idx]
            y_client = y[start_idx:end_idx]
            
            # Train model and calculate loss
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_client, y_client)
            y_pred = model.predict(X_client)
            loss = 1 - accuracy_score(y_client, y_pred)
            
            # Apply attack if malicious
            if i in malicious_indices:
                if attack_type == 'label_flipping':
                    loss = loss * 2.0
                elif attack_type == 'gradient_poisoning':
                    loss = loss * 1.5
                elif attack_type == 'backdoor':
                    loss = loss * 3.0
                true_malicious.append(i)
            
            client_data.append((X_client, y_client))
            client_losses.append(loss)
        
        return {
            'client_data': client_data,
            'client_losses': client_losses,
            'true_malicious': true_malicious,
            'attack_type': attack_type,
            'malicious_percentage': malicious_percentage
        }
    
    def _generate_synthetic_client_data(self):
        """Generate synthetic data for a single client."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def run_basic_detection(self, scenario_data, epsilon=1.0):
        """
        Method 1: Basic threshold-based detection with LDP noise.
        """
        start_time = time.time()
        
        try:
            client_losses = scenario_data['client_losses']
            true_malicious = scenario_data['true_malicious']
            
            # Apply LDP noise
            noise_scale = 1e-4 / epsilon
            noisy_losses = []
            for loss in client_losses:
                noise = np.random.laplace(0, noise_scale)
                noisy_loss = loss + noise
                noisy_losses.append(max(0, noisy_loss))
            
            # Threshold-based detection
            mean_loss = np.mean(noisy_losses)
            std_loss = np.std(noisy_losses)
            threshold = mean_loss + 0.5 * std_loss
            detected_malicious = [i for i, loss in enumerate(noisy_losses) if loss > threshold]
            
            # Calculate metrics
            true_malicious_set = set(true_malicious)
            detected_malicious_set = set(detected_malicious)
            
            true_positives = len(true_malicious_set & detected_malicious_set)
            false_positives = len(detected_malicious_set - true_malicious_set)
            false_negatives = len(true_malicious_set - detected_malicious_set)
            
            accuracy = (true_positives + len(client_losses) - len(true_malicious) - false_positives) / len(client_losses)
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1 = 2 * precision * recall / max(1, precision + recall)
            
            execution_time = time.time() - start_time
            
            return {
                'detected_malicious': detected_malicious,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives
                },
                'execution_time': execution_time,
                'method': 'Basic Threshold + LDP',
                'confidence': f1  # Use F1 score as confidence
            }
            
        except Exception as e:
            logger.error(f"Error in basic detection: {e}")
            return {
                'detected_malicious': [],
                'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                'execution_time': time.time() - start_time,
                'method': 'Basic Threshold + LDP',
                'confidence': 0,
                'error': str(e)
            }
    
    def run_server_detection(self, scenario_data):
        """
        Method 2: Server-side K-means clustering detection.
        """
        start_time = time.time()
        
        try:
            client_data = scenario_data['client_data']
            true_malicious = scenario_data['true_malicious']
            
            # Extract model parameters for clustering
            client_updates = []
            for X_client, y_client in client_data:
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_client, y_client)
                # Flatten parameters for clustering
                params = np.concatenate([model.coef_.flatten(), model.intercept_])
                client_updates.append(params)
            
            # Normalize updates
            scaler = StandardScaler()
            normalized_updates = scaler.fit_transform(client_updates)
            
            # K-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_updates)
            
            # Identify malicious cluster (higher mean loss)
            cluster_means = []
            for cluster_id in range(2):
                cluster_mask = cluster_labels == cluster_id
                cluster_losses = [scenario_data['client_losses'][i] for i in range(len(cluster_mask)) if cluster_mask[i]]
                cluster_means.append(np.mean(cluster_losses) if cluster_losses else 0)
            
            malicious_cluster_id = np.argmax(cluster_means)
            detected_malicious = [i for i, label in enumerate(cluster_labels) if label == malicious_cluster_id]
            
            # Calculate metrics
            true_malicious_set = set(true_malicious)
            detected_malicious_set = set(detected_malicious)
            
            true_positives = len(true_malicious_set & detected_malicious_set)
            false_positives = len(detected_malicious_set - true_malicious_set)
            false_negatives = len(true_malicious_set - detected_malicious_set)
            
            accuracy = (true_positives + len(client_data) - len(true_malicious) - false_positives) / len(client_data)
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1 = 2 * precision * recall / max(1, precision + recall)
            
            execution_time = time.time() - start_time
            
            return {
                'detected_malicious': detected_malicious,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives
                },
                'execution_time': execution_time,
                'method': 'K-means Clustering',
                'confidence': f1,
                'cluster_info': {
                    'cluster_means': cluster_means,
                    'malicious_cluster_id': malicious_cluster_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error in server detection: {e}")
            return {
                'detected_malicious': [],
                'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                'execution_time': time.time() - start_time,
                'method': 'K-means Clustering',
                'confidence': 0,
                'error': str(e)
            }
    
    def run_advanced_detection(self, scenario_data, epsilon=1.0):
        """
        Method 3: Advanced LDP + K-means + Ensemble detection.
        """
        start_time = time.time()
        
        try:
            client_losses = scenario_data['client_losses']
            true_malicious = scenario_data['true_malicious']
            
            # Use the advanced detector module
            detection_results = detect_malicious_clients(
                client_losses, 
                true_malicious, 
                epsilon=epsilon,
                sensitivity=1e-4,
                n_clusters=2
            )
            
            if 'error' in detection_results:
                raise Exception(detection_results['error'])
            
            detected_malicious = detection_results.get('detected_malicious', [])
            metrics = detection_results.get('detection_metrics', {})
            
            # Calculate additional metrics
            true_malicious_set = set(true_malicious)
            detected_malicious_set = set(detected_malicious)
            
            true_positives = len(true_malicious_set & detected_malicious_set)
            false_positives = len(detected_malicious_set - true_malicious_set)
            false_negatives = len(true_malicious_set - detected_malicious_set)
            
            accuracy = detection_results.get('accuracy', 0)
            f1 = detection_results.get('f1_score', 0)
            
            execution_time = time.time() - start_time
            
            return {
                'detected_malicious': detected_malicious,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': f1,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives
                },
                'execution_time': execution_time,
                'method': 'Advanced LDP + K-means',
                'confidence': f1,
                'ldp_info': {
                    'noisy_losses': detection_results.get('noisy_losses', []),
                    'normalization_factor': detection_results.get('normalization_factor', 1.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced detection: {e}")
            return {
                'detected_malicious': [],
                'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                'execution_time': time.time() - start_time,
                'method': 'Advanced LDP + K-means',
                'confidence': 0,
                'error': str(e)
            }
    
    def intelligent_detection_fusion(self, basic_results, server_results, advanced_results):
        """
        Intelligently fuse results from all 3 detection methods.
        """
        try:
            # Extract detection results
            basic_detected = set(basic_results.get('detected_malicious', []))
            server_detected = set(server_results.get('detected_malicious', []))
            advanced_detected = set(advanced_results.get('detected_malicious', []))
            
            # Get confidence scores
            basic_confidence = basic_results.get('confidence', 0)
            server_confidence = server_results.get('confidence', 0)
            advanced_confidence = advanced_results.get('confidence', 0)
            
            # Weighted voting
            all_clients = basic_detected | server_detected | advanced_detected
            client_scores = {}
            
            for client_id in all_clients:
                score = 0
                total_weight = 0
                
                if client_id in basic_detected:
                    score += basic_confidence * self.method_weights['basic_detection']
                    total_weight += self.method_weights['basic_detection']
                
                if client_id in server_detected:
                    score += server_confidence * self.method_weights['server_detection']
                    total_weight += self.method_weights['server_detection']
                
                if client_id in advanced_detected:
                    score += advanced_confidence * self.method_weights['advanced_detection']
                    total_weight += self.method_weights['advanced_detection']
                
                # Normalize score
                if total_weight > 0:
                    client_scores[client_id] = score / total_weight
                else:
                    client_scores[client_id] = 0
            
            # Determine final malicious clients (threshold = 0.5)
            final_detected = [client_id for client_id, score in client_scores.items() if score >= 0.5]
            
            # Calculate ensemble metrics
            ensemble_accuracy = np.mean([basic_results.get('metrics', {}).get('accuracy', 0),
                                       server_results.get('metrics', {}).get('accuracy', 0),
                                       advanced_results.get('metrics', {}).get('accuracy', 0)])
            
            ensemble_f1 = np.mean([basic_results.get('metrics', {}).get('f1_score', 0),
                                 server_results.get('metrics', {}).get('f1_score', 0),
                                 advanced_results.get('metrics', {}).get('f1_score', 0)])
            
            return {
                'final_detected': final_detected,
                'client_scores': client_scores,
                'ensemble_metrics': {
                    'accuracy': ensemble_accuracy,
                    'f1_score': ensemble_f1,
                    'agreement_rate': len(basic_detected & server_detected & advanced_detected) / max(1, len(all_clients))
                },
                'method_agreement': {
                    'basic_server': len(basic_detected & server_detected),
                    'basic_advanced': len(basic_detected & advanced_detected),
                    'server_advanced': len(server_detected & advanced_detected),
                    'all_three': len(basic_detected & server_detected & advanced_detected)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in detection fusion: {e}")
            return {
                'final_detected': [],
                'client_scores': {},
                'ensemble_metrics': {'accuracy': 0, 'f1_score': 0, 'agreement_rate': 0},
                'method_agreement': {},
                'error': str(e)
            }
    
    def classify_attack_types(self, scenario_data, detection_results):
        """
        Classify specific attack types based on detection patterns.
        """
        try:
            attack_type = scenario_data.get('attack_type', 'unknown')
            detected_malicious = detection_results.get('final_detected', [])
            
            # Analyze attack patterns
            attack_classification = {
                'primary_attack_type': self.attack_types.get(attack_type, 'Unknown Attack'),
                'detected_clients': len(detected_malicious),
                'attack_severity': 'High' if len(detected_malicious) > 3 else 'Medium' if len(detected_malicious) > 1 else 'Low',
                'confidence_level': detection_results.get('ensemble_metrics', {}).get('f1_score', 0),
                'recommendations': []
            }
            
            # Generate recommendations based on attack type
            if attack_type == 'label_flipping':
                attack_classification['recommendations'].append('Implement label validation mechanisms')
                attack_classification['recommendations'].append('Use robust aggregation methods')
            elif attack_type == 'gradient_poisoning':
                attack_classification['recommendations'].append('Apply gradient clipping')
                attack_classification['recommendations'].append('Use Byzantine-robust aggregation')
            elif attack_type == 'backdoor':
                attack_classification['recommendations'].append('Implement backdoor detection')
                attack_classification['recommendations'].append('Use certified defense mechanisms')
            
            return attack_classification
            
        except Exception as e:
            logger.error(f"Error in attack classification: {e}")
            return {
                'primary_attack_type': 'Unknown',
                'detected_clients': 0,
                'attack_severity': 'Unknown',
                'confidence_level': 0,
                'recommendations': ['Investigate detection system'],
                'error': str(e)
            }
    
    def run_comprehensive_detection(self, malicious_percentage=20, attack_type='label_flipping', epsilon=1.0):
        """
        Main method: Run all 3 detection methods and fuse results.
        """
        start_time = time.time()
        logger.info(f"Starting comprehensive detection: {attack_type} attack, {malicious_percentage}% malicious")
        
        try:
            # Step 1: Prepare attack scenario
            scenario_data = self.prepare_real_attack_scenario(malicious_percentage, attack_type, epsilon)
            
            # Step 2: Run all 3 detection methods
            logger.info("Running Method 1: Basic Detection")
            basic_results = self.run_basic_detection(scenario_data, epsilon)
            
            logger.info("Running Method 2: Server Detection")
            server_results = self.run_server_detection(scenario_data)
            
            logger.info("Running Method 3: Advanced Detection")
            advanced_results = self.run_advanced_detection(scenario_data, epsilon)
            
            # Step 3: Fuse results intelligently
            logger.info("Fusing detection results")
            fusion_results = self.intelligent_detection_fusion(basic_results, server_results, advanced_results)
            
            # Step 4: Classify attack types
            attack_classification = self.classify_attack_types(scenario_data, fusion_results)
            
            # Step 5: Compile comprehensive results
            total_execution_time = time.time() - start_time
            
            comprehensive_results = {
                'timestamp': datetime.now().isoformat(),
                'scenario': {
                    'malicious_percentage': malicious_percentage,
                    'attack_type': attack_type,
                    'epsilon': epsilon,
                    'total_clients': len(scenario_data['client_data']),
                    'true_malicious': scenario_data['true_malicious']
                },
                'detection_methods': {
                    'basic_detection': basic_results,
                    'server_detection': server_results,
                    'advanced_detection': advanced_results
                },
                'fusion_results': fusion_results,
                'attack_classification': attack_classification,
                'performance': {
                    'total_execution_time': total_execution_time,
                    'method_times': {
                        'basic': basic_results.get('execution_time', 0),
                        'server': server_results.get('execution_time', 0),
                        'advanced': advanced_results.get('execution_time', 0)
                    }
                },
                'summary': {
                    'final_detected_malicious': fusion_results.get('final_detected', []),
                    'ensemble_accuracy': fusion_results.get('ensemble_metrics', {}).get('accuracy', 0),
                    'ensemble_f1_score': fusion_results.get('ensemble_metrics', {}).get('f1_score', 0),
                    'attack_type_detected': attack_classification.get('primary_attack_type', 'Unknown'),
                    'confidence_level': attack_classification.get('confidence_level', 0)
                }
            }
            
            logger.info(f"Comprehensive detection completed in {total_execution_time:.2f} seconds")
            logger.info(f"Final detection: {len(fusion_results.get('final_detected', []))} malicious clients")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive detection: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }

def main():
    """
    Main function for testing the comprehensive detection system.
    """
    pipeline = ComprehensiveDetectionPipeline()
    
    # Test with different scenarios
    test_scenarios = [
        {'malicious_percentage': 20, 'attack_type': 'label_flipping', 'epsilon': 1.0},
        {'malicious_percentage': 30, 'attack_type': 'gradient_poisoning', 'epsilon': 0.5},
        {'malicious_percentage': 10, 'attack_type': 'backdoor', 'epsilon': 2.0}
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"TEST SCENARIO {i}: {scenario['attack_type']} - {scenario['malicious_percentage']}% malicious")
        print(f"{'='*60}")
        
        results = pipeline.run_comprehensive_detection(**scenario)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✅ Detection completed successfully!")
            print(f"📊 Final Results:")
            print(f"   - Detected Malicious: {len(results['summary']['final_detected_malicious'])}")
            print(f"   - Ensemble Accuracy: {results['summary']['ensemble_accuracy']:.3f}")
            print(f"   - Ensemble F1-Score: {results['summary']['ensemble_f1_score']:.3f}")
            print(f"   - Attack Type: {results['summary']['attack_type_detected']}")
            print(f"   - Execution Time: {results['performance']['total_execution_time']:.2f}s")

if __name__ == "__main__":
    main()
