#!/usr/bin/env python3
"""
Test script for data poisoning attack detection system
"""

import sys
import os
import time
import random
import numpy as np
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent / "server"))

from attack_detection import DataPoisoningDetector
from attack_simulator import DataPoisoningSimulator, AttackType

def create_sample_client_data(num_clients=10, num_features=20):
    """Create sample client data for testing"""
    client_updates = []
    
    for client_id in range(1, num_clients + 1):
        # Generate normal parameters
        coef = np.random.normal(0, 1, num_features)
        intercept = np.random.normal(0, 1, 1)
        parameters = (coef, intercept)
        
        # Generate normal metrics
        accuracy = random.uniform(0.7, 0.9)
        loss = random.uniform(0.1, 0.5)
        
        metrics = {
            "client_id": str(client_id),
            "accuracy": accuracy,
            "loss": loss,
            "precision": accuracy + random.uniform(-0.05, 0.05),
            "recall": accuracy + random.uniform(-0.05, 0.05),
            "f1_score": accuracy + random.uniform(-0.05, 0.05),
            "is_malicious": False,
            "labels_flipped": 0
        }
        
        num_examples = random.randint(1000, 5000)
        client_updates.append((parameters, metrics, num_examples))
    
    return client_updates

def test_attack_detection():
    """Test the attack detection system"""
    print("🧪 Testing Data Poisoning Attack Detection System")
    print("=" * 60)
    
    # Initialize detector and simulator
    detector = DataPoisoningDetector(
        anomaly_threshold=0.1,
        clustering_threshold=0.3,
        statistical_threshold=2.0,
        history_window=5
    )
    
    simulator = DataPoisoningSimulator(attack_probability=0.0)  # Manual control
    
    print("\n1. Testing Normal Client Detection")
    print("-" * 40)
    
    # Test with normal clients
    normal_clients = create_sample_client_data(10)
    detection_results = detector.detect_attacks(normal_clients, 1)
    
    print(f"Normal clients detected as malicious: {sum(len(clients) for clients in detection_results.values())}")
    for method, malicious_clients in detection_results.items():
        if malicious_clients:
            print(f"  {method}: {malicious_clients}")
    
    print("\n2. Testing Label Flipping Attack Detection")
    print("-" * 40)
    
    # Simulate label flipping attack
    simulator.set_client_attack("5", AttackType.LABEL_FLIPPING)
    simulator.set_client_attack("8", AttackType.LABEL_FLIPPING)
    
    # Apply attacks to client data
    attacked_clients = []
    for client_data in normal_clients:
        parameters, metrics, num_examples = client_data
        client_id = metrics["client_id"]
        
        if client_id in simulator.active_attacks:
            sim_parameters, sim_metrics = simulator.simulate_attack(client_id, parameters, metrics)
            attacked_clients.append((sim_parameters, sim_metrics, num_examples))
        else:
            attacked_clients.append(client_data)
    
    detection_results = detector.detect_attacks(attacked_clients, 2)
    
    print(f"Attacked clients detected as malicious: {sum(len(clients) for clients in detection_results.values())}")
    for method, malicious_clients in detection_results.items():
        if malicious_clients:
            print(f"  {method}: {malicious_clients}")
    
    print("\n3. Testing Byzantine Attack Detection")
    print("-" * 40)
    
    # Clear previous attacks and set Byzantine attacks
    simulator.clear_all_attacks()
    simulator.set_client_attack("3", AttackType.BYZANTINE)
    simulator.set_client_attack("7", AttackType.BYZANTINE)
    
    # Apply attacks
    byzantine_clients = []
    for client_data in normal_clients:
        parameters, metrics, num_examples = client_data
        client_id = metrics["client_id"]
        
        if client_id in simulator.active_attacks:
            sim_parameters, sim_metrics = simulator.simulate_attack(client_id, parameters, metrics)
            byzantine_clients.append((sim_parameters, sim_metrics, num_examples))
        else:
            byzantine_clients.append(client_data)
    
    detection_results = detector.detect_attacks(byzantine_clients, 3)
    
    print(f"Byzantine clients detected as malicious: {sum(len(clients) for clients in detection_results.values())}")
    for method, malicious_clients in detection_results.items():
        if malicious_clients:
            print(f"  {method}: {malicious_clients}")
    
    print("\n4. Testing Multiple Attack Types")
    print("-" * 40)
    
    # Set different attack types for different clients
    simulator.clear_all_attacks()
    simulator.set_client_attack("2", AttackType.GRADIENT_POISONING)
    simulator.set_client_attack("4", AttackType.MODEL_POISONING)
    simulator.set_client_attack("6", AttackType.SIGN_FLIPPING)
    simulator.set_client_attack("9", AttackType.SCALING_ATTACK)
    
    # Apply attacks
    mixed_attacks = []
    for client_data in normal_clients:
        parameters, metrics, num_examples = client_data
        client_id = metrics["client_id"]
        
        if client_id in simulator.active_attacks:
            sim_parameters, sim_metrics = simulator.simulate_attack(client_id, parameters, metrics)
            mixed_attacks.append((sim_parameters, sim_metrics, num_examples))
        else:
            mixed_attacks.append(client_data)
    
    detection_results = detector.detect_attacks(mixed_attacks, 4)
    
    print(f"Mixed attack clients detected as malicious: {sum(len(clients) for clients in detection_results.values())}")
    for method, malicious_clients in detection_results.items():
        if malicious_clients:
            print(f"  {method}: {malicious_clients}")
    
    print("\n5. Attack Summary")
    print("-" * 40)
    
    attack_summary = detector.get_attack_summary()
    print(f"Total attacks detected: {attack_summary['total_attacks']}")
    print(f"Malicious clients: {attack_summary['malicious_clients']}")
    print(f"Attack scores: {attack_summary['attack_scores']}")
    
    simulator_status = simulator.get_attack_status()
    print(f"Active attacks: {simulator_status['active_attacks']}")
    
    print("\n6. Performance Analysis")
    print("-" * 40)
    
    # Test detection performance over multiple rounds
    detector.reset_detection()
    
    total_detected = 0
    total_attacks = 0
    
    for round_num in range(1, 6):
        # Randomly select clients to attack
        num_attacks = random.randint(0, 3)
        attacked_clients = random.sample(range(1, 11), num_attacks)
        
        simulator.clear_all_attacks()
        for client_id in attacked_clients:
            attack_type = random.choice(list(AttackType))
            simulator.set_client_attack(str(client_id), attack_type)
        
        # Create client data with attacks
        client_data = create_sample_client_data(10)
        attacked_data = []
        
        for client_data_item in client_data:
            parameters, metrics, num_examples = client_data_item
            client_id = metrics["client_id"]
            
            if client_id in simulator.active_attacks:
                sim_parameters, sim_metrics = simulator.simulate_attack(client_id, parameters, metrics)
                attacked_data.append((sim_parameters, sim_metrics, num_examples))
                total_attacks += 1
            else:
                attacked_data.append(client_data_item)
        
        # Detect attacks
        detection_results = detector.detect_attacks(attacked_data, round_num)
        round_detected = sum(len(clients) for clients in detection_results.values())
        total_detected += round_detected
        
        print(f"Round {round_num}: {round_detected}/{num_attacks} attacks detected")
    
    detection_rate = (total_detected / total_attacks * 100) if total_attacks > 0 else 0
    print(f"\nOverall detection rate: {detection_rate:.1f}% ({total_detected}/{total_attacks})")
    
    print("\n✅ Attack detection testing completed!")

def test_attack_types():
    """Test different attack types individually"""
    print("\n🔬 Testing Individual Attack Types")
    print("=" * 60)
    
    simulator = DataPoisoningSimulator(attack_probability=0.0)
    
    # Create sample data
    coef = np.random.normal(0, 1, 10)
    intercept = np.random.normal(0, 1, 1)
    parameters = (coef, intercept)
    
    metrics = {
        "client_id": "test_client",
        "accuracy": 0.8,
        "loss": 0.3,
        "precision": 0.8,
        "recall": 0.8,
        "f1_score": 0.8,
        "is_malicious": False,
        "labels_flipped": 0
    }
    
    num_examples = 1000
    
    for attack_type in AttackType:
        print(f"\nTesting {attack_type.value}:")
        print("-" * 30)
        
        simulator.set_client_attack("test_client", attack_type)
        sim_parameters, sim_metrics = simulator.simulate_attack("test_client", parameters, metrics)
        
        print(f"  Original accuracy: {metrics['accuracy']:.3f}")
        print(f"  Attacked accuracy: {sim_metrics['accuracy']:.3f}")
        print(f"  Original loss: {metrics['loss']:.3f}")
        print(f"  Attacked loss: {sim_metrics['loss']:.3f}")
        print(f"  Attack detected: {sim_metrics.get('is_malicious', False)}")
        
        simulator.remove_client_attack("test_client")

if __name__ == "__main__":
    try:
        test_attack_detection()
        test_attack_types()
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()

