#!/usr/bin/env python3
"""
Demonstration script for the Data Poisoning Attack Detection System
Shows how to use the system with a simple example
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent / "server"))

from attack_detection import DataPoisoningDetector
from attack_simulator import DataPoisoningSimulator, AttackType

def create_demo_data():
    """Create demonstration data with normal and malicious clients"""
    print("🎭 Creating demonstration data...")
    
    # Normal clients (5 clients)
    normal_clients = []
    for i in range(1, 6):
        coef = np.random.normal(0, 1, 10)  # 10 features
        intercept = np.random.normal(0, 1, 1)
        parameters = (coef, intercept)
        
        metrics = {
            "client_id": str(i),
            "accuracy": np.random.uniform(0.75, 0.90),
            "loss": np.random.uniform(0.1, 0.4),
            "precision": np.random.uniform(0.75, 0.90),
            "recall": np.random.uniform(0.75, 0.90),
            "f1_score": np.random.uniform(0.75, 0.90),
            "is_malicious": False,
            "labels_flipped": 0
        }
        
        normal_clients.append((parameters, metrics, 1000))
    
    # Malicious clients (3 clients with different attack types)
    malicious_clients = []
    
    # Client 6: Label flipping attack
    coef = np.random.normal(0, 1, 10)
    intercept = np.random.normal(0, 1, 1)
    parameters = (coef, intercept)
    
    metrics = {
        "client_id": "6",
        "accuracy": 0.3,  # Low accuracy due to attack
        "loss": 1.2,      # High loss due to attack
        "precision": 0.3,
        "recall": 0.3,
        "f1_score": 0.3,
        "is_malicious": True,
        "labels_flipped": 500
    }
    
    malicious_clients.append((parameters, metrics, 1000))
    
    # Client 7: Byzantine attack (extreme parameters)
    coef = np.full(10, 100.0)  # Extreme values
    intercept = np.full(1, 100.0)
    parameters = (coef, intercept)
    
    metrics = {
        "client_id": "7",
        "accuracy": 0.1,  # Very low accuracy
        "loss": 2.5,      # Very high loss
        "precision": 0.1,
        "recall": 0.1,
        "f1_score": 0.1,
        "is_malicious": True,
        "labels_flipped": 0
    }
    
    malicious_clients.append((parameters, metrics, 1000))
    
    # Client 8: Gradient poisoning (noisy parameters)
    coef = np.random.normal(0, 5, 10)  # High noise
    intercept = np.random.normal(0, 5, 1)
    parameters = (coef, intercept)
    
    metrics = {
        "client_id": "8",
        "accuracy": 0.4,  # Moderate accuracy
        "loss": 0.8,      # Moderate loss
        "precision": 0.4,
        "recall": 0.4,
        "f1_score": 0.4,
        "is_malicious": True,
        "labels_flipped": 0
    }
    
    malicious_clients.append((parameters, metrics, 1000))
    
    return normal_clients + malicious_clients

def demonstrate_detection():
    """Demonstrate the attack detection system"""
    print("🛡️ Data Poisoning Attack Detection Demonstration")
    print("=" * 60)
    
    # Initialize the detection system
    detector = DataPoisoningDetector(
        anomaly_threshold=0.1,
        clustering_threshold=0.3,
        statistical_threshold=2.0,
        history_window=5
    )
    
    # Create demonstration data
    client_data = create_demo_data()
    
    print(f"\n📊 Created {len(client_data)} clients:")
    print("   - 5 Normal clients (clients 1-5)")
    print("   - 3 Malicious clients (clients 6-8)")
    print("     * Client 6: Label flipping attack")
    print("     * Client 7: Byzantine attack (extreme parameters)")
    print("     * Client 8: Gradient poisoning attack")
    
    # Run detection
    print(f"\n🔍 Running attack detection...")
    detection_results = detector.detect_attacks(client_data, 1)
    
    # Display results
    print(f"\n📈 Detection Results:")
    print("-" * 30)
    
    total_detected = 0
    for method, malicious_clients in detection_results.items():
        if malicious_clients:
            print(f"  {method}: {malicious_clients}")
            total_detected += len(malicious_clients)
        else:
            print(f"  {method}: No malicious clients detected")
    
    print(f"\n🎯 Total malicious clients detected: {total_detected}")
    print(f"   Expected: 3 (clients 6, 7, 8)")
    
    # Show attack summary
    attack_summary = detector.get_attack_summary()
    print(f"\n📋 Attack Summary:")
    print(f"   Total attacks: {attack_summary['total_attacks']}")
    print(f"   Malicious clients: {attack_summary['malicious_clients']}")
    print(f"   Attack scores: {attack_summary['attack_scores']}")
    
    return detection_results

def demonstrate_simulation():
    """Demonstrate attack simulation"""
    print(f"\n🎭 Attack Simulation Demonstration")
    print("=" * 60)
    
    # Initialize simulator
    simulator = DataPoisoningSimulator(attack_probability=0.0)
    
    # Create normal client data
    coef = np.random.normal(0, 1, 10)
    intercept = np.random.normal(0, 1, 1)
    parameters = (coef, intercept)
    
    metrics = {
        "client_id": "demo_client",
        "accuracy": 0.8,
        "loss": 0.3,
        "precision": 0.8,
        "recall": 0.8,
        "f1_score": 0.8,
        "is_malicious": False,
        "labels_flipped": 0
    }
    
    print(f"📊 Original client metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Loss: {metrics['loss']:.3f}")
    print(f"   Malicious: {metrics['is_malicious']}")
    
    # Test different attack types
    attack_types = [
        AttackType.LABEL_FLIPPING,
        AttackType.BYZANTINE,
        AttackType.GRADIENT_POISONING,
        AttackType.MODEL_POISONING
    ]
    
    for attack_type in attack_types:
        print(f"\n🎯 Testing {attack_type.value} attack:")
        print("-" * 40)
        
        # Set attack
        simulator.set_client_attack("demo_client", attack_type)
        
        # Simulate attack
        sim_parameters, sim_metrics = simulator.simulate_attack("demo_client", parameters, metrics)
        
        print(f"   Accuracy: {metrics['accuracy']:.3f} → {sim_metrics['accuracy']:.3f}")
        print(f"   Loss: {metrics['loss']:.3f} → {sim_metrics['loss']:.3f}")
        print(f"   Malicious: {metrics['is_malicious']} → {sim_metrics['is_malicious']}")
        
        if 'labels_flipped' in sim_metrics:
            print(f"   Labels flipped: {sim_metrics['labels_flipped']}")
        
        # Remove attack
        simulator.remove_client_attack("demo_client")
    
    # Show simulator status
    status = simulator.get_attack_status()
    print(f"\n📊 Simulator Status:")
    print(f"   Active attacks: {status['active_attacks']}")
    print(f"   Attack probability: {status['attack_probability']}")

def demonstrate_dashboard_integration():
    """Demonstrate dashboard integration"""
    print(f"\n🖥️ Dashboard Integration")
    print("=" * 60)
    
    print("📋 Available API endpoints:")
    print("   GET  /api/security/status          - Get security status")
    print("   POST /api/security/attack/simulate - Simulate attack")
    print("   POST /api/security/attack/remove   - Remove attack")
    print("   POST /api/security/attack/clear    - Clear all attacks")
    print("   GET  /api/security/statistics      - Get statistics")
    print("   GET  /api/security/report          - Get security report")
    
    print(f"\n🎮 Dashboard features:")
    print("   - Security tab with real-time monitoring")
    print("   - Attack simulation controls")
    print("   - Security status indicators")
    print("   - Active attacks list")
    print("   - Security event logging")
    
    print(f"\n🚀 To use the dashboard:")
    print("   1. Start the server: python server/server.py")
    print("   2. Start the dashboard: python dashboard/app.py")
    print("   3. Open http://localhost:5000")
    print("   4. Navigate to the 'Security' tab")
    print("   5. Use the attack simulation controls")

def main():
    """Main demonstration function"""
    print("🛡️ Data Poisoning Attack Detection System")
    print("   Comprehensive Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate detection
        detection_results = demonstrate_detection()
        
        # Demonstrate simulation
        demonstrate_simulation()
        
        # Demonstrate dashboard integration
        demonstrate_dashboard_integration()
        
        print(f"\n✅ Demonstration completed successfully!")
        print(f"\n📚 For more information, see ATTACK_DETECTION_README.md")
        print(f"🧪 For testing, run: python test_attack_detection.py")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

