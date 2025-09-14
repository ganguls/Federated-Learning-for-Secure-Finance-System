"""
Integration Example for Data Poisoning Detection in Federated Learning
This example shows how to integrate the detection system into an existing FL system
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from federated_detection import FederatedDetectionSystem
from tabular_data_utils import preprocess_tabular_data, split_data_among_clients, poison_client_data
from tabular_model import create_tabular_model

def create_synthetic_lending_club_data(num_samples=5000, num_features=20, 
                                     good_credit_ratio=0.7, random_state=42):
    """
    Create synthetic Lending Club-like data for demonstration
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        good_credit_ratio: Ratio of good credit samples
        random_state: Random seed
    
    Returns:
        df: Synthetic dataset as DataFrame
    """
    np.random.seed(random_state)
    
    # Generate features (financial indicators)
    feature_names = [
        'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc', 'out_prncp',
        'total_pymnt', 'last_pymnt_amnt', 'collections_12_mths_ex_med',
        'mths_since_last_delinq', 'acc_now_delinq', 'tot_coll_amt',
        'tot_cur_bal'
    ]
    
    # Ensure we have enough feature names
    while len(feature_names) < num_features:
        feature_names.append(f'feature_{len(feature_names)}')
    
    feature_names = feature_names[:num_features]
    
    # Generate realistic financial data
    data = {}
    for i, feature in enumerate(feature_names):
        if 'amnt' in feature or 'bal' in feature or 'pymnt' in feature:
            # Amount features - log-normal distribution
            data[feature] = np.random.lognormal(mean=8, sigma=1, size=num_samples)
        elif 'rate' in feature or 'util' in feature or 'dti' in feature:
            # Rate features - beta distribution
            data[feature] = np.random.beta(2, 5, size=num_samples) * 100
        elif 'mths' in feature or 'yrs' in feature:
            # Time features - exponential distribution
            data[feature] = np.random.exponential(scale=12, size=num_samples)
        else:
            # Other features - normal distribution
            data[feature] = np.random.normal(0, 1, size=num_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target based on feature combinations
    # Good credit if: low debt ratio, high income, low delinquency
    debt_ratio = df['dti'] if 'dti' in df.columns else df.iloc[:, 4]
    income = df['annual_inc'] if 'annual_inc' in df.columns else df.iloc[:, 3]
    delinquency = df['delinq_2yrs'] if 'delinq_2yrs' in df.columns else df.iloc[:, 5]
    
    # Create target variable
    target = np.zeros(num_samples)
    good_credit_mask = (
        (debt_ratio < debt_ratio.quantile(0.6)) &
        (income > income.quantile(0.4)) &
        (delinquency < delinquency.quantile(0.7))
    )
    
    # Adjust to match desired ratio
    current_ratio = good_credit_mask.mean()
    if current_ratio > good_credit_ratio:
        # Need to flip some good to bad
        flip_indices = np.random.choice(
            np.where(good_credit_mask)[0],
            size=int((current_ratio - good_credit_ratio) * num_samples),
            replace=False
        )
        good_credit_mask[flip_indices] = False
    else:
        # Need to flip some bad to good
        flip_indices = np.random.choice(
            np.where(~good_credit_mask)[0],
            size=int((good_credit_ratio - current_ratio) * num_samples),
            replace=False
        )
        good_credit_mask[flip_indices] = True
    
    df['loan_status'] = good_credit_mask.astype(int)
    
    return df

def demonstrate_detection_system():
    """
    Demonstrate the complete detection system
    """
    print("=" * 60)
    print("FEDERATED LEARNING DATA POISONING DETECTION DEMO")
    print("=" * 60)
    
    # Step 1: Create synthetic data
    print("\n1. Creating synthetic Lending Club dataset...")
    df = create_synthetic_lending_club_data(num_samples=2000, num_features=15)
    print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"Good credit ratio: {df['loan_status'].mean():.2%}")
    
    # Save synthetic data
    df.to_csv('synthetic_lending_club.csv', index=False)
    print("Synthetic data saved to 'synthetic_lending_club.csv'")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    train_dataset, test_dataset, feature_columns, scaler = preprocess_tabular_data(
        'synthetic_lending_club.csv',
        target_column='loan_status',
        test_size=0.2
    )
    
    # Step 3: Split data among clients
    print("\n3. Setting up federated learning...")
    num_clients = 20
    mal_percentage = 0.3  # 30% malicious clients
    
    user_groups, true_attackers = split_data_among_clients(
        train_dataset, 
        num_clients=num_clients,
        mal_percentage=mal_percentage,
        target_honest=1,  # Good credit
        target_mal=0      # Bad credit
    )
    
    print(f"Created {num_clients} clients with {len(true_attackers)} malicious clients")
    print(f"True attackers: {sorted(true_attackers)}")
    
    # Step 4: Poison malicious clients' data
    print("\n4. Poisoning malicious clients' data...")
    client_data_dict = {}
    
    for client_id in range(num_clients):
        client_indices = user_groups[client_id]
        client_data = []
        
        for idx in client_indices:
            features, label = train_dataset[idx]
            client_data.append((features, label))
        
        # Poison data for malicious clients
        if client_id in true_attackers:
            poisoned_data = poison_client_data(
                train_dataset, 
                client_indices, 
                target_honest=1, 
                target_mal=0
            )
            # Convert back to list format
            client_data = []
            for idx in client_indices:
                if idx in poisoned_data:
                    features, label = poisoned_data[idx]
                    client_data.append((features, label))
                else:
                    features, label = train_dataset[idx]
                    client_data.append((features, label))
        
        client_data_dict[client_id] = client_data
    
    # Step 5: Initialize detection system
    print("\n5. Initializing detection system...")
    detection_system = FederatedDetectionSystem(
        input_dim=len(feature_columns),
        num_classes=2,
        model_type='credit_risk',
        detection_method='kmeans',
        ldp_epsilon=1.0
    )
    
    # Step 6: Run federated training with detection
    print("\n6. Running federated training with detection...")
    num_rounds = 10
    clients_per_round = 10
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Select clients for this round
        selected_clients = np.random.choice(
            num_clients, 
            size=clients_per_round, 
            replace=False
        ).tolist()
        
        # Run federated training round
        round_results = detection_system.federated_training_round(
            client_data_dict=client_data_dict,
            selected_clients=selected_clients,
            local_epochs=3,
            learning_rate=0.01
        )
        
        print(f"Selected clients: {selected_clients}")
        print(f"Clean clients: {round_results['clean_clients']}")
        print(f"Detected attackers: {round_results['attackers']}")
        print(f"Average loss: {round_results['avg_loss']:.4f}")
    
    # Step 7: Evaluate detection performance
    print("\n7. Evaluating detection performance...")
    detection_metrics = detection_system.get_detection_metrics(true_attackers)
    
    print("\nDetection Performance:")
    print(f"Detection Accuracy: {detection_metrics['detection_accuracy']:.4f}")
    print(f"Detection Precision: {detection_metrics['detection_precision']:.4f}")
    print(f"Detection Recall: {detection_metrics['detection_recall']:.4f}")
    print(f"Detection F1-Score: {detection_metrics['detection_f1']:.4f}")
    print(f"False Positives: {detection_metrics['false_positives']}")
    print(f"False Negatives: {detection_metrics['false_negatives']}")
    
    # Step 8: Evaluate model performance
    print("\n8. Evaluating model performance...")
    test_data = [(test_dataset[i][0], test_dataset[i][1]) for i in range(len(test_dataset))]
    model_metrics = detection_system.evaluate_model(test_data)
    
    print("\nModel Performance:")
    print(f"Accuracy: {model_metrics['accuracy']:.4f}")
    print(f"Precision: {model_metrics['precision']:.4f}")
    print(f"Recall: {model_metrics['recall']:.4f}")
    print(f"F1-Score: {model_metrics['f1_score']:.4f}")
    print(f"Loss: {model_metrics['loss']:.4f}")
    
    # Step 9: Generate summary
    print("\n9. Detection Summary:")
    summary = detection_system.get_detection_summary()
    print(f"Total rounds: {summary['total_rounds']}")
    print(f"Total attackers detected: {summary['total_attackers_detected']}")
    print(f"Average attackers per round: {summary['avg_attackers_per_round']:.2f}")
    
    return detection_system, detection_metrics, model_metrics

def integration_guide():
    """
    Print integration guide for existing FL systems
    """
    print("\n" + "=" * 60)
    print("INTEGRATION GUIDE FOR EXISTING FL SYSTEMS")
    print("=" * 60)
    
    print("""
To integrate this detection system into your existing FL system:

1. COPY THESE FILES:
   - detection_utils.py
   - tabular_model.py
   - tabular_data_utils.py
   - federated_detection.py

2. MODIFY YOUR FL TRAINING LOOP:
   
   # Initialize detection system
   detection_system = FederatedDetectionSystem(
       input_dim=your_feature_dim,
       num_classes=your_num_classes,
       detection_method='kmeans'  # or 'fixed_percentage', 'z_score'
   )
   
   # In your training loop:
   for round_num in range(num_rounds):
       # Select clients
       selected_clients = select_clients()
       
       # Run federated training with detection
       round_results = detection_system.federated_training_round(
           client_data_dict=your_client_data,
           selected_clients=selected_clients,
           local_epochs=your_local_epochs,
           learning_rate=your_learning_rate
       )
       
       # Get clean clients and attackers
       clean_clients = round_results['clean_clients']
       attackers = round_results['attackers']

3. REQUIREMENTS FOR YOUR SYSTEM:
   - Tabular data format (features, labels)
   - PyTorch models with state_dict support
   - Client data as List[Tuple[features, labels]]
   - Support for fake training (loss computation without updates)

4. DETECTION METHODS AVAILABLE:
   - 'kmeans': K-means clustering on loss values (recommended)
   - 'fixed_percentage': Remove top X% clients by loss
   - 'z_score': Remove clients with loss beyond z-score threshold

5. PRIVACY PROTECTION:
   - Local Differential Privacy (LDP) applied to loss values
   - Adjustable epsilon parameter for privacy-utility tradeoff
   - Lower epsilon = more private, higher epsilon = better detection

6. EVALUATION:
   - detection_system.get_detection_metrics(true_attackers)
   - detection_system.evaluate_model(test_data)
   - detection_system.get_detection_summary()
""")

if __name__ == "__main__":
    # Run the demonstration
    detection_system, detection_metrics, model_metrics = demonstrate_detection_system()
    
    # Print integration guide
    integration_guide()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Files created:")
    print("- synthetic_lending_club.csv (synthetic dataset)")
    print("- detection_utils.py (detection algorithms)")
    print("- tabular_model.py (neural network models)")
    print("- tabular_data_utils.py (data processing)")
    print("- federated_detection.py (main detection system)")
    print("- integration_example.py (this file)")
