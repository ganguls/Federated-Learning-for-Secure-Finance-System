# Data Poisoning Detection for Federated Learning on Tabular Data

This package provides a comprehensive solution for detecting data poisoning attacks in federated learning systems, specifically designed for tabular datasets like the Lending Club credit risk assessment dataset.

## 🚀 Quick Start

```python
from federated_detection import FederatedDetectionSystem
from tabular_data_utils import preprocess_tabular_data, split_data_among_clients

# 1. Preprocess your tabular data
train_dataset, test_dataset, feature_columns, scaler = preprocess_tabular_data(
    'your_data.csv',
    target_column='target',
    test_size=0.2
)

# 2. Split data among clients
user_groups, true_attackers = split_data_among_clients(
    train_dataset, 
    num_clients=20,
    mal_percentage=0.3
)

# 3. Initialize detection system
detection_system = FederatedDetectionSystem(
    input_dim=len(feature_columns),
    num_classes=2,
    detection_method='kmeans'
)

# 4. Run federated training with detection
for round_num in range(10):
    selected_clients = select_clients()
    round_results = detection_system.federated_training_round(
        client_data_dict=client_data,
        selected_clients=selected_clients
    )
```

## 📁 Files Overview

### Core Detection Files

| File | Description |
|------|-------------|
| `detection_utils.py` | Core detection algorithms (K-means, LDP, metrics) |
| `tabular_model.py` | Neural network models for tabular data |
| `tabular_data_utils.py` | Data preprocessing and client splitting utilities |
| `federated_detection.py` | Main detection system class |
| `integration_example.py` | Complete working example |

### Key Components

#### 1. Detection Algorithms (`detection_utils.py`)
- **K-means Clustering**: Groups clients based on loss patterns
- **Local Differential Privacy**: Protects client privacy during detection
- **Multiple Detection Methods**: K-means, fixed percentage, z-score
- **Performance Metrics**: Accuracy, precision, recall, F1-score

#### 2. Tabular Models (`tabular_model.py`)
- **TabularNet**: Generic neural network for tabular data
- **CreditRiskNet**: Specialized model for credit risk assessment
- **Configurable Architecture**: Adjustable hidden layers and dropout
- **Xavier Initialization**: Proper weight initialization

#### 3. Data Processing (`tabular_data_utils.py`)
- **Preprocessing**: Missing value handling, normalization
- **Client Splitting**: Distribute data among federated clients
- **Data Poisoning**: Label flipping for malicious clients
- **Validation**: Data integrity checks

#### 4. Detection System (`federated_detection.py`)
- **FederatedDetectionSystem**: Main class for integration
- **Two-Phase Training**: Fake training for detection, real training for updates
- **Automatic Detection**: Integrated into federated learning loop
- **Performance Tracking**: Comprehensive metrics and history

## 🔧 Integration Requirements

### System Requirements
- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Scikit-learn
- Matplotlib (for visualization)

### Data Format Requirements
```python
# Your data must be in this format:
client_data = [
    (features_tensor, label_tensor),  # Sample 1
    (features_tensor, label_tensor),  # Sample 2
    # ... more samples
]
```

### Model Requirements
- PyTorch model with `state_dict()` support
- Tabular data architecture (fully connected layers)
- Consistent loss computation across clients

## 🛡️ Detection Methods

### 1. K-means Clustering (Recommended)
```python
detection_system = FederatedDetectionSystem(
    detection_method='kmeans',
    ldp_epsilon=1.0
)
```
- Groups clients based on loss patterns
- Assumes attackers have different loss distributions
- Works well with various attack patterns

### 2. Fixed Percentage
```python
detection_system = FederatedDetectionSystem(
    detection_method='fixed_percentage'
)
```
- Removes top X% of clients by loss value
- Simple but effective for known attack ratios
- Less adaptive than K-means

### 3. Z-score Threshold
```python
detection_system = FederatedDetectionSystem(
    detection_method='z_score'
)
```
- Removes clients with loss beyond z-score threshold
- Good for detecting outliers
- Sensitive to threshold selection

## 🔒 Privacy Protection

### Local Differential Privacy (LDP)
```python
# Adjust privacy level
detection_system = FederatedDetectionSystem(
    ldp_epsilon=0.5,      # Lower = more private
    ldp_sensitivity=0.001  # Sensitivity parameter
)
```

**Privacy Levels:**
- `epsilon=0.1`: High privacy, lower detection accuracy
- `epsilon=1.0`: Balanced privacy and accuracy
- `epsilon=10.0`: Low privacy, high detection accuracy

## 📊 Usage Examples

### Basic Integration
```python
from federated_detection import FederatedDetectionSystem

# Initialize system
detection_system = FederatedDetectionSystem(
    input_dim=20,
    num_classes=2,
    detection_method='kmeans'
)

# Run federated training
for round_num in range(num_rounds):
    round_results = detection_system.federated_training_round(
        client_data_dict=client_data,
        selected_clients=selected_clients
    )
    
    # Get results
    clean_clients = round_results['clean_clients']
    attackers = round_results['attackers']
```

### Advanced Configuration
```python
# Custom model architecture
detection_system = FederatedDetectionSystem(
    input_dim=50,
    num_classes=3,
    model_type='custom',
    detection_method='kmeans',
    ldp_epsilon=2.0,
    ldp_sensitivity=0.0001
)

# Custom model parameters
model = create_tabular_model(
    input_dim=50,
    num_classes=3,
    model_type='custom',
    hidden_dims=[256, 128, 64],
    dropout_rate=0.4
)
```

### Evaluation
```python
# Evaluate detection performance
detection_metrics = detection_system.get_detection_metrics(true_attackers)
print(f"Detection Accuracy: {detection_metrics['detection_accuracy']:.4f}")

# Evaluate model performance
model_metrics = detection_system.evaluate_model(test_data)
print(f"Model Accuracy: {model_metrics['accuracy']:.4f}")

# Get summary
summary = detection_system.get_detection_summary()
print(f"Total attackers detected: {summary['total_attackers_detected']}")
```

## 🎯 Performance Optimization

### Memory Optimization
```python
# Use smaller batch sizes for large datasets
client_data = create_client_dataloader(
    dataset, client_indices, 
    batch_size=16  # Smaller batch size
)
```

### Detection Accuracy
```python
# Tune LDP parameters for better detection
detection_system = FederatedDetectionSystem(
    ldp_epsilon=1.5,      # Increase for better detection
    ldp_sensitivity=0.0005  # Decrease for more sensitivity
)
```

### Model Performance
```python
# Use appropriate model architecture
model = create_tabular_model(
    input_dim=feature_dim,
    hidden_dims=[512, 256, 128],  # Adjust based on data size
    dropout_rate=0.3
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Low Detection Accuracy**
   - Increase `ldp_epsilon` value
   - Try different detection methods
   - Check data quality and preprocessing

2. **High False Positive Rate**
   - Decrease `ldp_epsilon` value
   - Use `fixed_percentage` method
   - Adjust detection thresholds

3. **Memory Issues**
   - Reduce batch sizes
   - Use smaller model architectures
   - Process clients in smaller batches

4. **Model Convergence Issues**
   - Adjust learning rates
   - Increase local training epochs
   - Check data normalization

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check detection history
detection_history = detection_system.detection_history
for record in detection_history:
    print(f"Round {record['round']}: {len(record['attackers'])} attackers detected")
```

## 📈 Results and Metrics

### Detection Metrics
- **Detection Accuracy**: Overall correctness of detection
- **Precision**: Ratio of correctly detected attackers
- **Recall**: Ratio of actual attackers detected
- **F1-Score**: Harmonic mean of precision and recall

### Model Metrics
- **Accuracy**: Overall model performance
- **Loss**: Training/validation loss
- **Per-class Metrics**: Precision, recall for each class

### Privacy Metrics
- **LDP Privacy Budget**: Total privacy cost
- **Sensitivity**: Maximum change in loss values
- **Noise Level**: Amount of added noise

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original research on federated learning data poisoning
- Lending Club dataset for credit risk assessment
- PyTorch and scikit-learn communities

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the integration examples

---

**Note**: This detection system is specifically designed for tabular data and may not work optimally with image or text data without modifications.
