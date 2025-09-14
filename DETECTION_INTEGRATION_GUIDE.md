# 🔍 Data Poisoning Detection Integration Guide

## **Overview**

This guide explains how the data poisoning detection system has been integrated into your Flower-based federated learning system. The integration provides real-time detection of malicious clients using advanced PyTorch-based algorithms while maintaining compatibility with your existing scikit-learn clients.

## **🏗️ Integration Architecture**

### **System Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   FL Server     │    │   FL Clients    │
│   (Port 5000)   │◄──►│   (Port 8080)   │◄──►│   (Ports N/A)  │
│                 │    │                 │    │                 │
│ • Demo Tab      │    │ • Enhanced      │    │ • Scikit-learn  │
│ • Run Detection │    │   Strategy      │    │ • LogisticReg   │
│ • Real Results  │    │ • Detection     │    │ • Real Data     │
│                 │    │   Adapter       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Key Integration Points**

1. **Server-Side Detection**: Enhanced `LoanServerStrategy` with PyTorch-based detection
2. **Data Conversion**: Automatic conversion from scikit-learn to PyTorch format
3. **Dashboard Integration**: Real-time detection results display
4. **API Endpoints**: RESTful API for detection control and monitoring

## **📁 File Structure**

### **New Files Added**

```
server/
├── detection_adapter.py          # Core detection adapter
├── enhanced_server_strategy.py   # Enhanced server strategy
└── server.py                     # Original server (unchanged)

dashboard/
├── detection_api.py              # Detection API endpoints
├── app.py                        # Updated with detection integration
└── templates/index.html          # Updated with enhanced detection UI

data_poisoning_detection/         # Your existing detection system
├── detection_utils.py
├── federated_detection.py
├── tabular_model.py
└── tabular_data_utils.py
```

## **🔧 Configuration**

### **Detection Configuration**

The detection system can be configured through the dashboard or by modifying the server configuration:

```python
detection_config = {
    'enabled': True,                    # Enable/disable detection
    'method': 'kmeans',                 # Detection method
    'ldp_epsilon': 1.0,                 # Privacy parameter
    'ldp_sensitivity': 0.001,           # LDP sensitivity
    'input_dim': 20,                    # Number of features
    'defense_threshold': 0.3            # Defense threshold
}
```

### **Available Detection Methods**

1. **K-means Clustering** (Recommended)
   - Groups clients based on parameter similarity
   - Identifies outliers as potentially malicious
   - Works well with various attack types

2. **Fixed Percentage**
   - Removes top X% of clients by loss value
   - Simple but effective for known attack ratios
   - Configurable percentage threshold

3. **Z-score Threshold**
   - Removes clients with loss beyond z-score threshold
   - Good for detecting statistical outliers
   - Configurable threshold value

## **🚀 Usage Instructions**

### **1. Starting the Enhanced System**

#### **Option A: Use Enhanced Server Strategy**

```python
# In your server startup script
from server.enhanced_server_strategy import create_enhanced_strategy

# Create enhanced strategy with detection
strategy = create_enhanced_strategy(
    ca_url="http://ca:9000",
    detection_config={
        'enabled': True,
        'method': 'kmeans',
        'ldp_epsilon': 1.0
    },
    min_fit_clients=10,
    min_evaluate_clients=10
)

# Start server with enhanced strategy
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=config,
    strategy=strategy
)
```

#### **Option B: Modify Existing Server**

```python
# In server/server.py, replace LoanServerStrategy with EnhancedLoanServerStrategy
from enhanced_server_strategy import EnhancedLoanServerStrategy

# Update strategy initialization
strategy = EnhancedLoanServerStrategy(
    ca_url=ca_url if enable_certificates else None,
    detection_config={
        'enabled': True,
        'method': 'kmeans',
        'ldp_epsilon': 1.0
    },
    min_fit_clients=min_clients,
    min_evaluate_clients=min_clients,
    min_available_clients=min_clients,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)
```

### **2. Running Detection from Dashboard**

1. **Start the system**:
   ```bash
   # Start FL server
   python server/server.py
   
   # Start dashboard
   python dashboard/app.py
   ```

2. **Access the dashboard**: Navigate to `http://localhost:5000`

3. **Go to Demo tab**: Click on the "Demo" tab in the dashboard

4. **Run Detection**: Click the "Run Detection" button to execute detection on real FL results

5. **View Results**: Detection results will be displayed in real-time with:
   - Detected malicious clients
   - Detection accuracy metrics
   - Detection method used
   - Execution time

### **3. API Usage**

#### **Run Detection**
```bash
curl -X POST http://localhost:5000/api/detection/run_enhanced \
  -H "Content-Type: application/json" \
  -d '{"use_cached": false}'
```

#### **Get Detection Status**
```bash
curl http://localhost:5000/api/detection/status
```

#### **Update Configuration**
```bash
curl -X POST http://localhost:5000/api/detection/config \
  -H "Content-Type: application/json" \
  -d '{"method": "z_score", "ldp_epsilon": 2.0}'
```

## **🔍 Detection Process**

### **Step-by-Step Detection Flow**

1. **Client Training**: Clients train locally with scikit-learn LogisticRegression
2. **Parameter Collection**: Server collects client parameters and metrics
3. **Data Conversion**: Parameters converted from scikit-learn to PyTorch format
4. **LDP Application**: Privacy-preserving noise added to parameters (if enabled)
5. **Detection Algorithm**: Selected detection method applied to converted data
6. **Malicious Client Identification**: Clients flagged as malicious based on detection results
7. **Defense Application**: Malicious clients filtered from aggregation
8. **Results Logging**: Detection results saved and displayed in dashboard

### **Data Flow Diagram**

```
Client Results → Parameter Extraction → Format Conversion → LDP Noise → Detection Algorithm → Malicious Clients → Filtering → Aggregation
     ↓                    ↓                    ↓              ↓              ↓                    ↓              ↓
[coef_, intercept_] → [Flatten Arrays] → [PyTorch Format] → [Laplace] → [K-means/Z-score] → [Client IDs] → [Filter] → [FedAvg]
```

## **📊 Monitoring and Metrics**

### **Detection Metrics**

The system tracks comprehensive detection metrics:

- **Accuracy**: Overall detection correctness
- **Precision**: Ratio of correctly detected malicious clients
- **Recall**: Ratio of actual malicious clients detected
- **F1-Score**: Harmonic mean of precision and recall
- **False Positives**: Incorrectly flagged benign clients
- **False Negatives**: Missed malicious clients

### **Performance Metrics**

- **Detection Time**: Time taken for detection algorithm
- **Total Clients**: Number of clients analyzed
- **Detection Method**: Algorithm used for detection
- **LDP Privacy Budget**: Privacy cost of detection

### **Logging**

Detection results are logged to:
- **Console**: Real-time detection logs
- **JSON Files**: `enhanced_training_metrics.json`
- **Dashboard**: Real-time display in Demo tab

## **⚙️ Advanced Configuration**

### **Privacy Settings**

```python
# High Privacy (lower detection accuracy)
detection_config = {
    'ldp_epsilon': 0.1,
    'ldp_sensitivity': 0.0001
}

# Balanced Privacy/Accuracy
detection_config = {
    'ldp_epsilon': 1.0,
    'ldp_sensitivity': 0.001
}

# High Accuracy (lower privacy)
detection_config = {
    'ldp_epsilon': 10.0,
    'ldp_sensitivity': 0.01
}
```

### **Detection Method Tuning**

```python
# K-means with custom parameters
detection_config = {
    'method': 'kmeans',
    'kmeans_clusters': 3,  # Number of clusters
    'kmeans_random_state': 42
}

# Z-score with custom threshold
detection_config = {
    'method': 'z_score',
    'z_score_threshold': 2.5  # Standard deviations
}

# Fixed percentage with custom ratio
detection_config = {
    'method': 'fixed_percentage',
    'fixed_percentage': 0.15  # Remove top 15%
}
```

## **🔧 Troubleshooting**

### **Common Issues**

1. **Detection Not Working**
   - Check if detection is enabled in configuration
   - Verify detection adapter is initialized
   - Check server logs for errors

2. **Low Detection Accuracy**
   - Try different detection methods
   - Adjust LDP epsilon parameter
   - Check data quality and preprocessing

3. **Performance Issues**
   - Reduce detection frequency
   - Use lighter detection methods
   - Check system resources

4. **Client Compatibility**
   - Ensure clients return proper parameter format
   - Check client data preprocessing
   - Verify feature dimensions match

### **Debug Mode**

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in detection adapter
detection_adapter = FLDetectionAdapter(
    enable_logging=True
)
```

## **📈 Performance Optimization**

### **Detection Frequency**

```python
# Run detection every N rounds
if server_round % 5 == 0:  # Every 5 rounds
    run_detection()
```

### **Memory Optimization**

```python
# Use smaller batch sizes for large datasets
detection_config = {
    'batch_size': 16,
    'max_clients_per_detection': 50
}
```

### **Caching**

```python
# Cache detection results
detection_results = cache.get('last_detection')
if not detection_results:
    detection_results = run_detection()
    cache.set('last_detection', detection_results)
```

## **🔄 Integration with Existing Workflows**

### **Docker Integration**

The detection system is fully compatible with your existing Docker setup:

```dockerfile
# Add detection dependencies to Dockerfile
RUN pip install torch torchvision scipy
```

### **Kubernetes Integration**

Detection configuration can be set via environment variables:

```yaml
env:
- name: DETECTION_ENABLED
  value: "true"
- name: DETECTION_METHOD
  value: "kmeans"
- name: LDP_EPSILON
  value: "1.0"
```

### **Monitoring Integration**

Detection metrics can be integrated with your existing monitoring:

```python
# Export metrics to Prometheus
from prometheus_client import Counter, Histogram

detection_counter = Counter('fl_detection_total', 'Total detections')
detection_accuracy = Histogram('fl_detection_accuracy', 'Detection accuracy')
```

## **📚 API Reference**

### **Detection API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detection/run_enhanced` | POST | Run detection on real FL results |
| `/api/detection/status` | GET | Get detection status |
| `/api/detection/config` | GET/POST | Get/update configuration |
| `/api/detection/history` | GET | Get detection history |
| `/api/detection/summary` | GET | Get detection summary |
| `/api/detection/toggle` | POST | Toggle detection on/off |
| `/api/detection/methods` | GET | Get available methods |
| `/api/detection/metrics` | GET | Get detailed metrics |

### **Request/Response Examples**

#### **Run Detection**
```json
// Request
POST /api/detection/run_enhanced
{
  "use_cached": false
}

// Response
{
  "success": true,
  "detected_malicious": ["client_1", "client_3"],
  "detection_metrics": {
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.90,
    "f1_score": 0.85
  },
  "total_clients": 10,
  "detection_time": 0.123
}
```

#### **Get Status**
```json
// Response
{
  "enabled": true,
  "method": "kmeans",
  "ldp_epsilon": 1.0,
  "total_detections": 5,
  "adapter_initialized": true
}
```

## **🎯 Best Practices**

### **Production Deployment**

1. **Enable Detection Gradually**: Start with detection disabled, then enable with monitoring
2. **Monitor Performance**: Track detection overhead and accuracy
3. **Tune Parameters**: Adjust detection parameters based on your data
4. **Backup Configuration**: Save working configurations
5. **Test Thoroughly**: Test with various attack scenarios

### **Security Considerations**

1. **Privacy Protection**: Use appropriate LDP epsilon values
2. **Access Control**: Restrict detection API access
3. **Audit Logging**: Log all detection decisions
4. **Data Retention**: Set appropriate data retention policies

### **Maintenance**

1. **Regular Updates**: Keep detection algorithms updated
2. **Performance Monitoring**: Monitor detection performance over time
3. **Configuration Review**: Regularly review and update configuration
4. **Log Analysis**: Analyze detection logs for patterns

## **📞 Support**

For issues or questions:

1. **Check Logs**: Review server and detection logs
2. **Verify Configuration**: Ensure proper configuration
3. **Test Components**: Test individual components
4. **Review Documentation**: Check this guide and code comments

## **🔄 Future Enhancements**

### **Planned Features**

1. **Advanced Detection Methods**: More sophisticated algorithms
2. **Real-time Monitoring**: Live detection monitoring
3. **Automated Tuning**: Self-tuning detection parameters
4. **Attack Simulation**: Built-in attack simulation tools
5. **Integration APIs**: More integration options

### **Customization Options**

1. **Custom Detection Methods**: Add your own detection algorithms
2. **Custom Metrics**: Define custom detection metrics
3. **Custom UI**: Customize dashboard display
4. **Custom Logging**: Custom logging and reporting

---

**This integration provides a robust, production-ready data poisoning detection system that seamlessly integrates with your existing Flower-based federated learning infrastructure while maintaining high performance and security standards.**
