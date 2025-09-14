# 🐳 Docker Integration Guide for Enhanced FL System with Detection

## **Overview**

This guide explains how to run the enhanced federated learning system with data poisoning detection using Docker. The system is fully containerized and includes all necessary components for production deployment.

## **🏗️ Docker Architecture**

### **Service Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   FL Server     │    │   FL Clients    │
│   (Port 5000)   │◄──►│   (Port 8080)   │◄──►│   (Ports 8082+) │
│                 │    │                 │    │                 │
│ • Detection UI  │    │ • Enhanced      │    │ • Scikit-learn  │
│ • Real Results  │    │   Strategy      │    │ • LogisticReg   │
│ • API Endpoints │    │ • Detection     │    │ • Real Data     │
│                 │    │   Adapter       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   CA Service    │
                    │   (Port 9000)   │
                    │                 │
                    │ • Certificates  │
                    │ • Validation    │
                    │ • Security      │
                    └─────────────────┘
```

### **Docker Services**

| Service | Port | Description |
|---------|------|-------------|
| `dashboard` | 5000 | Enhanced dashboard with detection UI |
| `server` | 8080 | FL server with detection integration |
| `ca` | 9000 | Certificate authority service |
| `client1-10` | 8082-8091 | FL clients (10 clients) |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3002 | Monitoring dashboard |
| `nginx` | 80/443 | Reverse proxy |

## **🚀 Quick Start**

### **1. Automated Setup**

```bash
# Run the automated setup script
python docker_integration_setup.py
```

This script will:
- Check Docker availability
- Build all Docker images
- Start all services
- Test detection integration
- Show access information

### **2. Manual Setup**

```bash
# Build all images
docker-compose build

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### **3. Access the System**

- **Dashboard**: http://localhost:5000
- **Server**: http://localhost:8080
- **CA Service**: http://localhost:9000
- **Prometheus**: http://localhost:9090

## **🔧 Configuration**

### **Environment Variables**

The system can be configured using environment variables:

```bash
# Detection Configuration
DETECTION_ENABLED=true
DETECTION_METHOD=kmeans
LDP_EPSILON=1.0

# Server Configuration
SERVER_PORT=8080
MIN_CLIENTS=10
NUM_ROUNDS=10
CA_URL=http://ca:9000
ENABLE_CERTIFICATES=true

# Dashboard Configuration
DASHBOARD_PORT=5000
FLASK_ENV=production
```

### **Docker Compose Configuration**

Key configuration in `docker-compose.yml`:

```yaml
services:
  dashboard:
    build: ./dashboard
    ports:
      - "5000:5000"
    volumes:
      - ./data_poisoning_detection:/app/data_poisoning_detection:ro
      - detection-results:/app/detection_results
    environment:
      - DETECTION_ENABLED=true
      - DETECTION_METHOD=kmeans
      - LDP_EPSILON=1.0

  server:
    build: ./server
    ports:
      - "8080:8080"
    volumes:
      - ./data_poisoning_detection:/app/data_poisoning_detection:ro
      - detection-results:/app/detection_results
    environment:
      - DETECTION_ENABLED=true
      - DETECTION_METHOD=kmeans
      - LDP_EPSILON=1.0
    command: python enhanced_server_strategy.py
```

## **📊 Using the Detection System**

### **1. Access the Dashboard**

1. Open http://localhost:5000 in your browser
2. Navigate to the "Demo" tab
3. Click the "Run Detection" button

### **2. Detection Process**

The detection system will:
1. **Collect FL Results**: Retrieve latest client training results
2. **Convert Data**: Transform scikit-learn parameters to PyTorch format
3. **Apply LDP**: Add privacy-preserving noise (if enabled)
4. **Run Detection**: Execute K-means/z-score detection algorithm
5. **Filter Malicious**: Remove detected malicious clients
6. **Display Results**: Show detection metrics and flagged clients

### **3. Detection Results**

The dashboard displays:
- **Detected Malicious Clients**: List of flagged client IDs
- **Detection Metrics**: Accuracy, precision, recall, F1-score
- **Detection Method**: Algorithm used (K-means, z-score, etc.)
- **Execution Time**: Time taken for detection
- **Privacy Budget**: LDP epsilon value used

## **🔍 Monitoring and Logs**

### **View Logs**

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f dashboard
docker-compose logs -f server

# Last 100 lines
docker-compose logs --tail=100 dashboard
```

### **Service Health Checks**

```bash
# Check service status
docker-compose ps

# Check service health
curl http://localhost:5000/health
curl http://localhost:8080/health
curl http://localhost:9000/health
```

### **Detection Metrics**

Detection results are saved to:
- **Dashboard Volume**: `/app/detection_results/latest_detection.json`
- **Server Volume**: `/app/detection_results/enhanced_training_metrics.json`

## **🛠️ Development and Debugging**

### **Development Mode**

```bash
# Run in development mode with live reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### **Debugging**

```bash
# Access container shell
docker-compose exec dashboard bash
docker-compose exec server bash

# Check container logs
docker logs <container_name>

# Inspect container
docker inspect <container_name>
```

### **Rebuilding Images**

```bash
# Rebuild specific service
docker-compose build dashboard

# Rebuild all services
docker-compose build --no-cache

# Force rebuild and restart
docker-compose up --build --force-recreate
```

## **📈 Performance Optimization**

### **Resource Limits**

Add resource limits to `docker-compose.yml`:

```yaml
services:
  server:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### **Scaling**

```bash
# Scale clients
docker-compose up --scale client1=3 --scale client2=3

# Scale with specific configuration
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up
```

## **🔒 Security Considerations**

### **Network Security**

- All services run in isolated Docker network
- Only necessary ports are exposed
- Internal communication uses service names

### **Data Security**

- Sensitive data mounted as read-only volumes
- Detection results stored in Docker volumes
- Certificate validation enabled by default

### **Access Control**

- Dashboard requires authentication (if enabled)
- API endpoints protected by CORS
- Certificate-based client authentication

## **🚨 Troubleshooting**

### **Common Issues**

1. **Services Not Starting**
   ```bash
   # Check Docker daemon
   docker info
   
   # Check available resources
   docker system df
   
   # Clean up unused resources
   docker system prune
   ```

2. **Detection Not Working**
   ```bash
   # Check detection logs
   docker-compose logs dashboard | grep detection
   
   # Check server logs
   docker-compose logs server | grep detection
   
   # Verify detection files
   docker-compose exec dashboard ls -la /app/data_poisoning_detection/
   ```

3. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :5000
   
   # Change ports in docker-compose.yml
   ports:
     - "5001:5000"  # Use port 5001 instead
   ```

4. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Memory
   ```

### **Debug Commands**

```bash
# Check container health
docker-compose ps

# View detailed logs
docker-compose logs --tail=50 -f

# Check container resources
docker stats

# Inspect container configuration
docker inspect <container_name>

# Access container shell
docker-compose exec <service_name> bash
```

## **📋 Maintenance**

### **Regular Maintenance**

```bash
# Update images
docker-compose pull

# Clean up old images
docker image prune

# Clean up volumes
docker volume prune

# Full cleanup
docker system prune -a
```

### **Backup and Restore**

```bash
# Backup volumes
docker run --rm -v detection-results:/data -v $(pwd):/backup alpine tar czf /backup/detection-results.tar.gz -C /data .

# Restore volumes
docker run --rm -v detection-results:/data -v $(pwd):/backup alpine tar xzf /backup/detection-results.tar.gz -C /data
```

## **🔄 Updates and Upgrades**

### **Updating the System**

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up --build -d

# Verify update
docker-compose ps
curl http://localhost:5000/health
```

### **Version Management**

```bash
# Tag current version
docker tag fl-system:latest fl-system:v1.0.0

# Rollback to previous version
docker-compose down
docker-compose -f docker-compose.yml -f docker-compose.v1.0.0.yml up -d
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

### **Example API Usage**

```bash
# Run detection
curl -X POST http://localhost:5000/api/detection/run_enhanced \
  -H "Content-Type: application/json" \
  -d '{"use_cached": false}'

# Get detection status
curl http://localhost:5000/api/detection/status

# Update configuration
curl -X POST http://localhost:5000/api/detection/config \
  -H "Content-Type: application/json" \
  -d '{"method": "z_score", "ldp_epsilon": 2.0}'
```

## **🎯 Best Practices**

### **Production Deployment**

1. **Use Environment Files**: Store sensitive configuration in `.env` files
2. **Enable Logging**: Configure proper log levels and rotation
3. **Monitor Resources**: Set up monitoring and alerting
4. **Backup Data**: Regular backup of detection results and models
5. **Security Updates**: Keep Docker images and dependencies updated

### **Development Workflow**

1. **Use Development Compose**: Separate compose file for development
2. **Volume Mounts**: Mount source code for live development
3. **Debug Mode**: Enable debug logging and verbose output
4. **Test Integration**: Regular testing of detection functionality

## **📞 Support**

For issues or questions:

1. **Check Logs**: Review container logs for errors
2. **Verify Configuration**: Ensure proper environment variables
3. **Test Components**: Test individual services
4. **Review Documentation**: Check this guide and code comments

---

**This Docker integration provides a robust, production-ready deployment of the enhanced federated learning system with comprehensive data poisoning detection capabilities.**
