# FL Enterprise System

A comprehensive, enterprise-grade Federated Learning system with advanced security, monitoring, and containerization capabilities.

## 🚀 Features

### Core FL System
- **Federated Learning Server**: Robust server implementation using Flower framework
- **Multiple Clients**: Scalable client architecture supporting up to 10 clients
- **Loan Prediction Model**: Machine learning model for loan approval prediction
- **Real-time Training**: Live monitoring of training progress and metrics

### Enterprise Features
- **Central Authority (CA)**: Complete PKI infrastructure for client authentication
- **Certificate Management**: Automated certificate generation, validation, and revocation
- **Security**: TLS/SSL encryption, client authentication, and access control
- **Monitoring**: Comprehensive system metrics and performance monitoring
- **Logging**: Structured logging with centralized log management

### Infrastructure
- **Docker Containerization**: All components run in isolated containers
- **Kubernetes Support**: Full K8s deployment with manifests and configurations
- **Load Balancing**: Nginx reverse proxy with SSL termination
- **Persistent Storage**: Data persistence across container restarts
- **Auto-scaling**: Dynamic client scaling based on demand

### Dashboard
- **Modern UI**: Enterprise-grade dashboard with real-time updates
- **System Controls**: Start/stop system and individual clients
- **Real-time Metrics**: Live system performance monitoring
- **Certificate Management**: CA status and certificate operations
- **Network Topology**: Container network visualization

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   FL Server     │    │   FL Clients    │
│   (Port 5000)   │◄──►│   (Port 8080)   │◄──►│   (Ports N/A)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Central Auth  │    │   Monitoring    │    │   Data Storage  │
│   (Port 9000)   │    │   (Prometheus)  │    │   (Persistent)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Python**: Version 3.9 or higher
- **Kubernetes**: kubectl and cluster access (optional)
- **Memory**: Minimum 8GB RAM
- **Storage**: Minimum 20GB free space

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "FL System"
```

### 2. Deploy the System
```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy everything (Docker + Kubernetes if available)
./deploy.sh

# Or deploy only with Docker
./deploy.sh --docker-only

# Or deploy only to Kubernetes
./deploy.sh --k8s-only
```

### 3. Access the Dashboard
- **Dashboard**: http://localhost:5000
- **Server**: http://localhost:8080
- **CA Service**: http://localhost:9000

## 🔧 Manual Deployment

### Docker Compose
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy storage
kubectl apply -f k8s/persistent-volumes.yaml

# Deploy configuration
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy services
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/services.yaml

# Deploy ingress
kubectl apply -f k8s/ingress.yaml
```

## 🔐 Central Authority

The CA system provides:
- **Certificate Generation**: Automated client certificate creation
- **Validation**: Real-time certificate validation
- **Revocation**: Secure certificate revocation
- **Database**: SQLite-based certificate management
- **API**: RESTful API for certificate operations

### CA Operations
```bash
# Initialize CA
python ca/ca.py --action init

# Generate client certificate
python ca/ca.py --action generate --client-id 1

# Validate certificate
python ca/ca.py --action validate --client-id 1

# List certificates
python ca/ca.py --action list

# Revoke certificate
python ca/ca.py --action revoke --client-id 1 --reason "compromise"
```

## 📊 Monitoring

### Prometheus Metrics
- System performance metrics
- Container resource usage
- Training progress tracking
- Client status monitoring

### Grafana Dashboards
- Pre-configured dashboards
- Real-time visualization
- Custom metric queries
- Alert configuration

## 🔒 Security Features

- **TLS/SSL**: End-to-end encryption
- **Client Authentication**: Certificate-based authentication
- **Access Control**: Role-based permissions
- **Audit Logging**: Comprehensive security logging
- **Secure Storage**: Encrypted sensitive data

## 📁 Project Structure

```
FL System/
├── ca/                     # Central Authority
│   ├── ca.py             # CA core functionality
│   ├── ca_service.py     # CA REST API service
│   ├── Dockerfile        # CA container
│   └── requirements.txt  # CA dependencies
├── dashboard/             # Web Dashboard
│   ├── app.py            # Dashboard backend
│   ├── templates/        # HTML templates
│   ├── Dockerfile        # Dashboard container
│   └── requirements.txt  # Dashboard dependencies
├── server/                # FL Server
│   ├── server.py         # Server implementation
│   ├── Dockerfile        # Server container
│   └── requirements.txt  # Server dependencies
├── clients/               # FL Clients
│   ├── client_template.py # Client template
│   ├── client1/          # Client 1
│   ├── client2/          # Client 2
│   ├── ...               # Additional clients
│   ├── Dockerfile        # Client container
│   └── requirements.txt  # Client dependencies
├── k8s/                   # Kubernetes manifests
│   ├── namespace.yaml    # Namespace definition
│   ├── configmap.yaml    # Configuration
│   ├── secret.yaml       # Secrets
│   ├── deployment.yaml   # Deployments
│   ├── services.yaml     # Services
│   ├── ingress.yaml      # Ingress rules
│   └── persistent-volumes.yaml # Storage
├── monitoring/            # Monitoring setup
├── docker-compose.yml     # Docker Compose
├── deploy.sh             # Deployment script
└── README.md             # This file
```

## 🧪 Testing

### System Health Check
```bash
# Check dashboard health
curl http://localhost:5000/api/system/health

# Check CA status
curl http://localhost:9000/status

# Check server status
curl http://localhost:8080/health
```

### Certificate Operations
```bash
# Generate test certificate
curl -X POST http://localhost:9000/certificates/generate \
  -H "Content-Type: application/json" \
  -d '{"client_id": "test", "permissions": "standard"}'

# Validate certificate
curl http://localhost:9000/certificates/test/validate
```

## 🔧 Configuration

### Environment Variables
- `CA_PORT`: CA service port (default: 9000)
- `SERVER_PORT`: FL server port (default: 8080)
- `DASHBOARD_PORT`: Dashboard port (default: 5000)
- `MIN_CLIENTS`: Minimum clients for training (default: 5)
- `NUM_ROUNDS`: Training rounds (default: 10)

### Configuration Files
- `k8s/configmap.yaml`: Kubernetes configuration
- `docker-compose.yml`: Docker Compose configuration
- `ca/ca.py`: CA configuration
- `server/server.py`: Server configuration

## 🚨 Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs <service-name>

# Restart specific service
docker-compose restart <service-name>
```

#### Kubernetes Issues
```bash
# Check pod status
kubectl get pods -n fl-enterprise

# View pod logs
kubectl logs <pod-name> -n fl-enterprise

# Describe pod
kubectl describe pod <pod-name> -n fl-enterprise
```

#### Certificate Issues
```bash
# Check CA status
curl http://localhost:9000/status

# Reinitialize CA
docker exec fl-ca python ca.py --action init

# Clean up expired certificates
docker exec fl-ca python ca.py --action cleanup
```

## 📈 Scaling

### Horizontal Scaling
- **Clients**: Scale client replicas in Kubernetes
- **Dashboard**: Multiple dashboard instances
- **CA**: High-availability CA setup

### Vertical Scaling
- **Resource Limits**: Adjust CPU/memory limits
- **Storage**: Increase persistent volume sizes
- **Network**: Optimize network policies

## 🔄 Updates and Maintenance

### System Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
./deploy.sh --docker-only

# Update Kubernetes deployment
kubectl rollout restart deployment/fl-dashboard -n fl-enterprise
```

### Backup and Recovery
```bash
# Backup certificates
docker exec fl-ca tar -czf /tmp/ca-backup.tar.gz /app/certs

# Backup data
docker exec fl-server tar -czf /tmp/data-backup.tar.gz /app/data

# Restore from backup
docker cp /tmp/ca-backup.tar.gz fl-ca:/tmp/
docker exec fl-ca tar -xzf /tmp/ca-backup.tar.gz -C /app/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation
- Contact the development team

## 🔮 Roadmap

- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] Automated model deployment
- [ ] Integration with ML platforms
- [ ] Enhanced security features
- [ ] Performance optimization
- [ ] Additional ML algorithms
- [ ] Cloud provider integration

---

**Built with ❤️ for the Federated Learning community**
