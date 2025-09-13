# FL System - Local Execution Guide

This guide explains how to run the entire Federated Learning system locally without Docker containers.

## 🚀 Quick Start

### Windows Users
```bash
# Double-click or run in Command Prompt
start_fl_system.bat
```

### Linux/Mac Users
```bash
# Make executable and run
chmod +x start_fl_system.sh
./start_fl_system.sh
```

### Manual Execution
```bash
# Install dependencies
pip install -r requirements_local.txt

# Run the system
python run_fl_system_local.py
```

## 📋 Prerequisites

- **Python 3.9+** (required)
- **8GB+ RAM** (recommended)
- **20GB+ free disk space**
- **Internet connection** (for initial package installation)

## 🔧 System Components

The local execution script manages these components:

### 1. Central Authority (CA) Service
- **Port**: 9000
- **Purpose**: Certificate management and client authentication
- **URL**: http://localhost:9000

### 2. FL Server
- **Port**: 8080
- **Purpose**: Federated learning coordination
- **URL**: http://localhost:8080

### 3. FL Clients (10 clients)
- **Demo Ports**: 8082-8091
- **Purpose**: Local model training and updates
- **Data**: Client-specific loan prediction data

### 4. Web Dashboard
- **Port**: 5000
- **Purpose**: System monitoring and control
- **URL**: http://localhost:5000

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   FL Server     │    │   FL Clients    │
│   (Port 5000)   │◄──►│   (Port 8080)   │◄──►│   (10 clients)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Central Auth  │    │   Data Storage  │    │   Attack Det.   │
│   (Port 9000)   │    │   (Local Files) │    │   (Integrated)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Features

### Automatic Management
- **Process Management**: Starts/stops all components
- **Port Management**: Checks port availability
- **Health Monitoring**: Monitors system health
- **Error Handling**: Graceful error recovery
- **Logging**: Comprehensive logging to `fl_system.log`

### Security Features
- **Certificate Generation**: Automatic client certificates
- **Attack Detection**: Built-in attack simulation and detection
- **Client Authentication**: Certificate-based authentication
- **Secure Communication**: HTTPS-ready architecture

### Monitoring
- **Real-time Dashboard**: Web-based monitoring interface
- **System Metrics**: CPU, memory, and process monitoring
- **Training Progress**: Live training metrics
- **Client Status**: Individual client monitoring

## 📊 Usage

### Starting the System
```bash
python run_fl_system_local.py
```

The script will:
1. Check dependencies and ports
2. Prepare data (if needed)
3. Start all services in order
4. Generate certificates
5. Open the dashboard in your browser
6. Monitor system health

### Stopping the System
- Press `Ctrl+C` to stop gracefully
- All processes will be terminated cleanly
- Logs are saved to `fl_system.log`

### Dashboard Access
Once started, access the dashboard at:
- **Main Dashboard**: http://localhost:5000
- **System Status**: http://localhost:5000/api/system/status
- **Security Status**: http://localhost:5000/api/security/status

## 🔍 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # Linux/Mac

# Kill the process
taskkill /PID <PID> /F        # Windows
kill -9 <PID>                 # Linux/Mac
```

#### Python Dependencies Missing
```bash
# Install requirements manually
pip install -r requirements_local.txt

# Or install specific packages
pip install flwr numpy pandas scikit-learn flask
```

#### Data Not Found
```bash
# Run data preprocessing manually
cd Datapre
python complete_datapre.py
cd ..
```

#### Permission Errors (Linux/Mac)
```bash
# Make scripts executable
chmod +x start_fl_system.sh
chmod +x run_fl_system_local.py
```

### Log Files
- **Main Log**: `fl_system.log`
- **CA Logs**: Check console output
- **Server Logs**: Check console output
- **Client Logs**: Check console output

### System Requirements
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **Disk Space**: 20GB+
- **CPU**: 4+ cores recommended

## 🧪 Testing

### Health Checks
```bash
# Check CA service
curl http://localhost:9000/health

# Check FL server
curl http://localhost:8080/health

# Check dashboard
curl http://localhost:5000/health
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

### Attack Simulation
```bash
# Simulate attack on client 1
curl -X POST http://localhost:5000/api/security/attack/simulate \
  -H "Content-Type: application/json" \
  -d '{"client_id": "1", "attack_type": "label_flipping"}'

# Remove attack
curl -X POST http://localhost:5000/api/security/attack/remove \
  -H "Content-Type: application/json" \
  -d '{"client_id": "1"}'
```

## 🔧 Configuration

### Environment Variables
```bash
# Set custom ports
export CA_PORT=9000
export SERVER_PORT=8080
export DASHBOARD_PORT=5000

# Set client count
export MIN_CLIENTS=10
export NUM_ROUNDS=10
```

### Configuration Files
- **Server Config**: `server/server.py`
- **CA Config**: `ca/ca.py`
- **Dashboard Config**: `dashboard/app.py`

## 📈 Performance

### Optimization Tips
1. **Close unnecessary applications** to free up RAM
2. **Use SSD storage** for better I/O performance
3. **Increase virtual memory** if needed
4. **Monitor system resources** during training

### Resource Usage
- **CA Service**: ~50MB RAM
- **FL Server**: ~200MB RAM
- **Dashboard**: ~100MB RAM
- **Each Client**: ~50MB RAM
- **Total**: ~800MB RAM (10 clients)

## 🚨 Security Notes

### Local Development Only
This setup is designed for **local development and testing only**. For production use:
- Use Docker containers
- Implement proper network security
- Use production-grade certificates
- Set up proper monitoring

### Data Privacy
- All data stays on your local machine
- No external network communication
- Certificates are self-signed
- Perfect for development and testing

## 📚 Additional Resources

- **Main README**: `README.md` (Docker/K8s setup)
- **Attack Detection**: `ATTACK_DETECTION_README.md`
- **API Documentation**: Available in dashboard
- **Logs**: Check `fl_system.log` for detailed information

## 🤝 Support

If you encounter issues:
1. Check the logs in `fl_system.log`
2. Verify all ports are available
3. Ensure all dependencies are installed
4. Check system resources (RAM, disk space)
5. Review the troubleshooting section above

---

**Happy Federated Learning! 🚀**

