# FL Enterprise Dashboard

A professional, enterprise-grade dashboard for monitoring and controlling the Federated Learning system. Built with modern web technologies and featuring a sleek dark theme.

## 🎯 Features

### **System Management**
- **Real-time System Control**: Start/stop the FL server with one click
- **System Status Monitoring**: Live status indicators and health checks
- **Performance Metrics**: CPU, memory, and disk usage monitoring

### **Client Management**
- **Individual Client Control**: Start/stop specific clients independently
- **Bulk Operations**: Start/stop all clients simultaneously
- **Real-time Status**: Live client status updates and uptime tracking

### **Training Monitoring**
- **Live Training Progress**: Real-time visualization of training rounds
- **Performance Metrics**: Accuracy, precision, recall, and F1-score tracking
- **Historical Data**: Training metrics history and analysis

### **Enterprise Features**
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Updates**: WebSocket-based live data streaming
- **Professional UI**: Dark theme with modern design principles
- **Scalable Architecture**: Built to handle multiple clients efficiently

## 🏗️ Architecture

```
Dashboard (Port 5000)
├── Flask Backend
│   ├── REST API endpoints
│   ├── WebSocket support
│   └── Process management
├── Frontend
│   ├── HTML5 + CSS3
│   ├── JavaScript (ES6+)
│   ├── Chart.js for visualizations
│   └── Socket.IO for real-time updates
└── Integration
    ├── FL Server communication
    ├── Client process management
    └── System metrics collection
```

## 🚀 Quick Start

### Option 1: Direct Python Execution

1. **Navigate to dashboard directory**:
   ```bash
   cd dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the dashboard**:
   ```bash
   python app.py
   ```

4. **Access the dashboard**: Open `http://localhost:5000` in your browser

### Option 2: Using Startup Script

#### Windows:
```bash
start_dashboard.bat
```

#### Linux/Mac:
```bash
python start_dashboard.py
```

### Option 3: Docker Deployment

1. **Build and run with docker-compose**:
   ```bash
   docker-compose up dashboard
   ```

2. **Access the dashboard**: Open `http://localhost:5000` in your browser

## 📊 Dashboard Components

### **Header Section**
- **System Title**: FL Enterprise Dashboard branding
- **Status Indicator**: Real-time system status with color-coded dots
- **Navigation**: Quick access to different sections

### **Sidebar Controls**
- **System Controls**: Start/stop server buttons
- **Client Management**: Individual client control cards
- **Quick Actions**: Bulk operations for all clients

### **Main Content Area**
- **System Metrics**: CPU, memory, disk usage with progress bars
- **Training Chart**: Real-time training progress visualization
- **Metrics Table**: Detailed training metrics in tabular format

### **Client Grid**
- **Status Cards**: Individual client status and controls
- **Real-time Updates**: Live status changes and uptime tracking
- **Individual Controls**: Start/stop buttons for each client

## 🔧 Configuration

### **Environment Variables**
```bash
FLASK_ENV=production          # Flask environment
PYTHONPATH=/app              # Python path
DASHBOARD_PORT=5000          # Dashboard port
DASHBOARD_HOST=0.0.0.0      # Dashboard host
```

### **Port Configuration**
- **Dashboard**: Port 5000 (configurable)
- **FL Server**: Port 8080 (must match server configuration)
- **Clients**: Dynamic ports (managed by system)

### **File Paths**
```bash
dashboard/
├── app.py                   # Main Flask application
├── templates/              # HTML templates
│   └── index.html         # Main dashboard page
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── start_dashboard.py     # Startup script
```

## 📈 API Endpoints

### **System Management**
- `GET /api/system/status` - Get system status
- `POST /api/system/start` - Start FL system
- `POST /api/system/stop` - Stop FL system

### **Client Management**
- `GET /api/clients/status` - Get all client statuses
- `POST /api/clients/start` - Start specific client
- `POST /api/clients/stop` - Stop specific client

### **Metrics**
- `GET /api/metrics/system` - Get system performance metrics
- `GET /api/metrics/training` - Get training metrics
- `GET /api/metrics/clients` - Get client metrics

### **WebSocket Events**
- `connect` - Client connection
- `disconnect` - Client disconnection
- `metrics_update` - Real-time metrics updates

## 🎨 Customization

### **Theme Customization**
The dashboard uses CSS custom properties for easy theming:

```css
:root {
    --primary-bg: #0f1419;        /* Main background */
    --secondary-bg: #1a1f2e;      /* Secondary background */
    --card-bg: #252b3d;           /* Card background */
    --accent-color: #4f46e5;      /* Primary accent */
    --text-primary: #ffffff;      /* Primary text */
    --text-secondary: #a1a1aa;    /* Secondary text */
}
```

### **Adding New Metrics**
1. **Backend**: Add new endpoint in `app.py`
2. **Frontend**: Add new metric card in `index.html`
3. **JavaScript**: Update `updateMetrics()` function

### **Custom Charts**
The dashboard uses Chart.js for visualizations:

```javascript
const chart = new Chart(ctx, {
    type: 'line',
    data: { /* chart data */ },
    options: { /* chart options */ }
});
```

## 🔍 Monitoring & Debugging

### **Log Files**
- **Dashboard logs**: `dashboard.log`
- **System logs**: Check system logs for process information
- **Error tracking**: Real-time error notifications in dashboard

### **Health Checks**
- **System status**: Real-time system health monitoring
- **Client status**: Individual client health tracking
- **Network status**: Connection status indicators

### **Performance Monitoring**
- **Response times**: API endpoint performance tracking
- **Resource usage**: CPU, memory, and disk monitoring
- **Client performance**: Individual client metrics

## 🚨 Troubleshooting

### **Common Issues**

#### **Dashboard Won't Start**
1. Check Python version (3.8+ required)
2. Verify dependencies are installed
3. Check port 5000 is available
4. Verify FL system components exist

#### **No Real-time Updates**
1. Check WebSocket connection status
2. Verify server is running
3. Check browser console for errors
4. Verify network connectivity

#### **Client Control Issues**
1. Check client scripts exist
2. Verify server is running
3. Check process permissions
4. Verify file paths are correct

### **Debug Mode**
Enable debug mode by setting:
```bash
export FLASK_ENV=development
```

### **Logging**
Increase log verbosity:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

### **Production Deployment**
- **HTTPS**: Always use HTTPS in production
- **Authentication**: Implement user authentication
- **Authorization**: Restrict access to authorized users
- **Firewall**: Configure firewall rules appropriately

### **Network Security**
- **Port restrictions**: Only expose necessary ports
- **VPN access**: Use VPN for remote access
- **IP whitelisting**: Restrict access to known IPs

## 📱 Mobile Support

The dashboard is fully responsive and works on:
- **Desktop**: Full feature set with optimal layout
- **Tablet**: Adapted layout for medium screens
- **Mobile**: Mobile-optimized interface

## 🔄 Updates & Maintenance

### **Regular Maintenance**
- **Log rotation**: Implement log file rotation
- **Performance monitoring**: Regular performance checks
- **Security updates**: Keep dependencies updated
- **Backup**: Regular configuration backups

### **Version Updates**
1. **Backup configuration**
2. **Update dependencies**
3. **Test functionality**
4. **Deploy updates**

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**
3. **Make changes**
4. **Test thoroughly**
5. **Submit pull request**

## 📄 License

This dashboard is part of the FL System project and follows the same license terms.

## 📞 Support

For dashboard-specific issues:
1. Check the troubleshooting section
2. Review logs and error messages
3. Check system requirements
4. Contact the development team

---

**Note**: This dashboard is designed for enterprise use and includes advanced monitoring capabilities. For production deployment, ensure proper security measures are in place.
