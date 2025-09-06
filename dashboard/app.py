from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import subprocess
import threading
import time
import json
import os
import psutil
from datetime import datetime
import logging
from pathlib import Path
import docker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fl-dashboard-secret-key-2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class FLDashboard:
    def __init__(self):
        self.clients = {}
        self.server_process = None
        self.training_rounds = 0
        self.system_status = "stopped"
        self.metrics_history = []
        self.client_statuses = {}
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Could not connect to Docker: {e}")
            self.docker_client = None
        
    def start_server(self):
        """Start the federated learning server"""
        try:
            server_script = Path("../server/server.py")
            if server_script.exists():
                self.server_process = subprocess.Popen([
                    "python", str(server_script)
                ], cwd="../server", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.system_status = "running"
                logger.info("Server started successfully")
                return True
            else:
                logger.error("Server script not found")
                return False
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the federated learning server"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.system_status = "stopped"
                logger.info("Server stopped successfully")
                return True
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
                return False
        return True
    
    def start_client(self, client_id):
        """Start a specific client"""
        try:
            client_dir = Path(f"../clients/client{client_id}")
            client_script = client_dir / "client.py"
            
            if client_script.exists():
                process = subprocess.Popen([
                    "python", str(client_script)
                ], cwd=client_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.clients[client_id] = {
                    'process': process,
                    'status': 'running',
                    'start_time': datetime.now(),
                    'metrics': {}
                }
                
                logger.info(f"Client {client_id} started successfully")
                return True
            else:
                logger.error(f"Client script not found: {client_script}")
                return False
        except Exception as e:
            logger.error(f"Error starting client {client_id}: {e}")
            return False
    
    def stop_client(self, client_id):
        """Stop a specific client"""
        if client_id in self.clients:
            try:
                process = self.clients[client_id]['process']
                process.terminate()
                process.wait(timeout=5)
                self.clients[client_id]['status'] = 'stopped'
                logger.info(f"Client {client_id} stopped successfully")
                return True
            except Exception as e:
                logger.error(f"Error stopping client {client_id}: {e}")
                return False
        return True
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_training_metrics(self):
        """Get training metrics from the server"""
        try:
            metrics_file = Path("../server/training_metrics.json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return metrics
            return []
        except Exception as e:
            logger.error(f"Error reading training metrics: {e}")
            return []
    
    def get_docker_containers(self):
        """Get Docker containers related to FL system"""
        if not self.docker_client:
            return {}
        
        try:
            containers = self.docker_client.containers.list(all=True)
            fl_containers = {}
            
            for container in containers:
                # Check if container is part of FL system
                if any(name in container.name for name in ['flsystem', 'fl-enterprise']):
                    container_type = 'unknown'
                    client_id = None
                    
                    if 'client' in container.name:
                        container_type = 'client'
                        # Extract client ID from container name
                        try:
                            client_id = int(container.name.split('client')[1].split('-')[0])
                        except:
                            pass
                    elif 'server' in container.name:
                        container_type = 'server'
                    elif 'ca' in container.name:
                        container_type = 'ca'
                    elif 'dashboard' in container.name:
                        container_type = 'dashboard'
                    
                    fl_containers[container.name] = {
                        'id': container.short_id,
                        'name': container.name,
                        'status': container.status,
                        'type': container_type,
                        'client_id': client_id,
                        'created': container.attrs['Created'],
                        'image': container.image.tags[0] if container.image.tags else 'unknown'
                    }
            
            return fl_containers
        except Exception as e:
            logger.error(f"Error getting Docker containers: {e}")
            return {}

    def get_client_metrics(self):
        """Get metrics for all clients"""
        if self.docker_client:
            # Use Docker container detection
            containers = self.get_docker_containers()
            client_metrics = {}
            
            for container_name, container_info in containers.items():
                if container_info['type'] == 'client' and container_info['client_id']:
                    client_id = container_info['client_id']
                    client_metrics[client_id] = {
                        'status': 'running' if container_info['status'] == 'running' else 'stopped',
                        'container_name': container_name,
                        'container_id': container_info['id'],
                        'uptime': 'N/A',  # Could calculate from created time
                        'metrics': {}
                    }
            
            return client_metrics
        else:
            # Fallback to subprocess detection
            client_metrics = {}
            for client_id, client_info in self.clients.items():
                if client_info['status'] == 'running':
                    # Check if client process is still running
                    if client_info['process'].poll() is not None:
                        client_info['status'] = 'stopped'
                    
                    # Get client-specific metrics
                    client_metrics[client_id] = {
                        'status': client_info['status'],
                        'uptime': str(datetime.now() - client_info['start_time']).split('.')[0],
                        'metrics': client_info.get('metrics', {})
                    }
            
            return client_metrics

# Initialize dashboard
dashboard = FLDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    # Set default username if not in session
    if 'username' not in session:
        session['username'] = 'admin'
    return render_template('index.html')

@app.route('/api/system/status')
def get_system_status():
    """Get current system status"""
    if dashboard.docker_client:
        # Use Docker container detection
        containers = dashboard.get_docker_containers()
        client_containers = [c for c in containers.values() if c['type'] == 'client']
        active_clients = len([c for c in client_containers if c['status'] == 'running'])
        
        # Check server status
        server_containers = [c for c in containers.values() if c['type'] == 'server']
        server_status = 'running' if any(c['status'] == 'running' for c in server_containers) else 'stopped'
        
        # Check CA status
        ca_containers = [c for c in containers.values() if c['type'] == 'ca']
        ca_status = 'running' if any(c['status'] == 'running' for c in ca_containers) else 'stopped'
        
        return jsonify({
            'status': server_status,
            'clients_count': len(client_containers),
            'active_clients': active_clients,
            'ca_status': ca_status,
            'containers': containers
        })
    else:
        # Fallback to subprocess detection
        return jsonify({
            'status': dashboard.system_status,
            'clients_count': len(dashboard.clients),
            'active_clients': len([c for c in dashboard.clients.values() if c['status'] == 'running'])
        })

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the federated learning system"""
    try:
        if dashboard.start_server():
            return jsonify({'success': True, 'message': 'System started successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to start system'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Stop the federated learning system"""
    try:
        if dashboard.stop_server():
            return jsonify({'success': True, 'message': 'System stopped successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to stop system'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/clients/start', methods=['POST'])
def start_client():
    """Start a specific client"""
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        
        if dashboard.start_client(client_id):
            return jsonify({'success': True, 'message': f'Client {client_id} started successfully'})
        else:
            return jsonify({'success': False, 'message': f'Failed to start client {client_id}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/clients/stop', methods=['POST'])
def stop_client():
    """Stop a specific client"""
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        
        if dashboard.stop_client(client_id):
            return jsonify({'success': True, 'message': f'Client {client_id} stopped successfully'})
        else:
            return jsonify({'success': False, 'message': f'Failed to stop client {client_id}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/metrics/system')
def get_system_metrics():
    """Get system performance metrics"""
    return jsonify(dashboard.get_system_metrics())

@app.route('/api/metrics/training')
def get_training_metrics():
    """Get training metrics"""
    return jsonify(dashboard.get_training_metrics())

@app.route('/api/metrics/clients')
def get_client_metrics():
    """Get client metrics"""
    return jsonify(dashboard.get_client_metrics())

@app.route('/api/clients/status')
def get_clients_status():
    """Get status of all clients"""
    return jsonify(dashboard.get_client_metrics())

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/test')
def test_page():
    """Simple test page to verify Flask is working"""
    return """
    <html>
    <head><title>FL Dashboard Test</title></head>
    <body>
        <h1>FL Dashboard is Working! 🎉</h1>
        <p>If you can see this page, Flask is running correctly.</p>
        <p>Dashboard should be available at: <a href="/">Main Dashboard</a></p>
    </body>
    </html>
    """

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', 'admin')
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('login'))

def background_metrics_collector():
    """Background task to collect metrics"""
    while True:
        try:
            # Get system metrics
            system_metrics = dashboard.get_system_metrics()
            
            # Get client metrics
            client_metrics = dashboard.get_client_metrics()
            
            # Get training metrics
            training_metrics = dashboard.get_training_metrics()
            
            # Emit metrics via WebSocket
            socketio.emit('metrics_update', {
                'system': system_metrics,
                'clients': client_metrics,
                'training': training_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in background metrics collector: {e}")
            time.sleep(10)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to dashboard")
    emit('connected', {'message': 'Connected to FL Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from dashboard")

if __name__ == '__main__':
    # Start background metrics collector
    metrics_thread = threading.Thread(target=background_metrics_collector, daemon=True)
    metrics_thread.start()
    
    # Run the dashboard
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
