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
import requests

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
            # Try to read from server container
            metrics_file = Path("../server/training_metrics.json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                # Ensure metrics is a list format
                if isinstance(metrics, list):
                    return metrics
                elif isinstance(metrics, dict) and 'rounds' in metrics:
                    return metrics['rounds']
                else:
                    # Convert single dict to list format
                    return [metrics] if metrics else []
            
            # If no real data, return sample data for demonstration
            return self.get_sample_training_data()
        except Exception as e:
            logger.error(f"Error reading training metrics: {e}")
            return self.get_sample_training_data()
    
    def get_sample_training_data(self):
        """Generate sample training data for demonstration"""
        import random
        import time
        
        # Generate sample data for 5 rounds with 10 clients
        sample_data = []
        for round_num in range(1, 6):
            client_metrics = {}
            accuracies = []
            
            for client_id in range(1, 11):  # 10 clients
                # Generate realistic accuracy values (improving over rounds)
                base_accuracy = 0.6 + (round_num - 1) * 0.08 + random.uniform(-0.05, 0.05)
                accuracy = max(0.5, min(0.95, base_accuracy))
                
                # Generate loss (decreasing over rounds)
                loss = 0.8 - (round_num - 1) * 0.12 + random.uniform(-0.1, 0.1)
                loss = max(0.1, min(1.0, loss))
                
                # Ensure precision, recall, f1 are close to accuracy
                precision = max(0.0, min(1.0, accuracy + random.uniform(-0.03, 0.03)))
                recall = max(0.0, min(1.0, accuracy + random.uniform(-0.03, 0.03)))
                f1_score = max(0.0, min(1.0, accuracy + random.uniform(-0.03, 0.03)))
                
                client_metrics[str(client_id)] = {
                    "accuracy": round(accuracy, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1_score": round(f1_score, 4),
                    "loss": round(loss, 4),
                    "num_examples": random.randint(20000, 25000)
                }
                accuracies.append(accuracy)
            
            avg_accuracy = sum(accuracies) / len(accuracies)
            
            sample_data.append({
                "round": round_num,
                "avg_accuracy": round(avg_accuracy, 4),
                "client_metrics": client_metrics
            })
        
        return sample_data
    
    def get_docker_containers(self):
        """Get Docker containers related to FL system"""
        if not self.docker_client:
            return {}
        
        try:
            containers = self.docker_client.containers.list(all=True)
            fl_containers = {}
            
            for container in containers:
                # Check if container is part of FL system (more flexible matching)
                container_name = container.name.lower()
                if any(name in container_name for name in ['flsystem', 'fl-enterprise', 'client', 'server', 'ca', 'dashboard']):
                    container_type = 'unknown'
                    client_id = None
                    
                    if 'client' in container_name:
                        container_type = 'client'
                        # Extract client ID from container name (handle multiple formats)
                        try:
                            # Try different patterns: client1, client-1, fl-system-client1, etc.
                            import re
                            match = re.search(r'client(\d+)', container_name)
                            if match:
                                client_id = int(match.group(1))
                        except:
                            pass
                    elif 'server' in container_name:
                        container_type = 'server'
                    elif 'ca' in container_name:
                        container_type = 'ca'
                    elif 'dashboard' in container_name:
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
        client_metrics = {}
        
        if self.docker_client:
            # Use Docker container detection
            containers = self.get_docker_containers()
            
            for container_name, container_info in containers.items():
                if container_info['type'] == 'client' and container_info['client_id']:
                    client_id = str(container_info['client_id'])
                    client_metrics[client_id] = {
                        'status': 'running' if container_info['status'] == 'running' else 'stopped',
                        'container_name': container_name,
                        'container_id': container_info['id'],
                        'uptime': 'N/A',  # Could calculate from created time
                        'metrics': {}
                    }
        else:
            # Fallback to subprocess detection
            for client_id, client_info in self.clients.items():
                if client_info['status'] == 'running':
                    # Check if client process is still running
                    if client_info['process'].poll() is not None:
                        client_info['status'] = 'stopped'
                    
                    # Get client-specific metrics
                    client_metrics[str(client_id)] = {
                        'status': client_info['status'],
                        'uptime': str(datetime.now() - client_info['start_time']).split('.')[0],
                        'metrics': client_info.get('metrics', {})
                    }
        
        # If no clients found, create sample data for demonstration
        if not client_metrics:
            for client_id in range(1, 11):
                client_metrics[str(client_id)] = {
                    'status': 'stopped',
                    'uptime': 'N/A',
                    'metrics': {}
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

@app.route('/api/debug/containers')
def debug_containers():
    """Debug endpoint to show all detected containers"""
    containers = dashboard.get_docker_containers()
    return jsonify({
        'containers': containers,
        'client_count': len([c for c in containers.values() if c['type'] == 'client']),
        'active_clients': len([c for c in containers.values() if c['type'] == 'client' and c['status'] == 'running']),
        'docker_available': dashboard.docker_client is not None
    })

@app.route('/api/ca/status')
def get_ca_status():
    """Get CA service status"""
    try:
        ca_url = os.environ.get('CA_URL', 'http://ca:9000')
        response = requests.get(f"{ca_url}/status", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'status': 'error', 'message': 'CA service unavailable'}), response.status_code
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ca/certificates')
def list_ca_certificates():
    """List all certificates from CA"""
    try:
        ca_url = os.environ.get('CA_URL', 'http://ca:9000')
        response = requests.get(f"{ca_url}/certificates", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'CA service unavailable'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ca/certificates/generate', methods=['POST'])
def generate_certificate():
    """Generate a new client certificate"""
    try:
        ca_url = os.environ.get('CA_URL', 'http://ca:9000')
        data = request.get_json()
        response = requests.post(f"{ca_url}/certificates/generate", json=data, timeout=10)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ca/certificates/<client_id>/validate')
def validate_certificate(client_id):
    """Validate a client certificate"""
    try:
        ca_url = os.environ.get('CA_URL', 'http://ca:9000')
        response = requests.get(f"{ca_url}/certificates/{client_id}/validate", timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# Demo API endpoints for malicious client simulation
@app.route('/api/demo/clients')
def get_demo_clients():
    """Get all clients with their demo status"""
    try:
        clients = []
        client_metrics = dashboard.get_client_metrics()
        
        for client_id in range(1, 11):  # Clients 1-10
            try:
                # Try to get status from client's demo server
                response = requests.get(f"http://client{client_id}:808{client_id + 1}/status", timeout=2)
                if response.status_code == 200:
                    client_data = response.json()
                    is_malicious = client_data.get('is_malicious', False)
                else:
                    is_malicious = False
            except:
                is_malicious = False
            
            # Get latest metrics for this client
            latest_accuracy = 0.0
            latest_loss = 1.0
            if isinstance(client_metrics, dict) and str(client_id) in client_metrics:
                client_info = client_metrics[str(client_id)]
                latest_accuracy = client_info.get('accuracy', 0.0)
                latest_loss = client_info.get('loss', 1.0)
            
            clients.append({
                'client_id': client_id,
                'is_malicious': is_malicious,
                'status': 'Malicious' if is_malicious else 'Normal',
                'accuracy': round(latest_accuracy, 4),
                'loss': round(latest_loss, 4),
                'demo_port': 8081 + client_id
            })
        
        return jsonify({'clients': clients})
    except Exception as e:
        logger.error(f"Error getting demo clients: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo/toggle/<int:client_id>', methods=['POST'])
def toggle_client_malicious(client_id):
    """Toggle malicious status for a specific client"""
    try:
        demo_port = 8081 + client_id
        response = requests.post(f"http://client{client_id}:{demo_port}/toggle_malicious", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'client_id': client_id,
                'is_malicious': result.get('is_malicious', False)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to toggle client status'}), 500
    except Exception as e:
        logger.error(f"Error toggling client {client_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/demo/reset_all', methods=['POST'])
def reset_all_clients():
    """Reset all clients to normal (non-malicious) status"""
    try:
        results = []
        for client_id in range(1, 11):
            try:
                demo_port = 8081 + client_id
                response = requests.post(f"http://client{client_id}:{demo_port}/reset_malicious", timeout=3)
                
                if response.status_code == 200:
                    results.append({'client_id': client_id, 'success': True})
                else:
                    results.append({'client_id': client_id, 'success': False, 'error': 'Request failed'})
            except Exception as e:
                results.append({'client_id': client_id, 'success': False, 'error': str(e)})
        
        success_count = sum(1 for r in results if r['success'])
        return jsonify({
            'success': True,
            'message': f'Reset {success_count}/10 clients successfully',
            'results': results
        })
    except Exception as e:
        logger.error(f"Error resetting all clients: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint for dashboard"""
    try:
        # Get system metrics
        try:
            system_metrics = dashboard.get_system_metrics()
            if not isinstance(system_metrics, dict):
                system_metrics = {}
        except:
            system_metrics = {}
            
        try:
            client_metrics = dashboard.get_client_metrics()
            if not isinstance(client_metrics, dict):
                client_metrics = {}
        except:
            client_metrics = {}
            
        try:
            training_metrics = dashboard.get_training_metrics()
            # Handle both dict and list formats
            if isinstance(training_metrics, list):
                rounds_completed = len(training_metrics)
            elif isinstance(training_metrics, dict):
                rounds_completed = training_metrics.get('rounds_completed', 0)
            else:
                rounds_completed = 0
        except:
            rounds_completed = 0
        
        # Generate Prometheus format metrics
        metrics_text = f"""# HELP dashboard_clients_total Total number of clients
# TYPE dashboard_clients_total gauge
dashboard_clients_total {client_metrics.get('total_clients', 0)}

# HELP dashboard_clients_active Number of active clients
# TYPE dashboard_clients_active gauge
dashboard_clients_active {client_metrics.get('active_clients', 0)}

# HELP dashboard_training_rounds_total Total training rounds completed
# TYPE dashboard_training_rounds_total counter
dashboard_training_rounds_total {rounds_completed}

# HELP dashboard_system_cpu_percent CPU usage percentage
# TYPE dashboard_system_cpu_percent gauge
dashboard_system_cpu_percent {system_metrics.get('cpu_percent', 0)}

# HELP dashboard_system_memory_percent Memory usage percentage
# TYPE dashboard_system_memory_percent gauge
dashboard_system_memory_percent {system_metrics.get('memory_percent', 0)}

# HELP dashboard_uptime_seconds Dashboard uptime in seconds
# TYPE dashboard_uptime_seconds counter
dashboard_uptime_seconds {time.time() - getattr(dashboard, 'start_time', time.time())}
"""
        
        from flask import Response
        return Response(metrics_text, mimetype='text/plain')
    except Exception as e:
        logger.error(f"Error generating dashboard metrics: {e}")
        from flask import Response
        return Response("# Error generating metrics\n", mimetype='text/plain'), 500

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
