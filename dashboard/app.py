from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import subprocess
import threading
import time
import json
import os
import psutil
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import requests
import hashlib
from typing import Dict, List, Optional
import docker
from kubernetes import client, config
import yaml
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fl-dashboard-secret-key-2024'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Database setup for users
def init_db():
    """Initialize the database with default admin user"""
    conn = sqlite3.connect('dashboard.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_login BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if admin user exists
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        # Create default admin user (admin/admin)
        admin_hash = hashlib.sha256('admin'.encode()).hexdigest()
        cursor.execute('INSERT INTO users (username, password_hash, first_login) VALUES (?, ?, ?)', 
                      ('admin', admin_hash, True))
        conn.commit()
    
    conn.close()

# Initialize database
init_db()

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    return hash_password(password) == password_hash

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def change_password_required(f):
    """Decorator to require password change after first login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' in session:
            conn = sqlite3.connect('dashboard.db')
            cursor = conn.cursor()
            cursor.execute('SELECT first_login FROM users WHERE id = ?', (session['user_id'],))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return redirect(url_for('change_password'))
        return f(*args, **kwargs)
    return decorated_function

class FLDashboard:
    def __init__(self):
        self.system_status = "running"  # Assume system is running since containers are managed externally
        self.metrics_history = []
        self.docker_client = None
        
        # Initialize Docker client
        self._initialize_docker_client()
        
    def _initialize_docker_client(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Docker client not available: {e}")
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get Docker container metrics if available
            docker_metrics = {}
            if self.docker_client:
                try:
                    containers = self.docker_client.containers.list()
                    docker_metrics = {
                        'total_containers': len(containers),
                        'running_containers': len([c for c in containers if c.status == 'running']),
                        'stopped_containers': len([c for c in containers if c.status == 'exited'])
                    }
                except Exception as e:
                    logger.warning(f"Could not get Docker metrics: {e}")
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'docker': docker_metrics
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
    
    def get_client_metrics(self):
        """Get metrics for all running client containers"""
        client_metrics = {}
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                for container in containers:
                    if container.name.startswith('flsystem-client'):
                        client_id = container.name.split('-')[-1] if '-' in container.name else 'unknown'
                        client_metrics[client_id] = {
                            'status': container.status,
                            'uptime': self._get_container_uptime(container),
                            'ip': self._get_container_ip(container),
                            'ports': container.attrs['NetworkSettings']['Ports']
                        }
            except Exception as e:
                logger.error(f"Error getting client metrics: {e}")
        
        return client_metrics
    
    def _get_container_uptime(self, container):
        """Get container uptime"""
        try:
            stats = container.stats(stream=False)
            if stats:
                # Calculate uptime from container stats
                return "Running"
        except:
            pass
        return "Unknown"
    
    def _get_container_ip(self, container):
        """Get container IP address"""
        try:
            networks = container.attrs['NetworkSettings']['Networks']
            for network_name, network_info in networks.items():
                if network_info.get('IPAddress'):
                    return network_info['IPAddress']
        except:
            pass
        return "N/A"
    
    def get_ca_status(self):
        """Get Central Authority status"""
        try:
            ca_dir = Path("../ca")
            if not ca_dir.exists():
                return {"status": "not_initialized", "message": "CA directory not found"}
            
            # Check if CA files exist
            ca_key = ca_dir / "ca_private_key.pem"
            ca_cert = ca_dir / "ca_certificate.pem"
            ca_db = ca_dir / "certificates.db"
            
            if not all([ca_key.exists(), ca_cert.exists(), ca_db.exists()]):
                return {"status": "incomplete", "message": "CA files missing"}
            
            # Get certificate count
            conn = sqlite3.connect(ca_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM certificates WHERE status = 'active'")
            active_certs = cursor.fetchone()[0]
            conn.close()
            
            return {
                "status": "active",
                "active_certificates": active_certs,
                "ca_certificate": ca_cert.exists(),
                "ca_private_key": ca_key.exists()
            }
        except Exception as e:
            logger.error(f"Error getting CA status: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_network_topology(self):
        """Get network topology information"""
        try:
            if self.docker_client:
                containers = self.docker_client.containers.list()
                network_info = {}
                
                for container in containers:
                    if container.name.startswith('flsystem-'):
                        network_info[container.name] = {
                            'status': container.status,
                            'ip': self._get_container_ip(container),
                            'ports': container.attrs['NetworkSettings']['Ports']
                        }
                
                return network_info
            else:
                return {"message": "Docker not available"}
        except Exception as e:
            logger.error(f"Error getting network topology: {e}")
            return {"error": str(e)}

# Initialize dashboard
dashboard = FLDashboard()

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username and password:
            conn = sqlite3.connect('dashboard.db')
            cursor = conn.cursor()
            cursor.execute('SELECT id, password_hash, first_login FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user and verify_password(password, user[1]):
                session['user_id'] = user[0]
                session['username'] = username
                session.permanent = True
                
                if user[2]:  # first_login
                    return redirect(url_for('change_password'))
                else:
                    return redirect(url_for('index'))
            else:
                flash('Invalid username or password', 'error')
        
        flash('Please provide username and password', 'error')
    
    return render_template('login.html')

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change password page for first login"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([current_password, new_password, confirm_password]):
            flash('All fields are required', 'error')
            return render_template('change_password.html')
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
            return render_template('change_password.html')
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('change_password.html')
        
        # Verify current password
        conn = sqlite3.connect('dashboard.db')
        cursor = conn.cursor()
        cursor.execute('SELECT password_hash FROM users WHERE id = ?', (session['user_id'],))
        user = cursor.fetchone()
        
        if user and verify_password(current_password, user[0]):
            # Update password and set first_login to False
            new_hash = hash_password(new_password)
            cursor.execute('UPDATE users SET password_hash = ?, first_login = ? WHERE id = ?', 
                         (new_hash, False, session['user_id']))
            conn.commit()
            conn.close()
            
            flash('Password changed successfully!', 'success')
            return redirect(url_for('index'))
        else:
            conn.close()
            flash('Current password is incorrect', 'error')
    
    return render_template('change_password.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
@change_password_required
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/system/status')
@login_required
def get_system_status():
    """Get current system status"""
    return jsonify({
        'status': dashboard.system_status,
        'clients_count': len(dashboard.get_client_metrics()),
        'active_clients': len([c for c in dashboard.get_client_metrics().values() if c['status'] == 'running']),
        'ca_status': dashboard.get_ca_status()
    })

@app.route('/api/metrics/system')
@login_required
def get_system_metrics():
    """Get system performance metrics"""
    return jsonify(dashboard.get_system_metrics())

@app.route('/api/metrics/training')
@login_required
def get_training_metrics():
    """Get training metrics"""
    return jsonify(dashboard.get_training_metrics())

@app.route('/api/metrics/clients')
@login_required
def get_client_metrics():
    """Get client metrics"""
    return jsonify(dashboard.get_client_metrics())

@app.route('/api/clients/status')
@login_required
def get_clients_status():
    """Get status of all clients"""
    return jsonify(dashboard.get_client_metrics())

@app.route('/api/ca/status')
@login_required
def get_ca_status():
    """Get Central Authority status"""
    return jsonify(dashboard.get_ca_status())

@app.route('/api/network/topology')
@login_required
def get_network_topology():
    """Get network topology"""
    return jsonify(dashboard.get_network_topology())

@app.route('/api/system/health')
@login_required
def system_health():
    """Get comprehensive system health"""
    try:
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': dashboard.system_status,
            'ca_status': dashboard.get_ca_status(),
            'system_metrics': dashboard.get_system_metrics(),
            'client_metrics': dashboard.get_client_metrics(),
            'network_topology': dashboard.get_network_topology()
        }
        return jsonify(health_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            
            # Get CA status
            ca_status = dashboard.get_ca_status()
            
            # Emit metrics via WebSocket
            socketio.emit('metrics_update', {
                'system': system_metrics,
                'clients': client_metrics,
                'training': training_metrics,
                'ca_status': ca_status,
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
    emit('connected', {'message': 'Connected to FL Enterprise Dashboard'})

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
