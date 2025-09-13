#!/usr/bin/env python3
"""
Federated Learning System - Local Execution Script
Runs the entire FL system without Docker containers
Manages all components: CA, Server, Clients, Dashboard
"""

import os
import sys
import time
import json
import signal
import threading
import subprocess
import requests
from pathlib import Path
from datetime import datetime
import logging
import psutil
import webbrowser
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fl_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FLSystemManager:
    """Manages the entire FL system locally without Docker"""
    
    def __init__(self):
        self.processes = {}
        self.threads = {}
        self.running = False
        self.ports = {
            'ca': 9000,
            'server': 8080,
            'dashboard': 5000,
            'clients': list(range(8092, 8102))  # 10 client demo ports (changed to avoid conflicts)
        }
        self.base_dir = Path(__file__).parent
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['logs', 'certs', 'data', 'models']
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'flwr', 'numpy', 'pandas', 'sklearn', 'joblib',
            'flask', 'flask_socketio', 'flask_cors', 'requests',
            'cryptography', 'psutil'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.error("Please install them using: pip install -r requirements.txt")
            return False
        
        logger.info("All dependencies are available!")
        return True
    
    def check_ports(self) -> bool:
        """Check if required ports are available"""
        logger.info("Checking port availability...")
        
        for service, port in self.ports.items():
            if service == 'clients':
                for client_port in port:
                    if not self.is_port_available(client_port):
                        logger.error(f"Port {client_port} is not available")
                        return False
            else:
                if not self.is_port_available(port):
                    logger.error(f"Port {port} is not available for {service}")
                    return False
        
        logger.info("All ports are available!")
        return True
    
    def is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    def prepare_data(self) -> bool:
        """Prepare data for FL training"""
        logger.info("Preparing data...")
        
        data_dir = self.base_dir / "Datapre" / "FL_clients"
        if not data_dir.exists() or len(list(data_dir.glob("client_*.csv"))) < 10:
            logger.info("Running data preprocessing...")
            preprocess_script = self.base_dir / "Datapre" / "complete_datapre.py"
            
            if not preprocess_script.exists():
                logger.error("Data preprocessing script not found!")
                return False
            
            try:
                result = subprocess.run([
                    sys.executable, str(preprocess_script)
                ], cwd=self.base_dir / "Datapre", 
                capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info("Data preprocessing completed!")
                    return True
                else:
                    logger.error(f"Data preprocessing failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                logger.error("Data preprocessing timed out")
                return False
            except Exception as e:
                logger.error(f"Error running data preprocessing: {e}")
                return False
        
        logger.info("Data is ready!")
        return True
    
    def start_ca_service(self) -> bool:
        """Start the Central Authority service"""
        logger.info("Starting CA service...")
        
        ca_script = self.base_dir / "ca" / "ca_service.py"
        if not ca_script.exists():
            logger.error("CA service script not found!")
            return False
        
        try:
            env = os.environ.copy()
            env['CA_PORT'] = str(self.ports['ca'])
            env['PYTHONPATH'] = str(self.base_dir)
            
            process = subprocess.Popen([
                sys.executable, str(ca_script)
            ], cwd=self.base_dir / "ca", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['ca'] = process
            
            # Wait for CA to start
            if self.wait_for_service('ca', f"http://localhost:{self.ports['ca']}/health", timeout=30):
                logger.info("CA service started successfully!")
                return True
            else:
                logger.error("CA service failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting CA service: {e}")
            return False
    
    def start_fl_server(self) -> bool:
        """Start the FL server"""
        logger.info("Starting FL server...")
        
        server_script = self.base_dir / "server" / "server.py"
        if not server_script.exists():
            logger.error("Server script not found!")
            return False
        
        try:
            env = os.environ.copy()
            env.update({
                'SERVER_PORT': str(self.ports['server']),
                'MIN_CLIENTS': '10',
                'NUM_ROUNDS': '10',
                'CA_URL': f'http://localhost:{self.ports["ca"]}',
                'ENABLE_CERTIFICATES': 'true',
                'PYTHONPATH': str(self.base_dir)
            })
            
            process = subprocess.Popen([
                sys.executable, str(server_script)
            ], cwd=self.base_dir / "server", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['server'] = process
            
            # Wait for server to start
            time.sleep(5)  # Give server time to initialize
            logger.info("FL server started successfully!")
            return True
                
        except Exception as e:
            logger.error(f"Error starting FL server: {e}")
            return False
    
    def start_fl_clients(self) -> bool:
        """Start all FL clients"""
        logger.info("Starting FL clients...")
        
        client_processes = []
        
        for client_id in range(1, 11):
            client_dir = self.base_dir / "clients" / f"client{client_id}"
            client_script = client_dir / "client.py"
            
            if not client_script.exists():
                logger.error(f"Client script not found: {client_script}")
                continue
            
            try:
                env = os.environ.copy()
                env.update({
                    'CLIENT_ID': str(client_id),
                    'SERVER_ADDRESS': f'localhost:{self.ports["server"]}',
                    'CA_URL': f'http://localhost:{self.ports["ca"]}',
                    'ENABLE_CERTIFICATES': 'true',
                    'PYTHONPATH': str(self.base_dir)
                })
                
                process = subprocess.Popen([
                    sys.executable, str(client_script)
                ], cwd=client_dir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                client_processes.append((process, client_id))
                logger.info(f"Started client {client_id}")
                
                # Small delay between client starts
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error starting client {client_id}: {e}")
        
        if client_processes:
            self.processes['clients'] = client_processes
            logger.info(f"Started {len(client_processes)} clients successfully!")
            return True
        else:
            logger.error("Failed to start any clients")
            return False
    
    def start_dashboard(self) -> bool:
        """Start the web dashboard"""
        logger.info("Starting dashboard...")
        
        # Try the local version first, fallback to original
        dashboard_script = self.base_dir / "dashboard" / "app_local.py"
        if not dashboard_script.exists():
            dashboard_script = self.base_dir / "dashboard" / "app.py"
        
        if not dashboard_script.exists():
            logger.error("Dashboard script not found!")
            return False
        
        try:
            env = os.environ.copy()
            env.update({
                'DASHBOARD_PORT': str(self.ports['dashboard']),
                'CA_URL': f'http://localhost:{self.ports["ca"]}',
                'SERVER_URL': f'http://localhost:{self.ports["server"]}',
                'FLASK_ENV': 'production',
                'PYTHONPATH': str(self.base_dir)
            })
            
            process = subprocess.Popen([
                sys.executable, str(dashboard_script)
            ], cwd=self.base_dir / "dashboard", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['dashboard'] = process
            
            # Wait for dashboard to start
            if self.wait_for_service('dashboard', f"http://localhost:{self.ports['dashboard']}/health", timeout=30):
                logger.info("Dashboard started successfully!")
                return True
            else:
                logger.error("Dashboard failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            return False
    
    def wait_for_service(self, service_name: str, url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"{service_name} is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        logger.error(f"{service_name} did not become available within {timeout} seconds")
        return False
    
    def monitor_system(self):
        """Monitor system health and performance"""
        logger.info("Starting system monitoring...")
        
        while self.running:
            try:
                # Check process health
                for service, process in self.processes.items():
                    if service == 'clients':
                        for proc, client_id in process:
                            if proc.poll() is not None:
                                logger.warning(f"Client {client_id} stopped unexpectedly")
                    else:
                        if process.poll() is not None:
                            logger.warning(f"{service} stopped unexpectedly")
                
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent}%")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)
    
    def generate_certificates(self):
        """Generate certificates for all clients"""
        logger.info("Generating certificates for clients...")
        
        ca_url = f"http://localhost:{self.ports['ca']}"
        
        for client_id in range(1, 11):
            try:
                response = requests.post(
                    f"{ca_url}/certificates/generate",
                    json={"client_id": str(client_id), "permissions": "standard"},
                    timeout=10
                )
                if response.status_code == 200:
                    logger.info(f"Certificate generated for client {client_id}")
                else:
                    logger.warning(f"Failed to generate certificate for client {client_id}")
            except Exception as e:
                logger.warning(f"Error generating certificate for client {client_id}: {e}")
    
    def start_system(self) -> bool:
        """Start the entire FL system"""
        logger.info("=" * 60)
        logger.info("Starting Federated Learning System (Local Mode)")
        logger.info("=" * 60)
        
        # Check prerequisites
        if not self.check_dependencies():
            return False
        
        if not self.check_ports():
            return False
        
        if not self.prepare_data():
            return False
        
        # Start services in order
        services = [
            ("CA Service", self.start_ca_service),
            ("FL Server", self.start_fl_server),
            ("FL Clients", self.start_fl_clients),
            ("Dashboard", self.start_dashboard)
        ]
        
        for service_name, start_func in services:
            logger.info(f"Starting {service_name}...")
            if not start_func():
                logger.error(f"Failed to start {service_name}")
                self.stop_system()
                return False
        
        # Generate certificates
        self.generate_certificates()
        
        # Start monitoring
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        monitor_thread.start()
        self.threads['monitor'] = monitor_thread
        
        logger.info("=" * 60)
        logger.info("FL System started successfully!")
        logger.info(f"Dashboard: http://localhost:{self.ports['dashboard']}")
        logger.info(f"Server: http://localhost:{self.ports['server']}")
        logger.info(f"CA Service: http://localhost:{self.ports['ca']}")
        logger.info("=" * 60)
        
        return True
    
    def stop_system(self):
        """Stop the entire FL system"""
        logger.info("Stopping FL system...")
        
        self.running = False
        
        # Stop all processes
        for service, process in self.processes.items():
            if service == 'clients':
                for proc, client_id in process:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                        logger.info(f"Client {client_id} stopped")
                    except:
                        proc.kill()
                        logger.info(f"Client {client_id} force stopped")
            else:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    logger.info(f"{service} stopped")
                except:
                    process.kill()
                    logger.info(f"{service} force stopped")
        
        # Wait for threads to finish
        for thread in self.threads.values():
            thread.join(timeout=5)
        
        logger.info("FL system stopped")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'running': self.running,
            'services': {},
            'ports': self.ports,
            'timestamp': datetime.now().isoformat()
        }
        
        for service, process in self.processes.items():
            if service == 'clients':
                status['services'][service] = {
                    'count': len(process),
                    'active': sum(1 for p, _ in process if p.poll() is None)
                }
            else:
                status['services'][service] = {
                    'running': process.poll() is None,
                    'pid': process.pid if process.poll() is None else None
                }
        
        return status
    
    def open_dashboard(self):
        """Open the dashboard in the default web browser"""
        dashboard_url = f"http://localhost:{self.ports['dashboard']}"
        logger.info(f"Opening dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal...")
    if 'fl_system' in globals():
        fl_system.stop_system()
    sys.exit(0)

def main():
    """Main execution function"""
    global fl_system
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create system manager
    fl_system = FLSystemManager()
    
    try:
        # Start the system
        if fl_system.start_system():
            # Open dashboard
            fl_system.open_dashboard()
            
            # Keep running until interrupted
            logger.info("System is running. Press Ctrl+C to stop.")
            while fl_system.running:
                time.sleep(1)
        else:
            logger.error("Failed to start FL system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        fl_system.stop_system()

if __name__ == "__main__":
    main()
