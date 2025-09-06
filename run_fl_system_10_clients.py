#!/usr/bin/env python3
"""
Federated Learning System for Loan Prediction - 10 Clients Version
Main execution script that runs the server and all 10 clients
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = ['flwr', 'numpy', 'pandas', 'sklearn', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install them using: pip install -r server/requirements.txt")
        return False
    
    print("All dependencies are available!")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("Checking data files...")
    
    data_path = Path("Datapre/FL_clients")
    if not data_path.exists():
        print("Data directory not found. Running data preprocessing...")
        return run_data_preprocessing()
    
    # Check if we have all 10 client data files
    client_files = list(data_path.glob("client_*.csv"))
    if len(client_files) < 10:
        print(f"Only {len(client_files)} client data files found. Need all 10.")
        return run_data_preprocessing()
    
    print(f"Found {len(client_files)} client data files.")
    return True

def run_data_preprocessing():
    """Run the data preprocessing script"""
    print("Running data preprocessing...")
    
    preprocess_script = Path("Datapre/complete_datapre.py")
    if not preprocess_script.exists():
        print("Data preprocessing script not found!")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(preprocess_script)
        ], cwd="Datapre", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Data preprocessing completed successfully!")
            return True
        else:
            print(f"Data preprocessing failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running data preprocessing: {e}")
        return False

def start_server():
    """Start the federated learning server"""
    print("Starting federated learning server...")
    
    server_script = Path("server/server.py")
    if not server_script.exists():
        print("Server script not found!")
        return None
    
    try:
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, str(server_script)
        ], cwd="server")
        
        # Wait a bit for server to start
        time.sleep(5)
        
        if server_process.poll() is None:
            print("Server started successfully!")
            return server_process
        else:
            print("Server failed to start!")
            return None
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

def start_all_clients():
    """Start all 10 federated learning clients"""
    print("Starting all 10 clients...")
    
    client_processes = []
    
    for client_id in range(1, 11):  # All 10 clients
        client_dir = Path(f"clients/client{client_id}")
        client_script = client_dir / "client.py"
        
        if not client_script.exists():
            print(f"Client script not found: {client_script}")
            continue
        
        try:
            # Start client in background
            client_process = subprocess.Popen([
                sys.executable, str(client_script)
            ], cwd=client_dir)
            
            client_processes.append((client_process, client_id))
            print(f"Started client {client_id}")
            
            # Small delay between client starts
            time.sleep(2)
            
        except Exception as e:
            print(f"Error starting client {client_id}: {e}")
    
    return client_processes

def monitor_training(server_process, client_processes, max_rounds=5):
    """Monitor the training process"""
    print(f"Monitoring training for {max_rounds} rounds...")
    print("=" * 60)
    
    round_count = 0
    start_time = time.time()
    
    try:
        while round_count < max_rounds:
            # Check if server is still running
            if server_process.poll() is not None:
                print("Server stopped unexpectedly!")
                break
            
            # Check client status
            active_clients = []
            for process, client_id in client_processes:
                if process.poll() is None:
                    active_clients.append(client_id)
            
            print(f"Round {round_count + 1}: {len(active_clients)} active clients")
            
            # Wait for round completion (simplified)
            time.sleep(30)  # Assume each round takes ~30 seconds
            round_count += 1
            
            # Check for training metrics
            metrics_file = Path("server/training_metrics.json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                print(f"Training metrics: {len(metrics)} rounds completed")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

def cleanup(server_process, client_processes):
    """Clean up processes"""
    print("Cleaning up processes...")
    
    # Stop server
    if server_process:
        server_process.terminate()
        print("Server stopped")
    
    # Stop clients
    for process, client_id in client_processes:
        try:
            process.terminate()
            print(f"Client {client_id} stopped")
        except:
            pass

def main():
    """Main execution function"""
    print("=" * 70)
    print("Federated Learning System for Loan Prediction - 10 Clients")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check data files
    if not check_data_files():
        print("Failed to prepare data files. Exiting.")
        return
    
    # Start server
    server_process = start_server()
    if not server_process:
        print("Failed to start server. Exiting.")
        return
    
    # Start all 10 clients
    client_processes = start_all_clients()
    if not client_processes:
        print("Failed to start any clients. Exiting.")
        cleanup(server_process, [])
        return
    
    print(f"Started {len(client_processes)} clients")
    
    try:
        # Monitor training
        monitor_training(server_process, client_processes)
        
    except Exception as e:
        print(f"Error during training: {e}")
    
    finally:
        # Cleanup
        cleanup(server_process, client_processes)
    
    print("Federated learning system with 10 clients completed!")

if __name__ == "__main__":
    main()
