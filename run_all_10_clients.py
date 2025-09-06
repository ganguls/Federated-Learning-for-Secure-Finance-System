#!/usr/bin/env python3
"""
Run all 10 Federated Learning Clients
This script starts all 10 clients for the federated learning system
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def run_client(client_id):
    """Run a single client in a separate process"""
    client_dir = Path(f"clients/client{client_id}")
    client_script = client_dir / "client.py"
    
    if client_script.exists():
        print(f"Starting Client {client_id}...")
        try:
            # Run client in background
            process = subprocess.Popen([
                sys.executable, str(client_script)
            ], cwd=client_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return process, client_id
        except Exception as e:
            print(f"Error starting Client {client_id}: {e}")
            return None, client_id
    else:
        print(f"Client script not found: {client_script}")
        return None, client_id

def main():
    """Run all 10 clients simultaneously"""
    print("Starting All 10 Federated Learning Clients...")
    print("=" * 60)
    
    # Start all 10 clients
    processes = []
    for client_id in range(1, 11):  # Clients 1-10
        process, cid = run_client(client_id)
        if process:
            processes.append((process, cid))
        time.sleep(1)  # Small delay between client starts
    
    print(f"\nStarted {len(processes)} clients")
    print("=" * 60)
    print("All 10 clients are running. Press Ctrl+C to stop all clients.")
    print("Make sure the server is also running on port 8080")
    
    try:
        # Wait for all processes to complete
        for process, client_id in processes:
            process.wait()
            print(f"Client {client_id} completed")
    except KeyboardInterrupt:
        print("\nStopping all clients...")
        for process, client_id in processes:
            try:
                process.terminate()
                print(f"Stopped Client {client_id}")
            except:
                pass
    
    print("All clients stopped.")

if __name__ == "__main__":
    main()
