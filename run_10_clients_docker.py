#!/usr/bin/env python3
"""
Run all 10 Federated Learning Clients using Docker Compose
This script manages the Docker containers for all 10 clients
"""

import subprocess
import sys
import time
import os

def run_docker_command(command):
    """Run a docker command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_docker():
    """Check if Docker and Docker Compose are available"""
    print("Checking Docker availability...")
    
    # Check Docker
    success, stdout, stderr = run_docker_command("docker --version")
    if not success:
        print("Docker is not available or not running!")
        return False
    print(f"Docker: {stdout.strip()}")
    
    # Check Docker Compose
    success, stdout, stderr = run_docker_command("docker-compose --version")
    if not success:
        print("Docker Compose is not available!")
        return False
    print(f"Docker Compose: {stdout.strip()}")
    
    return True

def start_all_services():
    """Start all services including server and all 10 clients"""
    print("Starting all services (CA, Server, Dashboard, and all 10 clients)...")
    print("=" * 70)
    
    # Start all services
    success, stdout, stderr = run_docker_command("docker-compose up -d")
    if not success:
        print(f"Failed to start services: {stderr}")
        return False
    
    print("All services started successfully!")
    return True

def start_clients_only():
    """Start only the client services (assumes server is already running)"""
    print("Starting all 10 client services...")
    print("=" * 50)
    
    client_services = [f"client{i}" for i in range(1, 11)]
    
    for client in client_services:
        success, stdout, stderr = run_docker_command(f"docker-compose up -d {client}")
        if success:
            print(f"✓ Started {client}")
        else:
            print(f"✗ Failed to start {client}: {stderr}")
        time.sleep(1)
    
    print("All 10 clients started!")

def stop_all_services():
    """Stop all services"""
    print("Stopping all services...")
    success, stdout, stderr = run_docker_command("docker-compose down")
    if success:
        print("All services stopped successfully!")
    else:
        print(f"Failed to stop services: {stderr}")

def stop_clients_only():
    """Stop only client services"""
    print("Stopping all client services...")
    
    client_services = [f"client{i}" for i in range(1, 11)]
    
    for client in client_services:
        success, stdout, stderr = run_docker_command(f"docker-compose stop {client}")
        if success:
            print(f"✓ Stopped {client}")
        else:
            print(f"✗ Failed to stop {client}: {stderr}")

def show_status():
    """Show status of all services"""
    print("Service Status:")
    print("=" * 50)
    success, stdout, stderr = run_docker_command("docker-compose ps")
    if success:
        print(stdout)
    else:
        print(f"Failed to get status: {stderr}")

def show_logs():
    """Show logs from all services"""
    print("Showing logs from all services...")
    success, stdout, stderr = run_docker_command("docker-compose logs --tail=50")
    if success:
        print(stdout)
    else:
        print(f"Failed to get logs: {stderr}")

def main():
    """Main function"""
    if not check_docker():
        return
    
    print("=" * 70)
    print("Federated Learning System - Docker Management")
    print("10 Clients Configuration")
    print("=" * 70)
    
    while True:
        print("\nOptions:")
        print("1. Start all services (CA, Server, Dashboard, 10 Clients)")
        print("2. Start only 10 clients (assumes server is running)")
        print("3. Stop all services")
        print("4. Stop only clients")
        print("5. Show service status")
        print("6. Show logs")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            start_all_services()
        elif choice == "2":
            start_clients_only()
        elif choice == "3":
            stop_all_services()
        elif choice == "4":
            stop_clients_only()
        elif choice == "5":
            show_status()
        elif choice == "6":
            show_logs()
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
