#!/usr/bin/env python3
"""
Docker Integration Setup Script
Sets up the enhanced FL system with detection in Docker
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def print_status(message, status="INFO"):
    """Print status message with formatting"""
    status_colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{status_colors.get(status, '')}[{status}]{status_colors['RESET']} {message}")

def check_docker():
    """Check if Docker is available"""
    print_status("Checking Docker availability...")
    
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status(f"✓ Docker available: {result.stdout.strip()}")
            return True
        else:
            print_status("✗ Docker not available", "ERROR")
            return False
    except FileNotFoundError:
        print_status("✗ Docker not found in PATH", "ERROR")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    print_status("Checking Docker Compose availability...")
    
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status(f"✓ Docker Compose available: {result.stdout.strip()}")
            return True
        else:
            print_status("✗ Docker Compose not available", "ERROR")
            return False
    except FileNotFoundError:
        print_status("✗ Docker Compose not found in PATH", "ERROR")
        return False

def create_docker_environment():
    """Create Docker environment configuration"""
    print_status("Creating Docker environment configuration...")
    
    env_config = {
        "DETECTION_ENABLED": "true",
        "DETECTION_METHOD": "kmeans",
        "LDP_EPSILON": "1.0",
        "SERVER_URL": "http://server:8080",
        "CA_URL": "http://ca:9000",
        "MIN_CLIENTS": "10",
        "NUM_ROUNDS": "10"
    }
    
    with open(".env", "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")
    
    print_status("✓ Docker environment configuration created", "SUCCESS")

def build_docker_images():
    """Build Docker images for the enhanced system"""
    print_status("Building Docker images...")
    
    services = ['ca', 'server', 'dashboard', 'clients']
    
    for service in services:
        print_status(f"Building {service} image...")
        try:
            result = subprocess.run(
                ['docker-compose', 'build', service],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print_status(f"✓ {service} image built successfully", "SUCCESS")
            else:
                print_status(f"✗ Failed to build {service} image", "ERROR")
                print_status(f"Error: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            print_status(f"✗ Timeout building {service} image", "ERROR")
            return False
        except Exception as e:
            print_status(f"✗ Error building {service} image: {e}", "ERROR")
            return False
    
    return True

def start_docker_services():
    """Start Docker services"""
    print_status("Starting Docker services...")
    
    try:
        # Start services in background
        result = subprocess.run(
            ['docker-compose', 'up', '-d'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        
        if result.returncode == 0:
            print_status("✓ Docker services started successfully", "SUCCESS")
            return True
        else:
            print_status("✗ Failed to start Docker services", "ERROR")
            print_status(f"Error: {result.stderr}", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("✗ Timeout starting Docker services", "ERROR")
        return False
    except Exception as e:
        print_status(f"✗ Error starting Docker services: {e}", "ERROR")
        return False

def check_service_health():
    """Check if services are healthy"""
    print_status("Checking service health...")
    
    import requests
    import time
    
    services = [
        ("Dashboard", "http://localhost:5000/health"),
        ("Server", "http://localhost:8080/health"),
        ("CA Service", "http://localhost:9000/health")
    ]
    
    for service_name, url in services:
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print_status(f"✓ {service_name} is healthy", "SUCCESS")
                    break
                else:
                    retry_count += 1
                    time.sleep(2)
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print_status(f"⏳ {service_name} not ready yet, retrying... ({retry_count}/{max_retries})")
                    time.sleep(2)
                else:
                    print_status(f"✗ {service_name} failed health check: {e}", "ERROR")
                    return False
    
    return True

def test_detection_integration():
    """Test detection integration"""
    print_status("Testing detection integration...")
    
    import requests
    
    try:
        # Test detection status endpoint
        response = requests.get("http://localhost:5000/api/detection/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print_status(f"✓ Detection status: {status.get('enabled', False)}", "SUCCESS")
        else:
            print_status("✗ Detection status endpoint failed", "WARNING")
        
        # Test detection run endpoint
        response = requests.post(
            "http://localhost:5000/api/detection/run_enhanced",
            json={"use_cached": True},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print_status("✓ Detection integration test passed", "SUCCESS")
                print_status(f"  Detected malicious: {result.get('detected_malicious', [])}")
                print_status(f"  Detection time: {result.get('detection_time', 0):.3f}s")
                return True
            else:
                print_status(f"✗ Detection test failed: {result.get('error')}", "WARNING")
                return False
        else:
            print_status(f"✗ Detection endpoint failed: {response.status_code}", "WARNING")
            return False
            
    except Exception as e:
        print_status(f"✗ Detection integration test failed: {e}", "WARNING")
        return False

def show_access_info():
    """Show access information"""
    print_status("=" * 70, "SUCCESS")
    print_status("Enhanced FL System with Detection is now running!", "SUCCESS")
    print_status("=" * 70, "SUCCESS")
    
    print_status("Access URLs:", "INFO")
    print_status("  • Dashboard: http://localhost:5000", "INFO")
    print_status("  • Server: http://localhost:8080", "INFO")
    print_status("  • CA Service: http://localhost:9000", "INFO")
    print_status("  • Prometheus: http://localhost:9090", "INFO")
    
    print_status("", "INFO")
    print_status("Detection Integration:", "INFO")
    print_status("  1. Go to http://localhost:5000", "INFO")
    print_status("  2. Click on the 'Demo' tab", "INFO")
    print_status("  3. Click 'Run Detection' button", "INFO")
    print_status("  4. View real-time detection results", "INFO")
    
    print_status("", "INFO")
    print_status("Docker Commands:", "INFO")
    print_status("  • View logs: docker-compose logs -f", "INFO")
    print_status("  • Stop system: docker-compose down", "INFO")
    print_status("  • Restart: docker-compose restart", "INFO")
    print_status("  • Rebuild: docker-compose up --build", "INFO")

def main():
    """Main integration function"""
    print_status("Starting Docker Integration Setup...", "INFO")
    print_status("=" * 70, "INFO")
    
    # Step 1: Check Docker availability
    if not check_docker():
        print_status("Docker is required but not available. Please install Docker.", "ERROR")
        return False
    
    if not check_docker_compose():
        print_status("Docker Compose is required but not available. Please install Docker Compose.", "ERROR")
        return False
    
    # Step 2: Create environment configuration
    create_docker_environment()
    
    # Step 3: Build Docker images
    if not build_docker_images():
        print_status("Failed to build Docker images. Please check the build logs.", "ERROR")
        return False
    
    # Step 4: Start Docker services
    if not start_docker_services():
        print_status("Failed to start Docker services. Please check the logs.", "ERROR")
        return False
    
    # Step 5: Check service health
    print_status("Waiting for services to start...")
    time.sleep(10)  # Give services time to start
    
    if not check_service_health():
        print_status("Some services failed health checks. Please check the logs.", "WARNING")
        print_status("You can check logs with: docker-compose logs", "INFO")
    
    # Step 6: Test detection integration
    print_status("Testing detection integration...")
    test_detection_integration()
    
    # Step 7: Show access information
    show_access_info()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
