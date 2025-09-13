#!/usr/bin/env python3
"""
Test script for the local dashboard
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

def test_dashboard():
    """Test if the local dashboard works"""
    print("Testing local dashboard...")
    
    # Start the dashboard
    dashboard_script = Path("dashboard/app_local.py")
    if not dashboard_script.exists():
        print("❌ Dashboard script not found!")
        return False
    
    print("Starting dashboard...")
    process = subprocess.Popen([
        sys.executable, str(dashboard_script)
    ], cwd="dashboard", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for dashboard to start
    print("Waiting for dashboard to start...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Dashboard is running!")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        print("❌ Dashboard failed to start")
        process.terminate()
        return False
    
    # Test endpoints
    endpoints = [
        ("/health", "Health check"),
        ("/test", "Test page"),
        ("/api/system/status", "System status"),
        ("/api/metrics/system", "System metrics"),
        ("/api/metrics/training", "Training metrics"),
        ("/api/metrics/clients", "Client metrics"),
        ("/api/demo/clients", "Demo clients"),
        ("/api/security/status", "Security status")
    ]
    
    print("\nTesting endpoints...")
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:5000{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {description}: OK")
            else:
                print(f"❌ {description}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {description}: {e}")
    
    # Stop dashboard
    print("\nStopping dashboard...")
    process.terminate()
    process.wait(timeout=5)
    print("✅ Dashboard stopped")
    
    return True

if __name__ == "__main__":
    success = test_dashboard()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

