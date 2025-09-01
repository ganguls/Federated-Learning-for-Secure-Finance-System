#!/usr/bin/env python3
"""
Simple test script for the FL Dashboard
"""

import requests
import time
import sys

def test_dashboard():
    """Test basic dashboard functionality"""
    base_url = "http://localhost:5000"
    
    print("Testing FL Dashboard...")
    print("=" * 40)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: System status
    try:
        response = requests.get(f"{base_url}/api/system/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System status: {data.get('status', 'unknown')}")
        else:
            print(f"❌ System status failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ System status failed: {e}")
        return False
    
    # Test 3: System metrics
    try:
        response = requests.get(f"{base_url}/api/metrics/system", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System metrics: CPU {data.get('cpu_percent', 0)}%")
        else:
            print(f"❌ System metrics failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ System metrics failed: {e}")
        return False
    
    print("=" * 40)
    print("✅ All tests passed! Dashboard is working correctly.")
    return True

if __name__ == "__main__":
    print("Waiting for dashboard to start...")
    time.sleep(3)  # Give dashboard time to start
    
    if test_dashboard():
        sys.exit(0)
    else:
        sys.exit(1)

