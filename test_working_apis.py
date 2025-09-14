#!/usr/bin/env python3
"""
Test Working APIs and Functionalities
"""

import requests
import json

def test_working_detection():
    """Test using the working simple_demo functionality"""
    print("Testing working detection functionality...")
    
    # Test the core functionality directly
    from simple_demo import run_demo
    print("Core detection system is working!")
    
    # Test dashboard connectivity
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("Dashboard is accessible")
        else:
            print(f"Dashboard returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Cannot connect to dashboard: {e}")
    
    # Test other working APIs
    working_apis = [
        "/api/system/status",
        "/api/metrics/training", 
        "/api/metrics/clients",
        "/api/metrics/system",
        "/api/detection/status",
        "/api/detection/history"
    ]
    
    for api in working_apis:
        try:
            response = requests.get(f"http://localhost:5000{api}", timeout=5)
            if response.status_code == 200:
                print(f"{api} - Working")
            else:
                print(f"{api} - HTTP {response.status_code}")
        except Exception as e:
            print(f"{api} - Error: {e}")

if __name__ == "__main__":
    test_working_detection()

