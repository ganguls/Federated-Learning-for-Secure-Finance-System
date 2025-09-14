#!/usr/bin/env python3
"""
Test script for dashboard attack detection integration
====================================================

This script tests the attack detection integration in the dashboard.

Usage:
    python test_dashboard_integration.py
"""

import requests
import json
import time
import sys

def test_dashboard_integration():
    """Test the dashboard attack detection integration"""
    base_url = "http://localhost:5000"
    
    print("Testing Dashboard Attack Detection Integration")
    print("=" * 50)
    
    # Test 1: Check if dashboard is running
    print("1. Testing dashboard connectivity...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Dashboard is running")
        else:
            print(f"   ❌ Dashboard returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to dashboard: {e}")
        return False
    
    # Test 2: Test detection status endpoint
    print("2. Testing detection status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/detection/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Detection status: {data.get('enabled', 'Unknown')}")
            print(f"   ✅ Malicious clients: {data.get('malicious_clients', [])}")
        else:
            print(f"   ❌ Detection status endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing detection status: {e}")
    
    # Test 3: Test detection metrics endpoint
    print("3. Testing detection metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/api/detection/metrics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Detection metrics: {data}")
        else:
            print(f"   ❌ Detection metrics endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing detection metrics: {e}")
    
    # Test 4: Test running detection demo
    print("4. Testing detection demo...")
    try:
        response = requests.post(f"{base_url}/api/detection/run_demo", 
                               json={"malicious_percentage": 20}, 
                               timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('result', {})
                print(f"   ✅ Detection demo completed successfully")
                print(f"   ✅ True malicious: {result.get('true_malicious', [])}")
                print(f"   ✅ Detected malicious: {result.get('detected_malicious', [])}")
                print(f"   ✅ Accuracy: {result.get('metrics', {}).get('accuracy', 0):.3f}")
                print(f"   ✅ F1 Score: {result.get('metrics', {}).get('f1_score', 0):.3f}")
            else:
                print(f"   ❌ Detection demo failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Detection demo endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing detection demo: {e}")
    
    # Test 5: Test detection history
    print("5. Testing detection history...")
    try:
        response = requests.get(f"{base_url}/api/detection/history", timeout=5)
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', [])
            print(f"   ✅ Detection history: {len(history)} entries")
            if history:
                latest = history[-1]
                print(f"   ✅ Latest detection: {latest.get('timestamp', 'Unknown')}")
        else:
            print(f"   ❌ Detection history endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing detection history: {e}")
    
    # Test 6: Test toggle detection
    print("6. Testing toggle detection...")
    try:
        response = requests.post(f"{base_url}/api/detection/toggle", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                enabled = data.get('enabled', False)
                print(f"   ✅ Detection toggled successfully: {'Enabled' if enabled else 'Disabled'}")
            else:
                print(f"   ❌ Toggle detection failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Toggle detection endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error testing toggle detection: {e}")
    
    print("\n" + "=" * 50)
    print("Integration test completed!")
    print("Check the dashboard at http://localhost:5000 and go to the Demo tab")
    print("to see the attack detection functionality in action.")
    
    return True

if __name__ == "__main__":
    success = test_dashboard_integration()
    sys.exit(0 if success else 1)



