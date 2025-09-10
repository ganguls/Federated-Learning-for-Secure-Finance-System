#!/usr/bin/env python3
"""
Test script to verify dashboard chart functionality
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_dashboard_endpoints():
    """Test all dashboard API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Dashboard API Endpoints...")
    print("=" * 50)
    
    endpoints = [
        ("/api/system/status", "System Status"),
        ("/api/metrics/system", "System Metrics"),
        ("/api/metrics/clients", "Client Metrics"),
        ("/api/metrics/training", "Training Metrics"),
        ("/health", "Health Check")
    ]
    
    results = {}
    
    for endpoint, name in endpoints:
        try:
            print(f"Testing {name}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ {name}: OK")
                print(f"     Response: {json.dumps(data, indent=2)[:100]}...")
                results[endpoint] = {"status": "success", "data": data}
            else:
                print(f"  ❌ {name}: HTTP {response.status_code}")
                results[endpoint] = {"status": "error", "code": response.status_code}
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ {name}: Connection failed - Dashboard not running?")
            results[endpoint] = {"status": "error", "message": "Connection failed"}
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
            results[endpoint] = {"status": "error", "message": str(e)}
    
    return results

def test_training_metrics_format():
    """Test if training metrics have the correct format for charts"""
    base_url = "http://localhost:5000"
    
    print("\n📊 Testing Training Metrics Format...")
    print("=" * 50)
    
    try:
        response = requests.get(f"{base_url}/api/metrics/training", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            print(f"Training metrics type: {type(data)}")
            print(f"Number of rounds: {len(data) if isinstance(data, list) else 'Not a list'}")
            
            if isinstance(data, list) and len(data) > 0:
                first_round = data[0]
                print(f"First round keys: {list(first_round.keys()) if isinstance(first_round, dict) else 'Not a dict'}")
                
                if isinstance(first_round, dict):
                    required_keys = ['round', 'avg_accuracy', 'client_metrics']
                    missing_keys = [key for key in required_keys if key not in first_round]
                    
                    if not missing_keys:
                        print("  ✅ Training metrics format is correct")
                        
                        # Check client metrics
                        client_metrics = first_round.get('client_metrics', {})
                        print(f"  Number of clients: {len(client_metrics)}")
                        
                        if client_metrics:
                            first_client = list(client_metrics.values())[0]
                            client_keys = ['accuracy', 'loss', 'precision', 'recall', 'f1_score']
                            client_missing = [key for key in client_keys if key not in first_client]
                            
                            if not client_missing:
                                print("  ✅ Client metrics format is correct")
                            else:
                                print(f"  ❌ Client metrics missing keys: {client_missing}")
                        else:
                            print("  ⚠️  No client metrics found")
                    else:
                        print(f"  ❌ Training metrics missing keys: {missing_keys}")
                else:
                    print("  ❌ First round is not a dictionary")
            else:
                print("  ❌ Training metrics is not a list or is empty")
        else:
            print(f"  ❌ Failed to fetch training metrics: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error testing training metrics: {e}")

def test_dashboard_ui():
    """Test if dashboard UI loads correctly"""
    base_url = "http://localhost:5000"
    
    print("\n🖥️  Testing Dashboard UI...")
    print("=" * 50)
    
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            content = response.text
            
            # Check for essential elements
            checks = [
                ("Chart.js", "Chart.js library loaded" in content),
                ("Socket.IO", "socket.io" in content),
                ("Training Chart", "trainingChart" in content),
                ("Client Charts", "clientCharts" in content),
                ("Chart Container", "chart-container" in content),
                ("Canvas Elements", "<canvas" in content)
            ]
            
            for check_name, passed in checks:
                status = "✅" if passed else "❌"
                print(f"  {status} {check_name}")
            
            # Check for JavaScript errors
            if "console.error" in content:
                print("  ⚠️  JavaScript error handling found (good)")
            else:
                print("  ⚠️  No JavaScript error handling found")
                
        else:
            print(f"  ❌ Dashboard UI failed to load: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error testing dashboard UI: {e}")

def main():
    """Main test function"""
    print("🚀 Dashboard Chart Fix Test")
    print("=" * 50)
    
    # Test API endpoints
    api_results = test_dashboard_endpoints()
    
    # Test training metrics format
    test_training_metrics_format()
    
    # Test dashboard UI
    test_dashboard_ui()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    successful_tests = sum(1 for result in api_results.values() if result.get("status") == "success")
    total_tests = len(api_results)
    
    print(f"API Endpoints: {successful_tests}/{total_tests} successful")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! Dashboard should be working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the dashboard configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
