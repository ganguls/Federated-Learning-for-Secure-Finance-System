#!/usr/bin/env python3
"""
Test script to verify dashboard detects all 10 clients
"""

import requests
import json
import sys

def test_dashboard_endpoints():
    """Test dashboard API endpoints"""
    base_url = "http://localhost:5000"
    
    endpoints = [
        "/api/system/status",
        "/api/metrics/training", 
        "/api/metrics/clients",
        "/api/debug/containers"
    ]
    
    print("Testing Dashboard API Endpoints")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ {endpoint}")
                
                if endpoint == "/api/system/status":
                    print(f"  - Status: {data.get('status', 'unknown')}")
                    print(f"  - Total Clients: {data.get('clients_count', 0)}")
                    print(f"  - Active Clients: {data.get('active_clients', 0)}")
                
                elif endpoint == "/api/metrics/training":
                    if isinstance(data, list) and len(data) > 0:
                        latest_round = data[-1]
                        client_count = len(latest_round.get('client_metrics', {}))
                        print(f"  - Training Rounds: {len(data)}")
                        print(f"  - Clients in Latest Round: {client_count}")
                        print(f"  - Client IDs: {list(latest_round.get('client_metrics', {}).keys())}")
                
                elif endpoint == "/api/metrics/clients":
                    print(f"  - Client Metrics Count: {len(data)}")
                    print(f"  - Client IDs: {list(data.keys())}")
                
                elif endpoint == "/api/debug/containers":
                    print(f"  - Docker Available: {data.get('docker_available', False)}")
                    print(f"  - Container Count: {len(data.get('containers', {}))}")
                    print(f"  - Client Count: {data.get('client_count', 0)}")
                    print(f"  - Active Clients: {data.get('active_clients', 0)}")
                    
                    # List all client containers
                    containers = data.get('containers', {})
                    client_containers = [(name, info) for name, info in containers.items() 
                                       if info.get('type') == 'client']
                    if client_containers:
                        print("  - Client Containers:")
                        for name, info in client_containers:
                            status = info.get('status', 'unknown')
                            client_id = info.get('client_id', 'unknown')
                            print(f"    * {name}: Client {client_id} ({status})")
                
                print()
            else:
                print(f"✗ {endpoint} - HTTP {response.status_code}")
                print(f"  Error: {response.text}")
                print()
        
        except requests.exceptions.RequestException as e:
            print(f"✗ {endpoint} - Connection Error")
            print(f"  Error: {e}")
            print()
    
    print("Test completed!")

if __name__ == "__main__":
    print("Dashboard 10-Client Test")
    print("Make sure the dashboard is running on http://localhost:5000")
    print()
    
    test_dashboard_endpoints()
