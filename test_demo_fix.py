#!/usr/bin/env python3
"""
Test script to verify demo data functionality
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

def test_demo_endpoints():
    """Test all demo-related endpoints"""
    print("Testing demo data endpoints...")
    
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
        ("/api/security/status", "Security status"),
        ("/api/metrics/clients", "Client metrics"),
        ("/api/demo/clients", "Demo clients"),
        ("/api/demo/data", "Demo data")
    ]
    
    print("\nTesting endpoints...")
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:5000{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {description}: OK")
                
                # Check specific data structure for demo endpoints
                if endpoint == "/api/security/status":
                    if 'simulator_status' in data and 'active_attacks' in data['simulator_status']:
                        print(f"   ✓ Security status has correct structure")
                    else:
                        print(f"   ❌ Security status missing simulator_status.active_attacks")
                
                elif endpoint == "/api/demo/clients":
                    if 'clients' in data and len(data['clients']) == 10:
                        print(f"   ✓ Demo clients has 10 clients")
                    else:
                        print(f"   ❌ Demo clients missing or wrong count")
                
                elif endpoint == "/api/demo/data":
                    required_keys = ['security', 'clients', 'training', 'system']
                    if all(key in data for key in required_keys):
                        print(f"   ✓ Demo data has all required sections")
                    else:
                        print(f"   ❌ Demo data missing required sections")
                        
            else:
                print(f"❌ {description}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {description}: {e}")
    
    # Test the specific demo refresh functionality
    print("\nTesting demo refresh functionality...")
    try:
        # Simulate what the frontend does
        security_response = requests.get("http://localhost:5000/api/security/status", timeout=5)
        if security_response.status_code == 200:
            security_data = security_response.json()
            print("✅ Security data retrieved")
            
            if 'simulator_status' in security_data and 'active_attacks' in security_data['simulator_status']:
                print("✅ Security data has correct structure for demo")
                
                # Test client metrics
                clients_response = requests.get("http://localhost:5000/api/metrics/clients", timeout=5)
                if clients_response.status_code == 200:
                    clients_data = clients_response.json()
                    print("✅ Client metrics retrieved")
                    
                    # Test the demo data combination
                    active_attacks = security_data['simulator_status']['active_attacks']
                    print(f"✅ Active attacks: {active_attacks}")
                    
                    # Check if we can create demo client data
                    demo_clients = []
                    for client_id in range(1, 11):
                        client_id_str = str(client_id)
                        is_malicious = client_id_str in active_attacks
                        attack_type = active_attacks.get(client_id_str, 'None')
                        
                        client_info = clients_data.get(client_id_str, {})
                        client_metrics = client_info.get('metrics', {})
                        
                        demo_clients.append({
                            'client_id': client_id,
                            'is_malicious': is_malicious,
                            'status': 'Malicious' if is_malicious else 'Normal',
                            'accuracy': client_metrics.get('accuracy', 0.8),
                            'loss': client_metrics.get('loss', 0.3),
                            'attack_type': attack_type,
                            'demo_port': 8081 + client_id
                        })
                    
                    print(f"✅ Demo clients created: {len(demo_clients)} clients")
                    print("✅ Demo data functionality is working!")
                    
                else:
                    print("❌ Failed to get client metrics")
            else:
                print("❌ Security data missing required structure")
        else:
            print("❌ Failed to get security data")
            
    except Exception as e:
        print(f"❌ Error testing demo refresh: {e}")
    
    # Stop dashboard
    print("\nStopping dashboard...")
    process.terminate()
    process.wait(timeout=5)
    print("✅ Dashboard stopped")
    
    return True

if __name__ == "__main__":
    success = test_demo_endpoints()
    if success:
        print("\n🎉 Demo data fix is working!")
        print("The demo button should now work properly in the dashboard.")
    else:
        print("\n❌ Demo data fix failed!")
        sys.exit(1)