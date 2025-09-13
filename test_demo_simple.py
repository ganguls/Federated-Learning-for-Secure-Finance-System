#!/usr/bin/env python3
"""
Simple test to check demo data functionality
"""

import requests
import json
import time

def test_demo_step_by_step():
    """Test demo data step by step"""
    print("🔍 Testing Demo Data Step by Step")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Check if dashboard is running
    print("1. Testing dashboard connectivity...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is running")
        else:
            print(f"❌ Dashboard returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        return False
    
    # Test 2: Test security status API
    print("\n2. Testing security status API...")
    try:
        response = requests.get(f"{base_url}/api/security/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Security API working")
            print(f"   Security status: {data.get('security_status')}")
            print(f"   Has simulator_status: {'simulator_status' in data}")
            if 'simulator_status' in data:
                print(f"   Has active_attacks: {'active_attacks' in data['simulator_status']}")
        else:
            print(f"❌ Security API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Security API error: {e}")
        return False
    
    # Test 3: Test client metrics API
    print("\n3. Testing client metrics API...")
    try:
        response = requests.get(f"{base_url}/api/metrics/clients", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Client metrics API working")
            print(f"   Number of clients: {len(data)}")
            print(f"   Sample client 1: {data.get('1', 'Not found')}")
        else:
            print(f"❌ Client metrics API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Client metrics API error: {e}")
        return False
    
    # Test 4: Test demo clients API
    print("\n4. Testing demo clients API...")
    try:
        response = requests.get(f"{base_url}/api/demo/clients", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Demo clients API working")
            print(f"   Number of clients: {len(data.get('clients', []))}")
            if 'clients' in data and len(data['clients']) > 0:
                print(f"   Sample client: {data['clients'][0]}")
        else:
            print(f"❌ Demo clients API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Demo clients API error: {e}")
        return False
    
    # Test 5: Test the exact frontend logic
    print("\n5. Testing frontend demo logic...")
    try:
        # Get security data
        security_response = requests.get(f"{base_url}/api/security/status", timeout=5)
        security_data = security_response.json()
        
        # Get client data
        clients_response = requests.get(f"{base_url}/api/metrics/clients", timeout=5)
        clients_data = clients_response.json()
        
        # Simulate frontend logic
        active_attacks = security_data.get('simulator_status', {}).get('active_attacks', {})
        print(f"   Active attacks: {active_attacks}")
        
        demo_clients = []
        for client_id in range(1, 11):
            client_id_str = str(client_id)
            is_malicious = client_id_str in active_attacks
            attack_type = active_attacks.get(client_id_str, 'None')
            
            client_info = clients_data.get(client_id_str, {})
            client_metrics = client_info.get('metrics', {})
            
            accuracy = client_metrics.get('accuracy', 0.8)
            loss = client_metrics.get('loss', 0.3)
            
            if is_malicious:
                accuracy = max(0.1, accuracy - 0.5)
                loss = min(2.0, loss + 0.7)
            
            demo_client = {
                'client_id': client_id,
                'is_malicious': is_malicious,
                'status': 'Malicious' if is_malicious else 'Normal',
                'accuracy': round(accuracy, 4),
                'loss': round(loss, 4),
                'attack_type': attack_type,
                'demo_port': 8081 + client_id
            }
            demo_clients.append(demo_client)
        
        print(f"✅ Demo clients created: {len(demo_clients)}")
        print("   Sample clients:")
        for i, client in enumerate(demo_clients[:3]):
            print(f"     Client {client['client_id']}: {client['status']}, Accuracy: {client['accuracy']}, Loss: {client['loss']}")
        
    except Exception as e:
        print(f"❌ Frontend logic error: {e}")
        return False
    
    print("\n🎉 All tests passed! Demo data should be working.")
    return True

if __name__ == "__main__":
    success = test_demo_step_by_step()
    if not success:
        print("\n❌ Some tests failed. Check the dashboard logs for errors.")

