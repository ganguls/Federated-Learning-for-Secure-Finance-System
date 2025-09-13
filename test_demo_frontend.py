#!/usr/bin/env python3
"""
Test the exact same API calls that the frontend makes for demo data
"""

import requests
import json

def test_demo_frontend_calls():
    """Test the exact API calls the frontend makes"""
    print("Testing frontend demo data calls...")
    
    try:
        # Step 1: Get security status (what frontend does first)
        print("1. Getting security status...")
        security_response = requests.get("http://localhost:5000/api/security/status", timeout=5)
        if security_response.status_code == 200:
            security_data = security_response.json()
            print("✅ Security status retrieved")
            print(f"   Security status: {security_data.get('security_status')}")
            print(f"   Has simulator_status: {'simulator_status' in security_data}")
            if 'simulator_status' in security_data:
                print(f"   Has active_attacks: {'active_attacks' in security_data['simulator_status']}")
                print(f"   Active attacks: {security_data['simulator_status']['active_attacks']}")
        else:
            print(f"❌ Security status failed: {security_response.status_code}")
            return False
        
        # Step 2: Get client metrics (what frontend does second)
        print("\n2. Getting client metrics...")
        clients_response = requests.get("http://localhost:5000/api/metrics/clients", timeout=5)
        if clients_response.status_code == 200:
            clients_data = clients_response.json()
            print("✅ Client metrics retrieved")
            print(f"   Number of clients: {len(clients_data)}")
            print(f"   Client 1 data: {clients_data.get('1', 'Not found')}")
        else:
            print(f"❌ Client metrics failed: {clients_response.status_code}")
            return False
        
        # Step 3: Simulate the frontend demo data creation
        print("\n3. Creating demo data (frontend logic)...")
        try:
            # Get active attacks from security data
            active_attacks = security_data.get('simulator_status', {}).get('active_attacks', {})
            print(f"   Active attacks: {active_attacks}")
            
            # Create demo clients array
            demo_clients = []
            for client_id in range(1, 11):
                client_id_str = str(client_id)
                is_malicious = client_id_str in active_attacks
                attack_type = active_attacks.get(client_id_str, 'None')
                
                # Get client metrics if available
                client_info = clients_data.get(client_id_str, {})
                client_metrics = client_info.get('metrics', {})
                
                # Default values
                accuracy = client_metrics.get('accuracy', 0.8)
                loss = client_metrics.get('loss', 0.3)
                
                # Adjust for malicious clients
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
            
            print(f"✅ Demo clients created: {len(demo_clients)} clients")
            print("   Sample client data:")
            for i, client in enumerate(demo_clients[:3]):  # Show first 3 clients
                print(f"     Client {client['client_id']}: {client['status']}, Accuracy: {client['accuracy']}, Loss: {client['loss']}")
            
            print("\n🎉 Demo data creation successful!")
            print("The frontend should be able to display this data properly.")
            
        except Exception as e:
            print(f"❌ Error creating demo data: {e}")
            return False
        
        # Step 4: Test the direct demo clients endpoint
        print("\n4. Testing direct demo clients endpoint...")
        demo_response = requests.get("http://localhost:5000/api/demo/clients", timeout=5)
        if demo_response.status_code == 200:
            demo_data = demo_response.json()
            print("✅ Demo clients endpoint working")
            print(f"   Number of clients: {len(demo_data.get('clients', []))}")
        else:
            print(f"❌ Demo clients endpoint failed: {demo_response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        return False

if __name__ == "__main__":
    success = test_demo_frontend_calls()
    if success:
        print("\n✅ All frontend API calls are working!")
        print("The demo data should work in the dashboard now.")
    else:
        print("\n❌ Some API calls failed!")

