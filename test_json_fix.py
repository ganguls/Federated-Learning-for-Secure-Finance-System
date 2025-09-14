#!/usr/bin/env python3
"""
Test script to verify the Flask JSON serialization fix
"""

import requests
import json
import time

def test_attack_detection_api():
    """Test the fixed attack detection API"""
    print("🧪 Testing Attack Detection API Fix")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'name': 'Basic Detection Test',
            'data': {'malicious_percentage': 20, 'attack_type': 'label_flipping', 'epsilon': 1.0}
        },
        {
            'name': 'Different Attack Type',
            'data': {'malicious_percentage': 30, 'attack_type': 'gradient_poisoning', 'epsilon': 0.5}
        },
        {
            'name': 'High Privacy Level',
            'data': {'malicious_percentage': 10, 'attack_type': 'backdoor', 'epsilon': 5.0}
        }
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test_case['name']}")
        print(f"   Data: {test_case['data']}")
        
        try:
            response = requests.post(
                "http://localhost:5000/api/detection/run_demo",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('success'):
                        print("   ✅ SUCCESS: API returned valid JSON response")
                        print(f"   📊 Result keys: {list(result.get('result', {}).keys())}")
                        success_count += 1
                    else:
                        print(f"   ❌ FAILED: API returned error: {result.get('error')}")
                except json.JSONDecodeError as e:
                    print(f"   ❌ FAILED: Invalid JSON response: {e}")
            else:
                print(f"   ❌ FAILED: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error details: {error_data}")
                except:
                    print(f"   Response text: {response.text[:200]}")
                    
        except requests.exceptions.Timeout:
            print("   ❌ FAILED: Request timeout")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
    
    print(f"\n📊 Test Results:")
    print(f"   Successful: {success_count}/{total_tests}")
    print(f"   Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("   🎉 ALL TESTS PASSED! JSON serialization fix is working!")
    elif success_count > 0:
        print("   ⚠️  Some tests passed. Partial fix working.")
    else:
        print("   ❌ All tests failed. Fix needs more work.")
    
    return success_count == total_tests

def test_other_apis():
    """Test other APIs to ensure they still work"""
    print(f"\n🔍 Testing Other APIs")
    print("=" * 50)
    
    apis = [
        ("/api/system/status", "System Status"),
        ("/api/metrics/training", "Training Metrics"),
        ("/api/metrics/clients", "Client Metrics"),
        ("/api/metrics/system", "System Metrics"),
        ("/api/detection/status", "Detection Status"),
        ("/api/detection/history", "Detection History")
    ]
    
    working_apis = 0
    
    for endpoint, name in apis:
        try:
            response = requests.get(f"http://localhost:5000{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {name}: Working")
                working_apis += 1
            else:
                print(f"   ❌ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    print(f"\n   Working APIs: {working_apis}/{len(apis)}")
    return working_apis == len(apis)

def main():
    """Main test function"""
    print("🎓 Flask JSON Serialization Fix Test")
    print("=" * 60)
    
    # Check if dashboard is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Dashboard is not running. Please start it first.")
            return
    except Exception as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        print("Please ensure Docker services are running:")
        print("   docker-compose up -d dashboard")
        return
    
    print("✅ Dashboard is running")
    
    # Test the fix
    api_fix_working = test_attack_detection_api()
    other_apis_working = test_other_apis()
    
    print(f"\n🎯 Final Results:")
    print(f"   Attack Detection API Fix: {'✅ WORKING' if api_fix_working else '❌ NOT WORKING'}")
    print(f"   Other APIs: {'✅ WORKING' if other_apis_working else '❌ NOT WORKING'}")
    
    if api_fix_working and other_apis_working:
        print(f"\n🎉 SUCCESS! The Flask JSON serialization error has been fixed!")
        print(f"   You can now use the 'Run Detection' button in the dashboard.")
    elif api_fix_working:
        print(f"\n✅ PARTIAL SUCCESS! Attack detection API is fixed.")
        print(f"   Some other APIs may have issues, but the main problem is solved.")
    else:
        print(f"\n❌ The fix needs more work. Check the error messages above.")

if __name__ == "__main__":
    main()

