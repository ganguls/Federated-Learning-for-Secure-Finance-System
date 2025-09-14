#!/usr/bin/env python3
"""
Comprehensive System Test for Federated Learning Attack Detection System
This script tests all functionalities and APIs to identify working and non-working components
"""

import requests
import json
import time
import sys
import subprocess
import os
from datetime import datetime

class SystemTester:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.results = {
            'working': [],
            'not_working': [],
            'errors': [],
            'summary': {}
        }
        self.start_time = datetime.now()
        
    def log_result(self, test_name, status, message="", error=None):
        """Log test results"""
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        if error:
            result['error'] = str(error)
            self.results['errors'].append(result)
        
        if status == 'PASS':
            self.results['working'].append(result)
        else:
            self.results['not_working'].append(result)
        
        print(f"{'✅' if status == 'PASS' else '❌'} {test_name}: {message}")
        if error:
            print(f"   Error: {error}")

    def test_connection(self):
        """Test basic connectivity"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.log_result("Basic Connectivity", "PASS", "Dashboard is accessible")
                return True
            else:
                self.log_result("Basic Connectivity", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Basic Connectivity", "FAIL", "Cannot connect to dashboard", e)
            return False

    def test_dashboard_pages(self):
        """Test dashboard pages"""
        pages = [
            ("/", "Main Dashboard"),
            ("/research", "Research Demo Page"),
            ("/health", "Health Check")
        ]
        
        for endpoint, name in pages:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    self.log_result(f"Page: {name}", "PASS", f"Accessible (HTTP {response.status_code})")
                else:
                    self.log_result(f"Page: {name}", "FAIL", f"HTTP {response.status_code}")
            except Exception as e:
                self.log_result(f"Page: {name}", "FAIL", "Cannot access", e)

    def test_api_endpoints(self):
        """Test all API endpoints"""
        api_tests = [
            # System APIs
            ("/api/system/status", "GET", "System Status API"),
            ("/api/metrics/training", "GET", "Training Metrics API"),
            ("/api/metrics/clients", "GET", "Client Metrics API"),
            ("/api/metrics/system", "GET", "System Metrics API"),
            
            # Demo APIs
            ("/api/demo/clients", "GET", "Demo Clients API"),
            
            # Detection APIs
            ("/api/detection/status", "GET", "Detection Status API"),
            ("/api/detection/history", "GET", "Detection History API"),
        ]
        
        for endpoint, method, name in api_tests:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                elif method == "POST":
                    response = requests.post(f"{self.base_url}{endpoint}", timeout=5)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        self.log_result(f"API: {name}", "PASS", f"Returns valid JSON with {len(data)} keys")
                    except json.JSONDecodeError:
                        self.log_result(f"API: {name}", "PASS", f"Returns content (not JSON)")
                else:
                    self.log_result(f"API: {name}", "FAIL", f"HTTP {response.status_code}")
            except Exception as e:
                self.log_result(f"API: {name}", "FAIL", "Request failed", e)

    def test_attack_detection_api(self):
        """Test the problematic attack detection API"""
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
        
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/api/detection/run_demo",
                    json=test_case['data'],
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if result.get('success'):
                            self.log_result(f"Detection API: {test_case['name']}", "PASS", "Successfully executed")
                        else:
                            self.log_result(f"Detection API: {test_case['name']}", "FAIL", f"API returned error: {result.get('error', 'Unknown error')}")
                    except json.JSONDecodeError:
                        self.log_result(f"Detection API: {test_case['name']}", "FAIL", "Invalid JSON response")
                else:
                    self.log_result(f"Detection API: {test_case['name']}", "FAIL", f"HTTP {response.status_code}")
            except Exception as e:
                self.log_result(f"Detection API: {test_case['name']}", "FAIL", "Request failed", e)

    def test_docker_services(self):
        """Test Docker services status"""
        try:
            result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                services = result.stdout
                service_count = len([line for line in services.split('\n') if 'flsystem-' in line and 'Up' in line])
                self.log_result("Docker Services", "PASS", f"{service_count} services running")
            else:
                self.log_result("Docker Services", "FAIL", "Cannot check Docker status")
        except Exception as e:
            self.log_result("Docker Services", "FAIL", "Docker command failed", e)

    def test_system_resources(self):
        """Test system resource usage"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.log_result("System Resources", "PASS", 
                f"CPU: {cpu_percent}%, Memory: {memory.percent}% ({memory.used // (1024**3)}GB used)")
        except ImportError:
            self.log_result("System Resources", "FAIL", "psutil not available")
        except Exception as e:
            self.log_result("System Resources", "FAIL", "Cannot check resources", e)

    def test_file_system(self):
        """Test important files and directories"""
        important_files = [
            'dashboard/app.py',
            'dashboard/requirements.txt',
            'docker-compose.yml',
            'server/server.py',
            'simple_demo.py',
            'comprehensive_system_test.py'
        ]
        
        for file_path in important_files:
            if os.path.exists(file_path):
                self.log_result(f"File: {file_path}", "PASS", "Exists")
            else:
                self.log_result(f"File: {file_path}", "FAIL", "Missing")

    def test_attack_detection_core(self):
        """Test the core attack detection functionality directly"""
        try:
            # Import and test the core detection logic
            sys.path.append('.')
            from simple_demo import apply_ldp, simple_detection
            
            # Test LDP function
            test_losses = [0.1, 0.2, 0.8, 0.3, 0.9]
            noisy_losses = apply_ldp(test_losses, epsilon=1.0)
            
            if len(noisy_losses) == len(test_losses):
                self.log_result("Core LDP Function", "PASS", "LDP noise application works")
            else:
                self.log_result("Core LDP Function", "FAIL", "LDP output length mismatch")
            
            # Test detection function
            true_malicious = [2, 4]  # Indices of malicious clients
            detected, metrics = simple_detection(test_losses, true_malicious, epsilon=1.0)
            
            if isinstance(detected, list) and isinstance(metrics, dict):
                self.log_result("Core Detection Function", "PASS", f"Detected {len(detected)} malicious clients")
            else:
                self.log_result("Core Detection Function", "FAIL", "Invalid return types")
                
        except Exception as e:
            self.log_result("Core Detection Function", "FAIL", "Import or execution failed", e)

    def test_web_interface(self):
        """Test web interface elements"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                content = response.text
                
                # Check for important elements
                checks = [
                    ("HTML Content", "<html" in content.lower()),
                    ("Dashboard Title", "dashboard" in content.lower() or "federated" in content.lower()),
                    ("JavaScript", "<script" in content.lower()),
                    ("CSS Styling", "<style" in content.lower() or "css" in content.lower())
                ]
                
                for check_name, condition in checks:
                    if condition:
                        self.log_result(f"Web Interface: {check_name}", "PASS", "Found in HTML")
                    else:
                        self.log_result(f"Web Interface: {check_name}", "FAIL", "Not found in HTML")
            else:
                self.log_result("Web Interface", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_result("Web Interface", "FAIL", "Cannot access", e)

    def generate_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        total_tests = len(self.results['working']) + len(self.results['not_working'])
        pass_rate = (len(self.results['working']) / total_tests * 100) if total_tests > 0 else 0
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed': len(self.results['working']),
            'failed': len(self.results['not_working']),
            'pass_rate': pass_rate,
            'duration_seconds': duration,
            'test_time': self.start_time.isoformat()
        }
        
        print("\n" + "="*80)
        print("🎓 FEDERATED LEARNING SYSTEM - COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {len(self.results['working'])}")
        print(f"   Failed: {len(self.results['not_working'])}")
        print(f"   Pass Rate: {pass_rate:.1f}%")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Test Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.results['working']:
            print(f"\n✅ WORKING FUNCTIONALITIES ({len(self.results['working'])}):")
            for result in self.results['working']:
                print(f"   • {result['test']}: {result['message']}")
        
        if self.results['not_working']:
            print(f"\n❌ NON-WORKING FUNCTIONALITIES ({len(self.results['not_working'])}):")
            for result in self.results['not_working']:
                print(f"   • {result['test']}: {result['message']}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
        
        if self.results['errors']:
            print(f"\n🚨 ERRORS ENCOUNTERED ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"   • {error['test']}: {error['error']}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if pass_rate >= 90:
            print("   🎉 Excellent! Your system is working very well.")
        elif pass_rate >= 70:
            print("   ✅ Good! Most functionalities are working. Fix the failed ones.")
        elif pass_rate >= 50:
            print("   ⚠️  Fair. Several issues need attention.")
        else:
            print("   🚨 Poor. Many critical functionalities are not working.")
        
        print(f"\n📁 Full report saved to: comprehensive_test_report.json")
        
        # Save detailed report
        with open('comprehensive_test_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return self.results

    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive System Test...")
        print("="*80)
        
        # Core functionality tests
        self.test_connection()
        if not any(r['test'] == 'Basic Connectivity' and r['status'] == 'PASS' for r in self.results['working']):
            print("❌ Cannot connect to dashboard. Please ensure Docker services are running.")
            return self.results
        
        # System tests
        self.test_docker_services()
        self.test_system_resources()
        self.test_file_system()
        
        # Web interface tests
        self.test_dashboard_pages()
        self.test_web_interface()
        
        # API tests
        self.test_api_endpoints()
        self.test_attack_detection_api()
        
        # Core functionality tests
        self.test_attack_detection_core()
        
        # Generate report
        return self.generate_report()

def main():
    """Main test execution"""
    print("🎓 Federated Learning Attack Detection System")
    print("🔍 Comprehensive Functionality Test Suite")
    print("="*80)
    
    tester = SystemTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['summary']['pass_rate'] >= 70:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Test completed with issues. Check the report above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

