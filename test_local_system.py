#!/usr/bin/env python3
"""
Test script for the local FL system
Verifies that all components can start and communicate properly
"""

import time
import requests
import subprocess
import sys
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    required_packages = [
        'flwr', 'numpy', 'pandas', 'sklearn', 'joblib',
        'flask', 'flask_socketio', 'flask_cors', 'requests',
        'cryptography', 'psutil'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Install with: pip install -r requirements_local.txt")
        return False
    
    print("All dependencies available!")
    return True

def test_ports():
    """Test if required ports are available"""
    print("\nTesting port availability...")
    
    ports = [5000, 8080, 9000] + list(range(8082, 8092))
    
    for port in ports:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) == 0:
                    print(f"  ✗ Port {port} is in use")
                    return False
                else:
                    print(f"  ✓ Port {port} is available")
        except Exception as e:
            print(f"  ✗ Error checking port {port}: {e}")
            return False
    
    print("All ports are available!")
    return True

def test_data_files():
    """Test if data files exist"""
    print("\nTesting data files...")
    
    data_dir = Path("Datapre/FL_clients")
    if not data_dir.exists():
        print("  ✗ Data directory not found")
        return False
    
    client_files = list(data_dir.glob("client_*.csv"))
    if len(client_files) < 10:
        print(f"  ✗ Only {len(client_files)} client files found (need 10)")
        return False
    
    print(f"  ✓ Found {len(client_files)} client data files")
    return True

def test_scripts():
    """Test if all required scripts exist"""
    print("\nTesting script files...")
    
    scripts = [
        "ca/ca_service.py",
        "server/server.py", 
        "dashboard/app.py",
        "clients/client1/client.py",
        "Datapre/complete_datapre.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (missing)")
            return False
    
    print("All required scripts found!")
    return True

def test_system_startup():
    """Test if the system can start (brief test)"""
    print("\nTesting system startup...")
    
    try:
        # Import the system manager
        sys.path.append(str(Path(__file__).parent))
        from run_fl_system_local import FLSystemManager
        
        # Create manager instance
        manager = FLSystemManager()
        
        # Test individual components
        print("  Testing CA service startup...")
        if manager.start_ca_service():
            print("  ✓ CA service started")
            time.sleep(2)
            manager.stop_system()
        else:
            print("  ✗ CA service failed to start")
            return False
        
        print("  ✓ System startup test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ System startup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("FL System Local Execution Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Ports", test_ports),
        ("Data Files", test_data_files),
        ("Scripts", test_scripts),
        ("System Startup", test_system_startup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! System is ready to run.")
        print("\nTo start the system:")
        print("  python run_fl_system_local.py")
        print("  or")
        print("  ./start_fl_system.sh (Linux/Mac)")
        print("  start_fl_system.bat (Windows)")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

