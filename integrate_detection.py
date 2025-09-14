#!/usr/bin/env python3
"""
Detection Integration Script
Automates the integration of data poisoning detection into the FL system
"""

import os
import sys
import shutil
import json
import subprocess
from pathlib import Path

def print_status(message, status="INFO"):
    """Print status message with formatting"""
    status_colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{status_colors.get(status, '')}[{status}]{status_colors['RESET']} {message}")

def check_dependencies():
    """Check if required dependencies are available"""
    print_status("Checking dependencies...")
    
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'flask',
        'flask-cors'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print_status(f"✗ {package} is missing", "WARNING")
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        print_status("Please install missing packages before proceeding")
        return False
    
    return True

def backup_existing_files():
    """Backup existing files before integration"""
    print_status("Creating backup of existing files...")
    
    backup_dir = Path("backup_before_detection_integration")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "server/server.py",
        "dashboard/app.py",
        "dashboard/templates/index.html"
    ]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = backup_dir / Path(file_path).name
            shutil.copy2(file_path, backup_path)
            print_status(f"✓ Backed up {file_path}")
    
    print_status(f"Backup created in {backup_dir}", "SUCCESS")

def verify_detection_system():
    """Verify that the detection system files are available"""
    print_status("Verifying detection system files...")
    
    detection_files = [
        "data_poisoning_detection/detection_utils.py",
        "data_poisoning_detection/federated_detection.py",
        "data_poisoning_detection/tabular_model.py",
        "data_poisoning_detection/tabular_data_utils.py"
    ]
    
    missing_files = []
    for file_path in detection_files:
        if os.path.exists(file_path):
            print_status(f"✓ {file_path} found")
        else:
            missing_files.append(file_path)
            print_status(f"✗ {file_path} missing", "WARNING")
    
    if missing_files:
        print_status(f"Missing detection files: {', '.join(missing_files)}", "ERROR")
        return False
    
    return True

def create_integration_config():
    """Create integration configuration file"""
    print_status("Creating integration configuration...")
    
    config = {
        "detection": {
            "enabled": True,
            "method": "kmeans",
            "ldp_epsilon": 1.0,
            "ldp_sensitivity": 0.001,
            "input_dim": 20,
            "defense_threshold": 0.3
        },
        "server": {
            "use_enhanced_strategy": True,
            "detection_frequency": 1,  # Every round
            "enable_logging": True
        },
        "dashboard": {
            "enable_enhanced_detection": True,
            "show_detection_metrics": True,
            "auto_refresh": True
        }
    }
    
    with open("detection_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print_status("✓ Integration configuration created", "SUCCESS")

def update_requirements():
    """Update requirements.txt with detection dependencies"""
    print_status("Updating requirements.txt...")
    
    detection_requirements = [
        "torch>=1.8.0",
        "scipy>=1.7.0"
    ]
    
    requirements_file = "requirements.txt"
    existing_requirements = set()
    
    if os.path.exists(requirements_file):
        with open(requirements_file, "r") as f:
            existing_requirements = set(line.strip() for line in f if line.strip())
    
    new_requirements = []
    for req in detection_requirements:
        if req not in existing_requirements:
            new_requirements.append(req)
    
    if new_requirements:
        with open(requirements_file, "a") as f:
            f.write("\n# Detection system requirements\n")
            for req in new_requirements:
                f.write(f"{req}\n")
        print_status(f"✓ Added {len(new_requirements)} new requirements", "SUCCESS")
    else:
        print_status("✓ All detection requirements already present", "SUCCESS")

def create_startup_script():
    """Create startup script for enhanced system"""
    print_status("Creating startup script...")
    
    startup_script = """#!/bin/bash
# Enhanced FL System Startup Script

echo "Starting Enhanced FL System with Detection..."

# Start CA service (if enabled)
if [ "$ENABLE_CA" = "true" ]; then
    echo "Starting CA service..."
    cd ca && python ca_service.py &
    cd ..
fi

# Start FL server with enhanced detection
echo "Starting FL server with enhanced detection..."
cd server && python enhanced_server_strategy.py &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Start dashboard
echo "Starting dashboard..."
cd ../dashboard && python app.py &
DASHBOARD_PID=$!

echo "Enhanced FL system started!"
echo "Server PID: $SERVER_PID"
echo "Dashboard PID: $DASHBOARD_PID"
echo "Dashboard URL: http://localhost:5000"
echo "Server URL: http://localhost:8080"

# Wait for processes
wait
"""
    
    with open("start_enhanced_system.sh", "w") as f:
        f.write(startup_script)
    
    os.chmod("start_enhanced_system.sh", 0o755)
    print_status("✓ Startup script created", "SUCCESS")

def create_test_script():
    """Create test script for detection integration"""
    print_status("Creating test script...")
    
    test_script = """#!/usr/bin/env python3
\"\"\"
Test script for detection integration
\"\"\"

import sys
import os
import requests
import time
import json

def test_detection_api():
    \"\"\"Test detection API endpoints\"\"\"
    print("Testing detection API...")
    
    base_url = "http://localhost:5000"
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/api/detection/status")
        if response.status_code == 200:
            print("✓ Status endpoint working")
            print(f"  Status: {response.json()}")
        else:
            print(f"✗ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Status endpoint error: {e}")
    
    # Test run detection endpoint
    try:
        response = requests.post(f"{base_url}/api/detection/run_enhanced", 
                               json={"use_cached": True})
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✓ Detection endpoint working")
                print(f"  Detected malicious: {result.get('detected_malicious', [])}")
                print(f"  Detection time: {result.get('detection_time', 0):.3f}s")
            else:
                print(f"✗ Detection failed: {result.get('error')}")
        else:
            print(f"✗ Detection endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Detection endpoint error: {e}")

def test_server_integration():
    \"\"\"Test server integration\"\"\"
    print("Testing server integration...")
    
    try:
        from server.enhanced_server_strategy import create_enhanced_strategy
        
        strategy = create_enhanced_strategy(
            detection_config={
                'enabled': True,
                'method': 'kmeans',
                'ldp_epsilon': 1.0
            }
        )
        
        print("✓ Enhanced server strategy created successfully")
        print(f"  Detection enabled: {strategy.detection_enabled}")
        print(f"  Detection method: {strategy.detection_config.get('method')}")
        
    except Exception as e:
        print(f"✗ Server integration test failed: {e}")

def test_detection_adapter():
    \"\"\"Test detection adapter\"\"\"
    print("Testing detection adapter...")
    
    try:
        from server.detection_adapter import create_detection_adapter
        
        adapter = create_detection_adapter(
            input_dim=20,
            detection_method='kmeans',
            ldp_epsilon=1.0
        )
        
        print("✓ Detection adapter created successfully")
        print(f"  Input dim: {adapter.input_dim}")
        print(f"  Detection method: {adapter.detection_method}")
        
    except Exception as e:
        print(f"✗ Detection adapter test failed: {e}")

if __name__ == "__main__":
    print("Running detection integration tests...")
    print("=" * 50)
    
    test_detection_adapter()
    test_server_integration()
    
    print("\\nTesting API endpoints (requires running system)...")
    print("Make sure to start the system first:")
    print("  python dashboard/app.py")
    print("  python server/enhanced_server_strategy.py")
    print()
    
    test_detection_api()
    
    print("\\nTest completed!")
"""
    
    with open("test_detection_integration.py", "w") as f:
        f.write(test_script)
    
    os.chmod("test_detection_integration.py", 0o755)
    print_status("✓ Test script created", "SUCCESS")

def main():
    """Main integration function"""
    print_status("Starting detection integration process...", "INFO")
    print_status("=" * 60, "INFO")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print_status("Dependency check failed. Please install missing packages.", "ERROR")
        return False
    
    # Step 2: Backup existing files
    backup_existing_files()
    
    # Step 3: Verify detection system
    if not verify_detection_system():
        print_status("Detection system verification failed.", "ERROR")
        return False
    
    # Step 4: Create configuration
    create_integration_config()
    
    # Step 5: Update requirements
    update_requirements()
    
    # Step 6: Create startup script
    create_startup_script()
    
    # Step 7: Create test script
    create_test_script()
    
    print_status("=" * 60, "SUCCESS")
    print_status("Detection integration completed successfully!", "SUCCESS")
    print_status("=" * 60, "SUCCESS")
    
    print_status("Next steps:", "INFO")
    print_status("1. Review the integration configuration in detection_config.json", "INFO")
    print_status("2. Install any missing dependencies: pip install -r requirements.txt", "INFO")
    print_status("3. Test the integration: python test_detection_integration.py", "INFO")
    print_status("4. Start the enhanced system: ./start_enhanced_system.sh", "INFO")
    print_status("5. Access the dashboard at http://localhost:5000", "INFO")
    print_status("6. Go to the Demo tab and click 'Run Detection'", "INFO")
    
    print_status("For detailed instructions, see DETECTION_INTEGRATION_GUIDE.md", "INFO")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
