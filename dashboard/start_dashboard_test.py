#!/usr/bin/env python3
"""
Start dashboard and run tests to verify chart functionality
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def start_dashboard():
    """Start the dashboard in the background"""
    print("🚀 Starting Dashboard...")
    
    try:
        # Change to dashboard directory
        dashboard_dir = Path(__file__).parent
        os.chdir(dashboard_dir)
        
        # Start dashboard
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("Dashboard started with PID:", process.pid)
        return process
        
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return None

def wait_for_dashboard(max_wait=30):
    """Wait for dashboard to be ready"""
    print("⏳ Waiting for dashboard to be ready...")
    
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            if response.status_code == 200:
                print("✅ Dashboard is ready!")
                return True
        except:
            pass
        
        print(f"   Waiting... ({i+1}/{max_wait})")
        time.sleep(1)
    
    print("❌ Dashboard failed to start within timeout")
    return False

def test_charts():
    """Test if charts are working"""
    print("\n📊 Testing Charts...")
    
    try:
        # Test training metrics endpoint
        response = requests.get("http://localhost:5000/api/metrics/training", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training metrics loaded: {len(data)} rounds")
            
            if data and isinstance(data, list):
                first_round = data[0]
                if 'client_metrics' in first_round:
                    client_count = len(first_round['client_metrics'])
                    print(f"✅ Client metrics found: {client_count} clients")
                else:
                    print("⚠️  No client metrics in training data")
            else:
                print("⚠️  Training data format issue")
        else:
            print(f"❌ Training metrics failed: HTTP {response.status_code}")
        
        # Test main dashboard page
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "trainingChart" in content and "Chart.js" in content:
                print("✅ Dashboard page loads with chart elements")
            else:
                print("⚠️  Dashboard page missing chart elements")
        else:
            print(f"❌ Dashboard page failed: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Chart test error: {e}")

def main():
    """Main function"""
    print("🧪 Dashboard Chart Fix Test")
    print("=" * 50)
    
    # Start dashboard
    process = start_dashboard()
    if not process:
        print("❌ Failed to start dashboard")
        return 1
    
    try:
        # Wait for dashboard to be ready
        if not wait_for_dashboard():
            return 1
        
        # Test charts
        test_charts()
        
        print("\n🎉 Dashboard test completed!")
        print("You can now open http://localhost:5000 in your browser")
        print("Press Ctrl+C to stop the dashboard")
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            process.terminate()
            process.wait()
            print("✅ Dashboard stopped")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        if process:
            process.terminate()
        return 1

if __name__ == "__main__":
    import os
    sys.exit(main())
