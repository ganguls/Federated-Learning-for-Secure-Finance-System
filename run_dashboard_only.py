#!/usr/bin/env python3
"""
Run only the dashboard for testing demo data functionality
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    """Run only the dashboard"""
    print("Starting Dashboard Only Mode...")
    print("=" * 50)
    
    # Change to dashboard directory
    dashboard_dir = Path("dashboard")
    if not dashboard_dir.exists():
        print("❌ Dashboard directory not found!")
        return False
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'DASHBOARD_PORT': '5000',
            'CA_URL': 'http://localhost:9000',
            'SERVER_URL': 'http://localhost:8080',
            'FLASK_ENV': 'production',
            'PYTHONPATH': str(Path(__file__).parent)
        })
        
        print("Starting dashboard...")
        process = subprocess.Popen([
            sys.executable, "app_local.py"
        ], cwd=dashboard_dir, env=env)
        
        print("✅ Dashboard started!")
        print("🌐 Dashboard URL: http://localhost:5000")
        print("📊 Demo data should now work properly")
        print("\nPress Ctrl+C to stop the dashboard")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        process.terminate()
        process.wait(timeout=5)
        print("✅ Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()

