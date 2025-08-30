#!/usr/bin/env python3
"""
FL Enterprise Dashboard Startup Script
This script starts the dashboard with proper configuration
"""

import os
import sys
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dashboard.log')
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'flask_socketio', 'psutil', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main startup function"""
    print("=" * 60)
    print("FL Enterprise Dashboard Startup")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        logger.error("Dependencies check failed")
        return False
    
    print("All checks passed. Starting dashboard...")
    
    try:
        # Import and run the dashboard
        from app import app, socketio
        
        print("Dashboard started successfully!")
        print("Access the dashboard at: http://localhost:5000")
        print("Press Ctrl+C to stop the dashboard")
        
        # Run the dashboard
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
    except ImportError as e:
        logger.error(f"Failed to import dashboard: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
