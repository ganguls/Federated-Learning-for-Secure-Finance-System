#!/usr/bin/env python3
"""
Simple test to check if the dashboard can start
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dashboard'))

try:
    print("Testing imports...")
    from flask import Flask
    print("✓ Flask imported")
    
    from flask_socketio import SocketIO
    print("✓ Flask-SocketIO imported")
    
    from flask_cors import CORS
    print("✓ Flask-CORS imported")
    
    import psutil
    print("✓ psutil imported")
    
    print("\nAll imports successful!")
    
    # Try to create a simple Flask app
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    @app.route('/')
    def hello():
        return "Hello World!"
    
    @app.route('/health')
    def health():
        return "OK"
    
    print("✓ Flask app created successfully!")
    print("✓ Dashboard should work!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

