#!/usr/bin/env python3
"""
Central Authority Service for Federated Learning
RESTful API service for certificate management
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from pathlib import Path
from ca import CentralAuthority
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize CA
ca = CentralAuthority()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'central-authority',
        'timestamp': time.time()
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get CA status"""
    try:
        status = ca.get_ca_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting CA status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/certificates', methods=['GET'])
def list_certificates():
    """List all certificates"""
    try:
        certificates = ca.list_certificates()
        return jsonify(certificates)
    except Exception as e:
        logger.error(f"Error listing certificates: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/certificates/generate', methods=['POST'])
def generate_certificate():
    """Generate a new client certificate"""
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        permissions = data.get('permissions', 'standard')
        
        if not client_id:
            return jsonify({'error': 'Client ID required'}), 400
        
        cert_path, key_path = ca.generate_client_certificate(client_id, permissions)
        
        return jsonify({
            'success': True,
            'client_id': client_id,
            'certificate_path': cert_path,
            'private_key_path': key_path,
            'message': f'Certificate generated for client {client_id}'
        })
    except Exception as e:
        logger.error(f"Error generating certificate: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/certificates/<client_id>/validate', methods=['GET'])
def validate_certificate(client_id):
    """Validate a client certificate"""
    try:
        result = ca.validate_certificate(client_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error validating certificate: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/certificates/<client_id>/revoke', methods=['POST'])
def revoke_certificate(client_id):
    """Revoke a client certificate"""
    try:
        data = request.get_json()
        reason = data.get('reason', 'unspecified')
        
        success = ca.revoke_certificate(client_id, reason)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Certificate for client {client_id} revoked successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to revoke certificate for client {client_id}'
            }), 400
    except Exception as e:
        logger.error(f"Error revoking certificate: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/certificates/cleanup', methods=['POST'])
def cleanup_certificates():
    """Clean up expired certificates"""
    try:
        cleaned_count = ca.cleanup_expired_certificates()
        return jsonify({
            'success': True,
            'cleaned_count': cleaned_count,
            'message': f'Cleaned up {cleaned_count} expired certificates'
        })
    except Exception as e:
        logger.error(f"Error cleaning up certificates: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ca/certificate', methods=['GET'])
def get_ca_certificate():
    """Get CA certificate"""
    try:
        ca_cert = ca.get_ca_certificate()
        return jsonify({
            'success': True,
            'certificate': ca_cert
        })
    except Exception as e:
        logger.error(f"Error getting CA certificate: {e}")
        return jsonify({'error': str(e)}), 500

def background_cleanup():
    """Background task to clean up expired certificates"""
    while True:
        try:
            time.sleep(3600)  # Run every hour
            cleaned = ca.cleanup_expired_certificates()
            if cleaned > 0:
                logger.info(f"Background cleanup: cleaned {cleaned} expired certificates")
        except Exception as e:
            logger.error(f"Error in background cleanup: {e}")

if __name__ == '__main__':
    # Start background cleanup task
    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Get port from environment or use default
    port = int(os.environ.get('CA_PORT', 9000))
    
    logger.info(f"Starting Central Authority Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
