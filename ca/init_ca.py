#!/usr/bin/env python3
"""
CA Initialization Script
Automatically generates certificates for all expected clients
"""

import os
import sys
import time
from ca import CentralAuthority
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_ca_with_clients(num_clients=10):
    """Initialize CA and generate certificates for all clients"""
    logger.info("Initializing Central Authority...")
    
    try:
        # Initialize CA
        ca = CentralAuthority()
        logger.info("CA initialized successfully")
        
        # Generate certificates for all expected clients
        logger.info(f"Generating certificates for {num_clients} clients...")
        
        for client_id in range(1, num_clients + 1):
            try:
                cert_path, key_path = ca.generate_client_certificate(
                    str(client_id), 
                    permissions="standard"
                )
                logger.info(f"✓ Generated certificate for client {client_id}")
                logger.info(f"  Certificate: {cert_path}")
                logger.info(f"  Private key: {key_path}")
            except Exception as e:
                logger.error(f"✗ Failed to generate certificate for client {client_id}: {e}")
        
        # Display CA status
        status = ca.get_ca_status()
        logger.info("CA Status:")
        logger.info(f"  - Active certificates: {status['active_certificates']}")
        logger.info(f"  - Total certificates: {status['total_certificates']}")
        logger.info(f"  - CA certificate: {status['ca_certificate_path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize CA: {e}")
        return False

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("FL Enterprise - CA Initialization")
    logger.info("=" * 60)
    
    # Get number of clients from environment
    num_clients = int(os.getenv("NUM_CLIENTS", "10"))
    
    if init_ca_with_clients(num_clients):
        logger.info("CA initialization completed successfully!")
        return 0
    else:
        logger.error("CA initialization failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
