#!/usr/bin/env python3
"""
Central Authority (CA) System for Federated Learning
Handles certificate generation, validation, and client authentication
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import logging
from typing import Dict, List, Optional, Tuple
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CentralAuthority:
    def __init__(self, ca_dir: str = "ca"):
        self.ca_dir = Path(ca_dir)
        self.ca_dir.mkdir(exist_ok=True)
        
        # CA private key and certificate paths
        self.ca_key_path = self.ca_dir / "ca_private_key.pem"
        self.ca_cert_path = self.ca_dir / "ca_certificate.pem"
        self.ca_serial_path = self.ca_dir / "ca.srl"
        
        # Database for certificate management
        self.db_path = self.ca_dir / "certificates.db"
        
        # Initialize CA
        self._initialize_ca()
        self._initialize_database()
    
    def _initialize_ca(self):
        """Initialize CA if it doesn't exist"""
        if not self.ca_key_path.exists() or not self.ca_cert_path.exists():
            logger.info("Initializing new CA...")
            self._generate_ca_key_pair()
            self._generate_ca_certificate()
            self._initialize_serial_file()
        else:
            logger.info("CA already exists, loading...")
    
    def _generate_ca_key_pair(self):
        """Generate CA private key"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        with open(self.ca_key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        logger.info(f"CA private key generated: {self.ca_key_path}")
    
    def _generate_ca_certificate(self):
        """Generate CA self-signed certificate"""
        with open(self.ca_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        
        # Create certificate subject
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FL Enterprise CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "fl-enterprise-ca.local"),
        ])
        
        # Create certificate
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=3650)  # 10 years
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        ).sign(private_key, hashes.SHA256())
        
        with open(self.ca_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        logger.info(f"CA certificate generated: {self.ca_cert_path}")
    
    def _initialize_serial_file(self):
        """Initialize serial number file"""
        with open(self.ca_serial_path, "w") as f:
            f.write("01\n")
    
    def _initialize_database(self):
        """Initialize SQLite database for certificate management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS certificates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT UNIQUE NOT NULL,
                certificate_path TEXT NOT NULL,
                private_key_path TEXT NOT NULL,
                issued_date TEXT NOT NULL,
                expiry_date TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                fingerprint TEXT NOT NULL,
                permissions TEXT DEFAULT 'standard'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS revoked_certificates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                certificate_path TEXT NOT NULL,
                revoked_date TEXT NOT NULL,
                reason TEXT DEFAULT 'unspecified'
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Certificate database initialized")
    
    def generate_client_certificate(self, client_id: str, permissions: str = "standard") -> Tuple[str, str]:
        """Generate client certificate and private key"""
        try:
            # Generate client private key
            client_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Create client certificate subject
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FL Enterprise Client"),
                x509.NameAttribute(NameOID.COMMON_NAME, f"client-{client_id}.fl-enterprise.local"),
            ])
            
            # Load CA private key and certificate
            with open(self.ca_key_path, "rb") as f:
                ca_private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            with open(self.ca_cert_path, "rb") as f:
                ca_cert = x509.load_pem_x509_certificate(f.read())
            
            # Generate serial number
            with open(self.ca_serial_path, "r") as f:
                serial = int(f.read().strip(), 16)
            
            # Create client certificate
            client_cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                ca_cert.subject
            ).public_key(
                client_private_key.public_key()
            ).serial_number(
                serial
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)  # 1 year
            ).add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True,
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(f"client-{client_id}.fl-enterprise.local"),
                    x509.IPAddress("127.0.0.1")
                ]),
                critical=False,
            ).sign(ca_private_key, hashes.SHA256())
            
            # Save client certificate and private key
            client_cert_path = self.ca_dir / f"client_{client_id}_cert.pem"
            client_key_path = self.ca_dir / f"client_{client_id}_key.pem"
            
            with open(client_cert_path, "wb") as f:
                f.write(client_cert.public_bytes(serialization.Encoding.PEM))
            
            with open(client_key_path, "wb") as f:
                f.write(client_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Update serial number
            with open(self.ca_serial_path, "w") as f:
                f.write(f"{serial + 1:02x}\n")
            
            # Calculate certificate fingerprint
            fingerprint = hashlib.sha256(
                client_cert.public_bytes(serialization.Encoding.DER)
            ).hexdigest()
            
            # Store in database
            self._store_certificate_in_db(
                client_id, str(client_cert_path), str(client_key_path),
                client_cert.not_valid_before, client_cert.not_valid_after,
                fingerprint, permissions
            )
            
            logger.info(f"Client certificate generated for {client_id}")
            return str(client_cert_path), str(client_key_path)
            
        except Exception as e:
            logger.error(f"Error generating client certificate: {e}")
            raise
    
    def _store_certificate_in_db(self, client_id: str, cert_path: str, key_path: str,
                                issued_date: datetime, expiry_date: datetime,
                                fingerprint: str, permissions: str):
        """Store certificate information in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO certificates 
            (client_id, certificate_path, private_key_path, issued_date, expiry_date, fingerprint, permissions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            client_id, cert_path, key_path,
            issued_date.isoformat(), expiry_date.isoformat(),
            fingerprint, permissions
        ))
        
        conn.commit()
        conn.close()
    
    def revoke_certificate(self, client_id: str, reason: str = "unspecified") -> bool:
        """Revoke a client certificate"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get certificate info
            cursor.execute('''
                SELECT certificate_path FROM certificates 
                WHERE client_id = ? AND status = 'active'
            ''', (client_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Certificate for client {client_id} not found or already revoked")
                return False
            
            cert_path = result[0]
            
            # Update status to revoked
            cursor.execute('''
                UPDATE certificates SET status = 'revoked' WHERE client_id = ?
            ''', (client_id,))
            
            # Add to revoked certificates table
            cursor.execute('''
                INSERT INTO revoked_certificates (client_id, certificate_path, revoked_date, reason)
                VALUES (?, ?, ?, ?)
            ''', (client_id, cert_path, datetime.utcnow().isoformat(), reason))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Certificate for client {client_id} revoked: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking certificate: {e}")
            return False
    
    def validate_certificate(self, client_id: str) -> Dict:
        """Validate a client certificate"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM certificates WHERE client_id = ? AND status = 'active'
            ''', (client_id,))
            
            result = cursor.fetchone()
            if not result:
                return {"valid": False, "reason": "Certificate not found or revoked"}
            
            # Check if certificate file exists
            cert_path = result[2]
            if not Path(cert_path).exists():
                return {"valid": False, "reason": "Certificate file not found"}
            
            # Load and validate certificate
            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())
            
            # Check expiry
            if datetime.utcnow() > cert.not_valid_after:
                return {"valid": False, "reason": "Certificate expired"}
            
            # Verify with CA
            with open(self.ca_cert_path, "rb") as f:
                ca_cert = x509.load_pem_x509_certificate(f.read())
            
            try:
                ca_cert.public_key().verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    cert.signature_hash_algorithm
                )
                signature_valid = True
            except Exception:
                signature_valid = False
            
            conn.close()
            
            return {
                "valid": signature_valid and datetime.utcnow() <= cert.not_valid_after,
                "client_id": client_id,
                "issued_date": cert.not_valid_before.isoformat(),
                "expiry_date": cert.not_valid_after.isoformat(),
                "permissions": result[8],
                "signature_valid": signature_valid
            }
            
        except Exception as e:
            logger.error(f"Error validating certificate: {e}")
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    def list_certificates(self) -> List[Dict]:
        """List all certificates"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT client_id, issued_date, expiry_date, status, permissions, fingerprint
                FROM certificates ORDER BY issued_date DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            certificates = []
            for row in results:
                certificates.append({
                    "client_id": row[0],
                    "issued_date": row[1],
                    "expiry_date": row[2],
                    "status": row[3],
                    "permissions": row[4],
                    "fingerprint": row[5]
                })
            
            return certificates
            
        except Exception as e:
            logger.error(f"Error listing certificates: {e}")
            return []
    
    def get_ca_certificate(self) -> str:
        """Get CA certificate as PEM string"""
        with open(self.ca_cert_path, "r") as f:
            return f.read()
    
    def get_ca_status(self) -> Dict:
        """Get CA status information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count active certificates
            cursor.execute("SELECT COUNT(*) FROM certificates WHERE status = 'active'")
            active_certs = cursor.fetchone()[0]
            
            # Count revoked certificates
            cursor.execute("SELECT COUNT(*) FROM certificates WHERE status = 'revoked'")
            revoked_certs = cursor.fetchone()[0]
            
            # Count expired certificates
            cursor.execute("SELECT COUNT(*) FROM certificates WHERE status = 'expired'")
            expired_certs = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "status": "healthy",
                "ca_initialized": self.ca_cert_path.exists() and self.ca_key_path.exists(),
                "active_certificates": active_certs,
                "revoked_certificates": revoked_certs,
                "expired_certificates": expired_certs,
                "total_certificates": active_certs + revoked_certs + expired_certs,
                "ca_certificate_path": str(self.ca_cert_path),
                "ca_private_key_path": str(self.ca_key_path),
                "database_path": str(self.db_path)
            }
        except Exception as e:
            logger.error(f"Error getting CA status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def cleanup_expired_certificates(self) -> int:
        """Remove expired certificates and return count of cleaned up"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find expired certificates
            cursor.execute('''
                SELECT client_id, certificate_path, private_key_path 
                FROM certificates 
                WHERE expiry_date < ? AND status = 'active'
            ''', (datetime.utcnow().isoformat(),))
            
            expired = cursor.fetchall()
            cleaned_count = 0
            
            for client_id, cert_path, key_path in expired:
                try:
                    # Remove files
                    if Path(cert_path).exists():
                        Path(cert_path).unlink()
                    if Path(key_path).exists():
                        Path(key_path).unlink()
                    
                    # Update database
                    cursor.execute('''
                        UPDATE certificates SET status = 'expired' WHERE client_id = ?
                    ''', (client_id,))
                    
                    cleaned_count += 1
                    logger.info(f"Cleaned up expired certificate for client {client_id}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up certificate for client {client_id}: {e}")
            
            conn.commit()
            conn.close()
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired certificates: {e}")
            return 0

def main():
    """Main function for CA operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FL Enterprise Central Authority")
    parser.add_argument("--action", choices=["init", "generate", "revoke", "validate", "list", "cleanup"],
                       default="init", help="Action to perform")
    parser.add_argument("--client-id", help="Client ID for certificate operations")
    parser.add_argument("--reason", help="Reason for revocation")
    parser.add_argument("--permissions", default="standard", help="Client permissions")
    
    args = parser.parse_args()
    
    ca = CentralAuthority()
    
    if args.action == "init":
        print("CA initialized successfully")
    
    elif args.action == "generate":
        if not args.client_id:
            print("Error: --client-id required for generate action")
            return
        cert_path, key_path = ca.generate_client_certificate(args.client_id, args.permissions)
        print(f"Certificate generated: {cert_path}")
        print(f"Private key generated: {key_path}")
    
    elif args.action == "revoke":
        if not args.client_id:
            print("Error: --client-id required for revoke action")
            return
        if ca.revoke_certificate(args.client_id, args.reason):
            print(f"Certificate for client {args.client_id} revoked successfully")
        else:
            print(f"Failed to revoke certificate for client {args.client_id}")
    
    elif args.action == "validate":
        if not args.client_id:
            print("Error: --client-id required for validate action")
            return
        result = ca.validate_certificate(args.client_id)
        print(json.dumps(result, indent=2))
    
    elif args.action == "list":
        certificates = ca.list_certificates()
        print(json.dumps(certificates, indent=2))
    
    elif args.action == "cleanup":
        cleaned = ca.cleanup_expired_certificates()
        print(f"Cleaned up {cleaned} expired certificates")

if __name__ == "__main__":
    main()
