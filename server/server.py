import flwr as fl
import ssl
from typing import Tuple

# =========================
# Flower Server Configuration
# =========================

# Number of FL rounds
NUM_ROUNDS = 5

# TLS / mTLS placeholders
# Replace these file paths with actual certificate files when ready
SERVER_CERT = "ca/server.crt"
SERVER_KEY = "ca/server.key"
ROOT_CA_CERT = "ca/ca.crt"

def main():
    """
    Start a Flower federated learning server.
    TLS / mTLS can be configured by providing SSL credentials.
    """
    # Create server SSL credentials (placeholder)
    # Note: Uncomment and configure for actual TLS/mTLS
    """
    with open(SERVER_CERT, "rb") as f:
        server_cert = f.read()
    with open(SERVER_KEY, "rb") as f:
        server_key = f.read()
    with open(ROOT_CA_CERT, "rb") as f:
        root_cert = f.read()

    server_credentials = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_credentials.load_cert_chain(certfile=SERVER_CERT, keyfile=SERVER_KEY)
    server_credentials.load_verify_locations(ROOT_CA_CERT)
    server_credentials.verify_mode = ssl.CERT_REQUIRED
    """

    # Start the Flower server
    print("Starting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": NUM_ROUNDS},
        # grpc_ssl_server_credentials=server_credentials  # Uncomment when TLS ready
    )

if __name__ == "__main__":
    main()
