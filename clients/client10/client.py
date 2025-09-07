import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

class LoanClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.is_malicious = False  # Demo: malicious client flag
        self.load_data()
        self.initialize_model()
    
    def load_data(self):
        """Load client-specific data"""
        # Try Docker path first, then local path
        docker_path = f"/app/data/FL_clients/client_{self.client_id}.csv"
        local_path = f"../../Datapre/FL_clients/client_{self.client_id}.csv"
        
        data_path = docker_path if os.path.exists(docker_path) else local_path
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"Client {self.client_id}: Loaded {len(df)} samples from {data_path}")
            
            # Split features and target
            X = df.drop(columns=['loan_status_binary'])
            y = df['loan_status_binary']
            
            # Train-test split (80-20)
            split_idx = int(0.8 * len(X))
            self.X_train = X[:split_idx]
            self.y_train = y[:split_idx]
            self.X_test = X[split_idx:]
            self.y_test = y[split_idx:]
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            print(f"Client {self.client_id}: Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        else:
            print(f"Error: Data file not found at either {docker_path} or {local_path}")
            raise FileNotFoundError(f"Data file not found at {data_path}")
    
    def initialize_model(self):
        """Initialize the local model"""
        n_features = self.X_train.shape[1]
        self.model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
            warm_start=True
        )
        # Initialize with zeros
        self.model.coef_ = np.zeros((1, n_features))
        self.model.intercept_ = np.zeros(1)
        self.model.classes_ = np.array([0, 1])
    
    def get_parameters(self, config):
        """Get model parameters"""
        return [self.model.coef_, self.model.intercept_]
    
    def set_parameters(self, parameters):
        """Set model parameters"""
        coef, intercept = parameters
        self.model.coef_ = coef
        self.model.intercept_ = intercept
    
    
    def apply_label_flipping_attack(self, y_data):
        """Apply label flipping attack for malicious client simulation"""
        if not self.is_malicious:
            return y_data
        
        # Create a copy to avoid modifying original data
        y_attacked = y_data.copy()
        
        # Label flipping attack: flip all labels
        # 0 (bad loan) -> 1 (good loan)
        # 1 (good loan) -> 0 (bad loan)
        y_attacked = 1 - y_attacked
        
        print(f"Client {self.client_id}: Applying label flipping attack - {np.sum(y_data != y_attacked)} labels flipped")
        return y_attacked
    
    def toggle_malicious_status(self):
        """Toggle malicious status for demo purposes"""
        self.is_malicious = not self.is_malicious
        status = "MALICIOUS" if self.is_malicious else "NORMAL"
        print(f"Client {self.client_id}: Status changed to {status}")
        return self.is_malicious

    def fit(self, parameters, config):
        """Train the model on local data"""
        self.set_parameters(parameters)
        
        # Apply label flipping attack if client is malicious
        y_train_modified = self.apply_label_flipping_attack(self.y_train)
        
        # Train the model (with potentially flipped labels)
        self.model.fit(self.X_train, y_train_modified)
        
        # Return updated parameters and training info
        return self.get_parameters(config), len(self.X_train), {
            "client_id": self.client_id,
            "is_malicious": self.is_malicious,
            "labels_flipped": np.sum(self.y_train != y_train_modified) if self.is_malicious else 0
        }
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local test data"""
        self.set_parameters(parameters)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # Return loss (1 - accuracy), num_examples, metrics dict
        loss = 1.0 - float(accuracy)
        return (
            loss,
            len(self.X_test),
            {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "client_id": self.client_id
            }
        )

def main():
    """Start the client"""
    client_id = 10  # This client's ID
    
    print(f"Starting Client {client_id}")
    
    # Create and start client
    client = LoanClient(client_id)
    
    
    # Start a simple HTTP server for demo controls in a separate thread
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class DemoHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/toggle_malicious':
                client.toggle_malicious_status()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'is_malicious': client.is_malicious}).encode())
            elif self.path == '/reset_malicious':
                client.is_malicious = False
                print(f"Client {client.client_id}: Reset to NORMAL status")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'is_malicious': client.is_malicious}).encode())
        
        def do_GET(self):
            if self.path == '/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'client_id': client.client_id,
                    'is_malicious': client.is_malicious
                }).encode())
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
        
        def log_message(self, format, *args):
            pass  # Suppress HTTP server logs
    
    # Start demo control server
    demo_port = 8081 + client_id
    try:
        demo_server = HTTPServer(('0.0.0.0', demo_port), DemoHandler)
        demo_thread = threading.Thread(target=demo_server.serve_forever, daemon=True)
        demo_thread.start()
        print(f"Demo control server started on port {demo_port}")
    except Exception as e:
        print(f"Warning: Could not start demo control server: {e}")

    # Start Flower client
    # Try Docker service name first, then localhost
    server_addresses = ["server:8080", "localhost:8080"]
    
    for server_address in server_addresses:
        try:
            print(f"Attempting to connect to {server_address}...")
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client
            )
            break
        except Exception as e:
            print(f"Failed to connect to {server_address}: {e}")
            if server_address == server_addresses[-1]:
                print("Failed to connect to all server addresses")
                raise

if __name__ == "__main__":
    main()
