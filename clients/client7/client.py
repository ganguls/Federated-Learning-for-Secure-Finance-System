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
        self.load_data()
        self.initialize_model()
    
    def load_data(self):
        """Load client-specific data"""
        data_path = f"../../Datapre/FL_clients/client_{self.client_id}.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"Client {self.client_id}: Loaded {len(df)} samples")
            
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
            print(f"Error: Data file not found at {data_path}")
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
    
    def fit(self, parameters, config):
        """Train the model on local data"""
        self.set_parameters(parameters)
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Return updated parameters and training info
        return self.get_parameters(config), len(self.X_train), {}
    
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
        
        return (
            float(accuracy),
            len(self.X_test),
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "client_id": self.client_id
            }
        )

def main():
    """Start the client"""
    client_id = 7  # This client's ID
    
    print(f"Starting Client {client_id}")
    
    # Create and start client
    client = LoanClient(client_id)
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        config=fl.server.ServerConfig(num_rounds=5)
    )

if __name__ == "__main__":
    main()
