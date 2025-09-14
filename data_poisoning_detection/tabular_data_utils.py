"""
Tabular Data Processing Utilities
Extracted from the original project for handling tabular datasets in federated learning
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class TabularDataset(Dataset):
    """Custom Dataset for tabular data"""
    
    def __init__(self, features, labels):
        """
        Initialize the dataset
        
        Args:
            features: Feature matrix (numpy array)
            labels: Label array (numpy array)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def preprocess_tabular_data(csv_path, target_column, feature_columns=None, 
                           test_size=0.2, random_state=42, handle_missing='median'):
    """
    Preprocess tabular data for federated learning
    
    Args:
        csv_path: Path to the CSV file
        target_column: Name of the target column
        feature_columns: List of feature columns to use (None for all except target)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        handle_missing: Strategy for handling missing values ('median', 'mean', 'mode')
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        feature_columns: List of feature column names
        scaler: Fitted StandardScaler
    """
    
    # Load the dataset
    print("Loading tabular dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Select feature columns
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Filter out columns that don't exist in the dataset
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(available_features)} features out of {len(feature_columns)} specified")
    
    # Handle missing values
    df_clean = df[available_features + [target_column]].copy()
    
    for col in available_features:
        if df_clean[col].dtype in ['int64', 'float64']:
            if handle_missing == 'median':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif handle_missing == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            # For categorical columns, fill with mode
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col] = df_clean[col].fillna(mode_value[0])
            else:
                df_clean[col] = df_clean[col].fillna(0)
    
    # Remove rows with missing target values
    df_clean = df_clean.dropna(subset=[target_column])
    
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    print(f"Target distribution: {df_clean[target_column].value_counts()}")
    
    # Separate features and target
    X = df_clean[available_features].values
    y = df_clean[target_column].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create PyTorch datasets
    train_dataset = TabularDataset(X_train_scaled, y_train)
    test_dataset = TabularDataset(X_test_scaled, y_test)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Feature dimension: {X_train_scaled.shape[1]}")
    
    return train_dataset, test_dataset, available_features, scaler

def split_data_among_clients(dataset, num_clients, mal_percentage=0.0, 
                           target_honest=1, target_mal=0, random_state=42):
    """
    Split tabular data among clients for federated learning
    
    Args:
        dataset: TabularDataset object
        num_clients: Total number of clients
        mal_percentage: Percentage of malicious clients (0.0 to 1.0)
        target_honest: Target class for honest clients
        target_mal: Target class for malicious clients
        random_state: Random seed for reproducibility
    
    Returns:
        user_groups: Dictionary mapping client IDs to their data indices
        attackers: Set of attacker client IDs
    """
    np.random.seed(random_state)
    
    num_items = len(dataset) // num_clients
    user_groups = {}
    
    # Create data splits
    for i in range(num_clients):
        start_idx = i * num_items
        end_idx = (i + 1) * num_items
        user_groups[i] = list(range(start_idx, end_idx))
    
    # Identify malicious clients
    num_attackers = int(mal_percentage * num_clients)
    attackers = set(range(num_attackers))  # First 'num_attackers' clients are attackers
    
    print(f"Created {num_clients} clients with {len(attackers)} malicious clients")
    print(f"Data per client: ~{num_items} samples")
    
    return user_groups, attackers

def poison_client_data(dataset, user_indices, target_honest, target_mal):
    """
    Poison client data by flipping labels for malicious clients
    
    Args:
        dataset: TabularDataset object
        user_indices: List of data indices belonging to the client
        target_honest: Original target class
        target_mal: Target class to flip to
    
    Returns:
        poisoned_data: Dictionary with poisoned data
    """
    poisoned_data = {}
    
    for idx in range(len(dataset)):
        features, label = dataset[idx]
        
        if idx in user_indices and label == target_honest:
            # Flip the label for malicious clients
            poisoned_data[idx] = (features, target_mal)
        else:
            # Keep original data
            poisoned_data[idx] = (features, label)
    
    return poisoned_data

def create_client_dataloader(dataset, client_indices, batch_size=32, shuffle=True):
    """
    Create a DataLoader for a specific client
    
    Args:
        dataset: TabularDataset object
        client_indices: List of data indices for this client
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    client_data = [dataset[i] for i in client_indices]
    client_dataset = TabularDataset(
        np.array([data[0].numpy() for data in client_data]),
        np.array([data[1].numpy() for data in client_data])
    )
    
    return DataLoader(client_dataset, batch_size=batch_size, shuffle=shuffle)

def get_data_statistics(dataset, feature_columns):
    """
    Get statistics about the dataset
    
    Args:
        dataset: TabularDataset object
        feature_columns: List of feature column names
    
    Returns:
        stats: Dictionary with dataset statistics
    """
    features = []
    labels = []
    
    for i in range(len(dataset)):
        feature, label = dataset[i]
        features.append(feature.numpy())
        labels.append(label.numpy())
    
    features = np.array(features)
    labels = np.array(labels)
    
    stats = {
        'num_samples': len(dataset),
        'num_features': len(feature_columns),
        'num_classes': len(np.unique(labels)),
        'class_distribution': np.bincount(labels),
        'feature_means': np.mean(features, axis=0),
        'feature_stds': np.std(features, axis=0),
        'feature_mins': np.min(features, axis=0),
        'feature_maxs': np.max(features, axis=0)
    }
    
    return stats

def validate_data_integrity(dataset, expected_features, expected_classes):
    """
    Validate the integrity of the dataset
    
    Args:
        dataset: TabularDataset object
        expected_features: Expected number of features
        expected_classes: Expected number of classes
    
    Returns:
        is_valid: Boolean indicating if data is valid
        issues: List of issues found
    """
    issues = []
    
    # Check feature dimension
    if len(dataset) > 0:
        sample_feature, _ = dataset[0]
        if sample_feature.shape[0] != expected_features:
            issues.append(f"Feature dimension mismatch: expected {expected_features}, got {sample_feature.shape[0]}")
    
    # Check class distribution
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    unique_classes = len(set(labels))
    if unique_classes != expected_classes:
        issues.append(f"Class count mismatch: expected {expected_classes}, got {unique_classes}")
    
    # Check for NaN values
    for i in range(min(100, len(dataset))):  # Check first 100 samples
        feature, _ = dataset[i]
        if torch.isnan(feature).any():
            issues.append(f"NaN values found in sample {i}")
            break
    
    is_valid = len(issues) == 0
    return is_valid, issues

def create_synthetic_tabular_data(num_samples=1000, num_features=20, num_classes=2, 
                                noise_level=0.1, random_state=42):
    """
    Create synthetic tabular data for testing
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        num_classes: Number of classes
        noise_level: Level of noise to add
        random_state: Random seed
    
    Returns:
        dataset: TabularDataset object
        feature_columns: List of feature names
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(num_samples, num_features)
    
    # Generate labels based on a simple rule
    y = np.zeros(num_samples)
    for i in range(num_samples):
        # Simple rule: if sum of first 5 features > 0, class 1, else class 0
        if np.sum(X[i, :5]) > 0:
            y[i] = 1
        else:
            y[i] = 0
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, (num_samples, num_features))
        X += noise
    
    # Create feature names
    feature_columns = [f'feature_{i}' for i in range(num_features)]
    
    # Create dataset
    dataset = TabularDataset(X, y.astype(int))
    
    return dataset, feature_columns
