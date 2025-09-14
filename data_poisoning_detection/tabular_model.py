"""
Neural Network Model for Tabular Data
Extracted from the original project for credit risk assessment and other tabular tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TabularNet(nn.Module):
    """
    Neural network for tabular data classification
    Adaptable to any tabular dataset with configurable architecture
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=2, dropout_rate=0.3):
        """
        Initialize the tabular network
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(TabularNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build the network dynamically based on input dimension
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            log_probs: Log probabilities of shape (batch_size, num_classes)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward pass through the network
        logits = self.network(x)
        
        # Return log probabilities for NLLLoss
        return F.log_softmax(logits, dim=1)
    
    def get_embedding(self, x):
        """
        Get the embedding (features before the final classification layer)
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            embedding: Feature embedding of shape (batch_size, last_hidden_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward pass through all layers except the last one
        for i, layer in enumerate(self.network):
            if i < len(self.network) - 1:  # All layers except the last one
                x = layer(x)
            else:
                break
        
        return x
    
    def get_parameters_count(self):
        """Get the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())

class CreditRiskNet(TabularNet):
    """
    Specialized neural network for credit risk assessment
    Inherits from TabularNet with credit-specific configurations
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=2, dropout_rate=0.3):
        """
        Initialize the credit risk network
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions (default for credit risk)
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate for regularization
        """
        super(CreditRiskNet, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

def create_tabular_model(input_dim, num_classes=2, model_type='default', **kwargs):
    """
    Factory function to create tabular models
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        model_type: Type of model ('default', 'credit_risk', 'custom')
        **kwargs: Additional arguments for model configuration
    
    Returns:
        model: Initialized neural network model
    """
    if model_type == 'credit_risk':
        hidden_dims = kwargs.get('hidden_dims', [256, 128, 64])
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        return CreditRiskNet(input_dim, hidden_dims, num_classes, dropout_rate)
    
    elif model_type == 'default':
        hidden_dims = kwargs.get('hidden_dims', [512, 256, 128])
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        return TabularNet(input_dim, hidden_dims, num_classes, dropout_rate)
    
    elif model_type == 'custom':
        hidden_dims = kwargs.get('hidden_dims', [256, 128])
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        return TabularNet(input_dim, hidden_dims, num_classes, dropout_rate)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model_from_checkpoint(checkpoint_path, input_dim, num_classes=2, model_type='credit_risk'):
    """
    Load a model from a checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        input_dim: Number of input features
        num_classes: Number of output classes
        model_type: Type of model to create
    
    Returns:
        model: Loaded model with weights
    """
    # Create model
    model = create_tabular_model(input_dim, num_classes, model_type)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'].state_dict())
    else:
        model.load_state_dict(checkpoint)
    
    return model

def save_model_checkpoint(model, save_path, additional_info=None):
    """
    Save a model checkpoint
    
    Args:
        model: Model to save
        save_path: Path to save the checkpoint
        additional_info: Additional information to save (dict)
    """
    checkpoint = {
        'model': model,
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'num_classes': model.num_classes
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
