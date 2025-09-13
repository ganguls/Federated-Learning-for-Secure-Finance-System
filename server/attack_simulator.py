"""
Data Poisoning Attack Simulator
Simulates various types of data poisoning attacks for testing detection mechanisms
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AttackType(Enum):
    """Types of data poisoning attacks"""
    LABEL_FLIPPING = "label_flipping"
    GRADIENT_POISONING = "gradient_poisoning"
    MODEL_POISONING = "model_poisoning"
    BACKDOOR = "backdoor"
    BYZANTINE = "byzantine"
    SIGN_FLIPPING = "sign_flipping"
    SCALING_ATTACK = "scaling_attack"

class DataPoisoningSimulator:
    """
    Simulates various data poisoning attacks for testing detection mechanisms
    """
    
    def __init__(self, attack_probability: float = 0.1):
        """
        Initialize the attack simulator
        
        Args:
            attack_probability: Probability of a client being malicious
        """
        self.attack_probability = attack_probability
        self.active_attacks = {}
        self.attack_history = []
        
        logger.info("Data poisoning simulator initialized")
    
    def simulate_attack(self, 
                       client_id: str, 
                       parameters: Tuple, 
                       metrics: Dict, 
                       attack_type: AttackType = None) -> Tuple[Tuple, Dict]:
        """
        Simulate a data poisoning attack on client data
        
        Args:
            client_id: ID of the client
            parameters: Model parameters (coef, intercept)
            metrics: Client metrics
            attack_type: Type of attack to simulate (random if None)
            
        Returns:
            Modified parameters and metrics
        """
        if client_id in self.active_attacks:
            attack_type = self.active_attacks[client_id]
        elif random.random() < self.attack_probability:
            attack_type = random.choice(list(AttackType))
            self.active_attacks[client_id] = attack_type
        else:
            return parameters, metrics
        
        # Apply attack based on type
        if attack_type == AttackType.LABEL_FLIPPING:
            return self._label_flipping_attack(parameters, metrics, client_id)
        elif attack_type == AttackType.GRADIENT_POISONING:
            return self._gradient_poisoning_attack(parameters, metrics, client_id)
        elif attack_type == AttackType.MODEL_POISONING:
            return self._model_poisoning_attack(parameters, metrics, client_id)
        elif attack_type == AttackType.BACKDOOR:
            return self._backdoor_attack(parameters, metrics, client_id)
        elif attack_type == AttackType.BYZANTINE:
            return self._byzantine_attack(parameters, metrics, client_id)
        elif attack_type == AttackType.SIGN_FLIPPING:
            return self._sign_flipping_attack(parameters, metrics, client_id)
        elif attack_type == AttackType.SCALING_ATTACK:
            return self._scaling_attack(parameters, metrics, client_id)
        else:
            return parameters, metrics
    
    def _label_flipping_attack(self, parameters: Tuple, metrics: Dict, client_id: str) -> Tuple[Tuple, Dict]:
        """Simulate label flipping attack"""
        coef, intercept = parameters
        
        # Flip labels by negating coefficients
        poisoned_coef = -coef
        poisoned_intercept = -intercept
        
        # Update metrics to reflect attack
        poisoned_metrics = metrics.copy()
        poisoned_metrics['is_malicious'] = True
        poisoned_metrics['attack_type'] = 'label_flipping'
        poisoned_metrics['labels_flipped'] = random.randint(100, 1000)
        
        # Degrade performance metrics
        poisoned_metrics['accuracy'] = max(0.1, metrics.get('accuracy', 0.8) - 0.3)
        poisoned_metrics['loss'] = min(2.0, metrics.get('loss', 0.5) + 0.5)
        
        logger.warning(f"Label flipping attack simulated for client {client_id}")
        return (poisoned_coef, poisoned_intercept), poisoned_metrics
    
    def _gradient_poisoning_attack(self, parameters: Tuple, metrics: Dict, client_id: str) -> Tuple[Tuple, Dict]:
        """Simulate gradient poisoning attack"""
        coef, intercept = parameters
        
        # Add noise to gradients
        noise_scale = 2.0
        coef_noise = np.random.normal(0, noise_scale, coef.shape)
        intercept_noise = np.random.normal(0, noise_scale, intercept.shape)
        
        poisoned_coef = coef + coef_noise
        poisoned_intercept = intercept + intercept_noise
        
        # Update metrics
        poisoned_metrics = metrics.copy()
        poisoned_metrics['is_malicious'] = True
        poisoned_metrics['attack_type'] = 'gradient_poisoning'
        poisoned_metrics['gradient_noise'] = float(np.mean(np.abs(coef_noise)))
        
        # Slightly degrade performance
        poisoned_metrics['accuracy'] = max(0.2, metrics.get('accuracy', 0.8) - 0.1)
        poisoned_metrics['loss'] = min(1.5, metrics.get('loss', 0.5) + 0.2)
        
        logger.warning(f"Gradient poisoning attack simulated for client {client_id}")
        return (poisoned_coef, poisoned_intercept), poisoned_metrics
    
    def _model_poisoning_attack(self, parameters: Tuple, metrics: Dict, client_id: str) -> Tuple[Tuple, Dict]:
        """Simulate model poisoning attack"""
        coef, intercept = parameters
        
        # Replace model with random parameters
        poisoned_coef = np.random.normal(0, 1.0, coef.shape)
        poisoned_intercept = np.random.normal(0, 1.0, intercept.shape)
        
        # Update metrics
        poisoned_metrics = metrics.copy()
        poisoned_metrics['is_malicious'] = True
        poisoned_metrics['attack_type'] = 'model_poisoning'
        poisoned_metrics['model_replaced'] = True
        
        # Severely degrade performance
        poisoned_metrics['accuracy'] = random.uniform(0.1, 0.3)
        poisoned_metrics['loss'] = random.uniform(1.0, 2.0)
        
        logger.warning(f"Model poisoning attack simulated for client {client_id}")
        return (poisoned_coef, poisoned_intercept), poisoned_metrics
    
    def _backdoor_attack(self, parameters: Tuple, metrics: Dict, client_id: str) -> Tuple[Tuple, Dict]:
        """Simulate backdoor attack"""
        coef, intercept = parameters
        
        # Add backdoor trigger by modifying specific features
        poisoned_coef = coef.copy()
        backdoor_features = random.sample(range(len(coef)), min(3, len(coef)))
        
        for feature_idx in backdoor_features:
            poisoned_coef[feature_idx] *= 10  # Amplify specific features
        
        # Update metrics
        poisoned_metrics = metrics.copy()
        poisoned_metrics['is_malicious'] = True
        poisoned_metrics['attack_type'] = 'backdoor'
        poisoned_metrics['backdoor_features'] = backdoor_features
        
        # Maintain normal performance to avoid detection
        poisoned_metrics['accuracy'] = metrics.get('accuracy', 0.8)
        poisoned_metrics['loss'] = metrics.get('loss', 0.5)
        
        logger.warning(f"Backdoor attack simulated for client {client_id}")
        return (poisoned_coef, intercept), poisoned_metrics
    
    def _byzantine_attack(self, parameters: Tuple, metrics: Dict, client_id: str) -> Tuple[Tuple, Dict]:
        """Simulate Byzantine attack (extreme parameter values)"""
        coef, intercept = parameters
        
        # Set extreme parameter values
        poisoned_coef = np.full_like(coef, 100.0)  # Extreme values
        poisoned_intercept = np.full_like(intercept, 100.0)
        
        # Update metrics
        poisoned_metrics = metrics.copy()
        poisoned_metrics['is_malicious'] = True
        poisoned_metrics['attack_type'] = 'byzantine'
        poisoned_metrics['extreme_values'] = True
        
        # Severely degrade performance
        poisoned_metrics['accuracy'] = random.uniform(0.05, 0.2)
        poisoned_metrics['loss'] = random.uniform(1.5, 3.0)
        
        logger.warning(f"Byzantine attack simulated for client {client_id}")
        return (poisoned_coef, poisoned_intercept), poisoned_metrics
    
    def _sign_flipping_attack(self, parameters: Tuple, metrics: Dict, client_id: str) -> Tuple[Tuple, Dict]:
        """Simulate sign flipping attack"""
        coef, intercept = parameters
        
        # Flip signs of parameters
        poisoned_coef = -coef
        poisoned_intercept = -intercept
        
        # Update metrics
        poisoned_metrics = metrics.copy()
        poisoned_metrics['is_malicious'] = True
        poisoned_metrics['attack_type'] = 'sign_flipping'
        poisoned_metrics['signs_flipped'] = True
        
        # Degrade performance
        poisoned_metrics['accuracy'] = max(0.1, metrics.get('accuracy', 0.8) - 0.4)
        poisoned_metrics['loss'] = min(2.0, metrics.get('loss', 0.5) + 0.8)
        
        logger.warning(f"Sign flipping attack simulated for client {client_id}")
        return (poisoned_coef, poisoned_intercept), poisoned_metrics
    
    def _scaling_attack(self, parameters: Tuple, metrics: Dict, client_id: str) -> Tuple[Tuple, Dict]:
        """Simulate scaling attack (amplify parameters)"""
        coef, intercept = parameters
        
        # Scale parameters by large factor
        scale_factor = random.uniform(5.0, 20.0)
        poisoned_coef = coef * scale_factor
        poisoned_intercept = intercept * scale_factor
        
        # Update metrics
        poisoned_metrics = metrics.copy()
        poisoned_metrics['is_malicious'] = True
        poisoned_metrics['attack_type'] = 'scaling_attack'
        poisoned_metrics['scale_factor'] = scale_factor
        
        # Moderate performance degradation
        poisoned_metrics['accuracy'] = max(0.2, metrics.get('accuracy', 0.8) - 0.2)
        poisoned_metrics['loss'] = min(1.2, metrics.get('loss', 0.5) + 0.3)
        
        logger.warning(f"Scaling attack simulated for client {client_id}")
        return (poisoned_coef, poisoned_intercept), poisoned_metrics
    
    def set_client_attack(self, client_id: str, attack_type: AttackType):
        """Set a specific client to use a specific attack type"""
        self.active_attacks[client_id] = attack_type
        logger.info(f"Set client {client_id} to use {attack_type.value} attack")
    
    def remove_client_attack(self, client_id: str):
        """Remove attack from a specific client"""
        if client_id in self.active_attacks:
            del self.active_attacks[client_id]
            logger.info(f"Removed attack from client {client_id}")
    
    def clear_all_attacks(self):
        """Clear all active attacks"""
        self.active_attacks.clear()
        logger.info("Cleared all active attacks")
    
    def get_attack_status(self) -> Dict:
        """Get current attack status"""
        return {
            'active_attacks': {client_id: attack_type.value for client_id, attack_type in self.active_attacks.items()},
            'attack_probability': self.attack_probability,
            'total_attacks': len(self.active_attacks)
        }
    
    def generate_attack_report(self) -> Dict:
        """Generate a report of all attacks"""
        report = {
            'timestamp': np.datetime64('now').astype(str),
            'active_attacks': len(self.active_attacks),
            'attack_types': {},
            'malicious_clients': list(self.active_attacks.keys())
        }
        
        # Count attack types
        for attack_type in self.active_attacks.values():
            attack_name = attack_type.value
            report['attack_types'][attack_name] = report['attack_types'].get(attack_name, 0) + 1
        
        return report

