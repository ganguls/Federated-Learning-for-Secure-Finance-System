#!/usr/bin/env python3
"""
Configuration file for local FL system execution
Modify these settings to customize the system behavior
"""

# Port Configuration
PORTS = {
    'ca': 9000,
    'server': 8080,
    'dashboard': 5000,
    'clients': list(range(8082, 8092))  # 10 client demo ports
}

# System Configuration
SYSTEM_CONFIG = {
    'min_clients': 10,
    'num_rounds': 10,
    'enable_certificates': True,
    'enable_attack_detection': True,
    'enable_monitoring': True,
    'log_level': 'INFO'
}

# Training Configuration
TRAINING_CONFIG = {
    'max_iterations': 1000,
    'solver': 'lbfgs',
    'random_state': 42,
    'warm_start': True
}

# Security Configuration
SECURITY_CONFIG = {
    'attack_probability': 0.1,
    'anomaly_threshold': 0.1,
    'clustering_threshold': 0.3,
    'statistical_threshold': 2.0,
    'history_window': 10
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'check_interval': 10,  # seconds
    'cpu_threshold': 90,   # percentage
    'memory_threshold': 90, # percentage
    'log_retention_days': 7
}

# Data Configuration
DATA_CONFIG = {
    'data_dir': 'Datapre/FL_clients',
    'min_samples_per_client': 1000,
    'train_test_split': 0.8,
    'feature_scaling': True
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'auto_open_browser': True,
    'refresh_interval': 5,  # seconds
    'max_log_lines': 1000,
    'theme': 'light'  # 'light' or 'dark'
}

# Development Configuration
DEV_CONFIG = {
    'debug_mode': False,
    'verbose_logging': False,
    'save_models': True,
    'export_metrics': True
}

# Environment Variables (will be set automatically)
ENV_VARS = {
    'PYTHONPATH': '.',
    'FLASK_ENV': 'production',
    'FLASK_DEBUG': 'False'
}

def get_config():
    """Get the complete configuration dictionary"""
    return {
        'ports': PORTS,
        'system': SYSTEM_CONFIG,
        'training': TRAINING_CONFIG,
        'security': SECURITY_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'data': DATA_CONFIG,
        'dashboard': DASHBOARD_CONFIG,
        'development': DEV_CONFIG,
        'environment': ENV_VARS
    }

def print_config():
    """Print the current configuration"""
    config = get_config()
    print("Current FL System Configuration:")
    print("=" * 40)
    
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config()

