#!/usr/bin/env python3
"""
Script to fix the __init__ method in all client files to add the is_malicious flag
"""

import os
from pathlib import Path

def fix_client_init(client_path):
    """Fix the __init__ method to add is_malicious flag"""
    if not client_path.exists():
        return False
    
    with open(client_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'self.is_malicious = False  # Demo: malicious client flag' in content:
        print(f"Client file {client_path} already has is_malicious flag")
        return True
    
    # Find and replace the __init__ method
    old_init = '''        self.X_test = None
        self.y_test = None
        self.load_data()
        self.initialize_model()'''
    
    new_init = '''        self.X_test = None
        self.y_test = None
        self.is_malicious = False  # Demo: malicious client flag
        self.load_data()
        self.initialize_model()'''
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        
        # Write the updated content back
        with open(client_path, 'w') as f:
            f.write(content)
        
        print(f"Fixed {client_path}")
        return True
    else:
        print(f"Could not find expected pattern in {client_path}")
        return False

def main():
    """Fix all client files"""
    clients_dir = Path("clients")
    fixed_count = 0
    
    for client_id in range(1, 11):
        client_path = clients_dir / f"client{client_id}" / "client.py"
        if fix_client_init(client_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} client files")

if __name__ == "__main__":
    main()
