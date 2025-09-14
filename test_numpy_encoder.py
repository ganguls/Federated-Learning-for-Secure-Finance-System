#!/usr/bin/env python3
"""
Test the NumpyEncoder directly to verify it works
"""

import json
import numpy as np
from dashboard.app import NumpyEncoder, convert_numpy_types

def test_numpy_encoder():
    """Test the NumpyEncoder class directly"""
    print("Testing NumpyEncoder directly...")
    
    # Create test data with NumPy types
    test_data = {
        'int_value': np.int64(42),
        'float_value': np.float64(3.14),
        'array_value': np.array([1, 2, 3]),
        'nested': {
            'inner_int': np.int32(100),
            'inner_float': np.float32(2.71)
        },
        'list_with_numpy': [np.int64(1), np.float64(2.5), 'string']
    }
    
    print("Original data types:")
    print(f"  int_value: {type(test_data['int_value'])}")
    print(f"  float_value: {type(test_data['float_value'])}")
    print(f"  array_value: {type(test_data['array_value'])}")
    print(f"  nested.inner_int: {type(test_data['nested']['inner_int'])}")
    print(f"  nested.inner_float: {type(test_data['nested']['inner_float'])}")
    
    # Test NumpyEncoder
    try:
        json_str = json.dumps(test_data, cls=NumpyEncoder)
        print("✅ NumpyEncoder works!")
        print(f"JSON: {json_str[:100]}...")
        
        # Parse back to verify
        parsed_data = json.loads(json_str)
        print("✅ JSON parsing works!")
        print(f"Parsed int_value: {parsed_data['int_value']} (type: {type(parsed_data['int_value'])})")
        
    except Exception as e:
        print(f"❌ NumpyEncoder failed: {e}")
    
    # Test convert_numpy_types function
    try:
        converted_data = convert_numpy_types(test_data)
        print("✅ convert_numpy_types works!")
        print(f"Converted int_value: {converted_data['int_value']} (type: {type(converted_data['int_value'])})")
        
        # Test JSON serialization of converted data
        json_str2 = json.dumps(converted_data)
        print("✅ JSON serialization of converted data works!")
        
    except Exception as e:
        print(f"❌ convert_numpy_types failed: {e}")

def test_flask_jsonify():
    """Test Flask jsonify with NumPy types"""
    print("\nTesting Flask jsonify...")
    
    try:
        from flask import Flask, jsonify
        
        app = Flask(__name__)
        app.json_encoder = NumpyEncoder
        
        with app.app_context():
            test_data = {
                'int_value': np.int64(42),
                'float_value': np.float64(3.14),
                'array_value': np.array([1, 2, 3])
            }
            
            response = jsonify(test_data)
            print("✅ Flask jsonify with NumpyEncoder works!")
            print(f"Response data: {response.get_data(as_text=True)}")
            
    except Exception as e:
        print(f"❌ Flask jsonify failed: {e}")

if __name__ == "__main__":
    test_numpy_encoder()
    test_flask_jsonify()

