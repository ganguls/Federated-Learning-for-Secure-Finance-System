#!/usr/bin/env python3
"""
CA Integration Validation Script
Tests all CA functionality and integration points
"""

import requests
import json
import time
import sys
import os

def test_ca_service():
    """Test CA service endpoints"""
    print("=" * 60)
    print("TESTING CA SERVICE INTEGRATION")
    print("=" * 60)
    
    ca_url = "http://localhost:9000"
    issues = []
    
    # Test 1: Health check
    print("1. Testing CA health check...")
    try:
        response = requests.get(f"{ca_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print("   ✓ CA service is healthy")
            else:
                print(f"   ⚠ CA service reports unhealthy: {data}")
                issues.append("CA service unhealthy")
        else:
            print(f"   ✗ CA health check failed: HTTP {response.status_code}")
            issues.append(f"CA health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ Cannot connect to CA service: {e}")
        issues.append(f"Cannot connect to CA service: {e}")
        return issues
    
    # Test 2: CA status
    print("2. Testing CA status...")
    try:
        response = requests.get(f"{ca_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ CA Status: {data.get('status')}")
            print(f"   - Active certificates: {data.get('active_certificates', 0)}")
            print(f"   - Total certificates: {data.get('total_certificates', 0)}")
        else:
            print(f"   ✗ CA status failed: HTTP {response.status_code}")
            issues.append("CA status endpoint failed")
    except Exception as e:
        print(f"   ✗ CA status error: {e}")
        issues.append(f"CA status error: {e}")
    
    # Test 3: Certificate generation
    print("3. Testing certificate generation...")
    try:
        test_client_id = "test_client"
        response = requests.post(
            f"{ca_url}/certificates/generate",
            json={"client_id": test_client_id, "permissions": "standard"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   ✓ Certificate generated for {test_client_id}")
                print(f"   - Certificate path: {data.get('certificate_path')}")
            else:
                print(f"   ✗ Certificate generation failed: {data}")
                issues.append("Certificate generation failed")
        else:
            print(f"   ✗ Certificate generation failed: HTTP {response.status_code}")
            issues.append(f"Certificate generation failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ Certificate generation error: {e}")
        issues.append(f"Certificate generation error: {e}")
    
    # Test 4: Certificate validation
    print("4. Testing certificate validation...")
    try:
        response = requests.get(f"{ca_url}/certificates/{test_client_id}/validate", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('valid'):
                print(f"   ✓ Certificate validation successful")
                print(f"   - Client ID: {data.get('client_id')}")
                print(f"   - Signature valid: {data.get('signature_valid')}")
            else:
                print(f"   ⚠ Certificate validation failed: {data.get('reason')}")
                issues.append(f"Certificate validation failed: {data.get('reason')}")
        else:
            print(f"   ✗ Certificate validation error: HTTP {response.status_code}")
            issues.append(f"Certificate validation error: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ Certificate validation error: {e}")
        issues.append(f"Certificate validation error: {e}")
    
    # Test 5: List certificates
    print("5. Testing certificate listing...")
    try:
        response = requests.get(f"{ca_url}/certificates", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Certificate list retrieved: {len(data)} certificates")
        else:
            print(f"   ✗ Certificate listing failed: HTTP {response.status_code}")
            issues.append("Certificate listing failed")
    except Exception as e:
        print(f"   ✗ Certificate listing error: {e}")
        issues.append(f"Certificate listing error: {e}")
    
    return issues

def test_server_ca_integration():
    """Test server CA integration"""
    print("\n" + "=" * 60)
    print("TESTING SERVER CA INTEGRATION")
    print("=" * 60)
    
    print("Note: Server CA integration can only be fully tested when server is running")
    print("Check server logs for certificate validation messages during FL training")
    
    return []

def test_dashboard_ca_integration():
    """Test dashboard CA integration"""
    print("\n" + "=" * 60)
    print("TESTING DASHBOARD CA INTEGRATION")
    print("=" * 60)
    
    dashboard_url = "http://localhost:5000"
    issues = []
    
    # Test dashboard CA endpoints
    ca_endpoints = [
        "/api/ca/status",
        "/api/ca/certificates"
    ]
    
    for endpoint in ca_endpoints:
        print(f"Testing dashboard endpoint: {endpoint}")
        try:
            response = requests.get(f"{dashboard_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"   ✓ {endpoint} working")
            else:
                print(f"   ⚠ {endpoint} returned HTTP {response.status_code}")
                issues.append(f"Dashboard {endpoint} failed")
        except Exception as e:
            print(f"   ✗ {endpoint} error: {e}")
            issues.append(f"Dashboard {endpoint} error: {e}")
    
    return issues

def main():
    """Main validation function"""
    print("CA INTEGRATION VALIDATION")
    print("=" * 80)
    print("This script validates the CA implementation and integration")
    print("Make sure the CA service is running on port 9000")
    print()
    
    all_issues = []
    
    # Test CA service
    all_issues.extend(test_ca_service())
    
    # Test server integration (informational)
    all_issues.extend(test_server_ca_integration())
    
    # Test dashboard integration
    all_issues.extend(test_dashboard_ca_integration())
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if not all_issues:
        print("🎉 ALL CA INTEGRATION TESTS PASSED!")
        print("The CA system is properly implemented and integrated.")
        return 0
    else:
        print(f"⚠️  FOUND {len(all_issues)} ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        
        print("\nRecommendations:")
        print("- Ensure CA service is running: docker-compose up ca")
        print("- Check CA service logs: docker-compose logs ca")
        print("- Verify network connectivity between services")
        return 1

if __name__ == "__main__":
    sys.exit(main())
