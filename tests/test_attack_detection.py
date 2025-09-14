#!/usr/bin/env python3
"""
Unit Tests for Federated Learning Attack Detection
=================================================

This script contains smoke tests and unit tests for the attack detection system.
It verifies that the system works correctly with different configurations.

Usage:
    python tests/test_attack_detection.py
    pytest tests/test_attack_detection.py -v

Author: FL Defense System
Date: 2025
"""

import os
import sys
import tempfile
import shutil
import unittest
import subprocess
import json
import pickle
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_defenses.detector import (
    apply_ldp, eliminate_kmeans, calculate_accuracy, 
    calculate_f1_score, detect_malicious_clients
)


class TestAttackDetection(unittest.TestCase):
    """Test cases for the attack detection system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_losses = [0.1, 0.12, 0.11, 0.8, 0.9, 0.85]  # First 3 benign, last 3 malicious
        self.true_malicious = [3, 4, 5]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_apply_ldp(self):
        """Test LDP noise application."""
        # Test basic functionality
        noisy_losses = apply_ldp(self.test_losses, epsilon=1.0, sensitivity=1e-4)
        
        self.assertEqual(len(noisy_losses), len(self.test_losses))
        self.assertTrue(all(isinstance(x, float) for x in noisy_losses))
        
        # Test with different parameters
        noisy_losses_2 = apply_ldp(self.test_losses, epsilon=0.5, sensitivity=1e-3)
        self.assertEqual(len(noisy_losses_2), len(self.test_losses))
        
        # Test with empty input
        empty_result = apply_ldp([])
        self.assertEqual(empty_result, [])
    
    def test_eliminate_kmeans(self):
        """Test K-means clustering for malicious client detection."""
        # Test with clear separation
        malicious_indices, metrics = eliminate_kmeans(self.test_losses, n_clusters=2)
        
        self.assertIsInstance(malicious_indices, list)
        self.assertIsInstance(metrics, dict)
        self.assertIn("cluster_means", metrics)
        self.assertIn("cluster_sizes", metrics)
        self.assertIn("malicious_cluster_id", metrics)
        
        # Test with insufficient data
        insufficient_data = [0.1, 0.2]
        malicious_indices, metrics = eliminate_kmeans(insufficient_data, n_clusters=2)
        self.assertIn("error", metrics)
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        # Test with perfect detection
        accuracy = calculate_accuracy(self.true_malicious, self.true_malicious, 6)
        self.assertEqual(accuracy, 1.0)
        
        # Test with no detection
        accuracy = calculate_accuracy(self.true_malicious, [], 6)
        self.assertEqual(accuracy, 0.5)  # 3 correct negatives out of 6 total
        
        # Test with partial detection
        partial_detection = [3, 4, 6]  # One false positive, one false negative
        accuracy = calculate_accuracy(self.true_malicious, partial_detection, 6)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Test with zero total clients
        accuracy = calculate_accuracy([], [], 0)
        self.assertEqual(accuracy, 0.0)
    
    def test_calculate_f1_score(self):
        """Test F1 score calculation."""
        # Test with perfect detection
        f1 = calculate_f1_score(self.true_malicious, self.true_malicious)
        self.assertEqual(f1, 1.0)
        
        # Test with no detection
        f1 = calculate_f1_score(self.true_malicious, [])
        self.assertEqual(f1, 0.0)
        
        # Test with partial detection
        partial_detection = [3, 4, 6]  # One false positive, one false negative
        f1 = calculate_f1_score(self.true_malicious, partial_detection)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        
        # Test with no malicious clients
        f1 = calculate_f1_score([], [])
        self.assertEqual(f1, 1.0)
    
    def test_detect_malicious_clients(self):
        """Test complete detection pipeline."""
        # Test with ground truth
        results = detect_malicious_clients(
            self.test_losses, 
            true_malicious=self.true_malicious,
            epsilon=1.0,
            sensitivity=1e-4
        )
        
        self.assertIn("detected_malicious", results)
        self.assertIn("detection_metrics", results)
        self.assertIn("accuracy", results)
        self.assertIn("f1_score", results)
        self.assertIn("original_losses", results)
        self.assertIn("noisy_losses", results)
        
        # Test without ground truth
        results_no_gt = detect_malicious_clients(self.test_losses)
        self.assertIn("detected_malicious", results_no_gt)
        self.assertNotIn("accuracy", results_no_gt)
        self.assertNotIn("f1_score", results_no_gt)
        
        # Test with empty input
        results_empty = detect_malicious_clients([])
        self.assertIn("error", results_empty)


class TestSmokeTests(unittest.TestCase):
    """Smoke tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_attack_detection.py")
        
        # Create a dummy dataset for testing
        self.create_dummy_dataset()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_dummy_dataset(self):
        """Create a dummy Lending Club dataset for testing."""
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        n_samples = 1000
        data = {
            'id': range(n_samples),
            'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], n_samples, p=[0.8, 0.2]),
            'loan_amnt': np.random.normal(10000, 3000, n_samples),
            'funded_amnt': np.random.normal(10000, 3000, n_samples),
            'funded_amnt_inv': np.random.normal(10000, 3000, n_samples),
            'term': np.random.choice(['36 months', '60 months'], n_samples),
            'int_rate': np.random.normal(12, 3, n_samples),
            'installment': np.random.normal(300, 100, n_samples),
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples),
            'sub_grade': np.random.choice(['A1', 'A2', 'A3', 'A4', 'A5'], n_samples),
            'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years'], n_samples),
            'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples),
            'annual_inc': np.random.normal(50000, 20000, n_samples),
            'verification_status': np.random.choice(['Verified', 'Not Verified'], n_samples),
            'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement'], n_samples),
            'addr_state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_samples),
            'dti': np.random.normal(15, 5, n_samples),
            'delinq_2yrs': np.random.poisson(0.5, n_samples),
            'inq_last_6mths': np.random.poisson(1, n_samples),
            'open_acc': np.random.poisson(8, n_samples),
            'pub_rec': np.random.poisson(0.2, n_samples),
            'revol_bal': np.random.normal(10000, 5000, n_samples),
            'revol_util': np.random.normal(50, 20, n_samples),
            'total_acc': np.random.poisson(20, n_samples),
            'initial_list_status': np.random.choice(['f', 'w'], n_samples),
            'out_prncp': np.random.normal(5000, 2000, n_samples),
            'out_prncp_inv': np.random.normal(5000, 2000, n_samples),
            'total_pymnt': np.random.normal(12000, 3000, n_samples),
            'total_pymnt_inv': np.random.normal(12000, 3000, n_samples),
            'total_rec_prncp': np.random.normal(10000, 2000, n_samples),
            'total_rec_int': np.random.normal(2000, 500, n_samples),
            'total_rec_late_fee': np.random.normal(50, 20, n_samples),
            'recoveries': np.random.normal(100, 50, n_samples),
            'collection_recovery_fee': np.random.normal(50, 20, n_samples),
            'last_pymnt_d': '2020-01-01',
            'last_pymnt_amnt': np.random.normal(300, 100, n_samples),
            'next_pymnt_d': '2020-02-01',
            'last_credit_pull_d': '2020-01-01',
            'last_fico_range_high': np.random.normal(750, 50, n_samples),
            'last_fico_range_low': np.random.normal(700, 50, n_samples),
            'collections_12_mths_ex_med': np.random.poisson(0.1, n_samples),
            'mths_since_last_major_derog': np.random.normal(24, 12, n_samples),
            'policy_code': np.random.choice([1], n_samples),
            'application_type': np.random.choice(['Individual'], n_samples),
            'annual_inc_joint': np.random.normal(60000, 25000, n_samples),
            'dti_joint': np.random.normal(15, 5, n_samples),
            'verification_status_joint': np.random.choice(['Verified', 'Not Verified'], n_samples),
            'acc_now_delinq': np.random.poisson(0.1, n_samples),
            'tot_coll_amt': np.random.normal(1000, 500, n_samples),
            'tot_cur_bal': np.random.normal(20000, 10000, n_samples),
            'open_acc_6m': np.random.poisson(2, n_samples),
            'open_act_il': np.random.poisson(3, n_samples),
            'open_il_12m': np.random.poisson(1, n_samples),
            'open_il_24m': np.random.poisson(2, n_samples),
            'mths_since_rcnt_il': np.random.normal(12, 6, n_samples),
            'total_bal_il': np.random.normal(5000, 2000, n_samples),
            'il_util': np.random.normal(30, 15, n_samples),
            'open_rv_12m': np.random.poisson(1, n_samples),
            'open_rv_24m': np.random.poisson(2, n_samples),
            'max_bal_bc': np.random.normal(8000, 3000, n_samples),
            'all_util': np.random.normal(40, 20, n_samples),
            'total_rev_hi_lim': np.random.normal(15000, 5000, n_samples),
            'inq_fi': np.random.poisson(1, n_samples),
            'total_cu_tl': np.random.poisson(5, n_samples),
            'inq_last_12m': np.random.poisson(2, n_samples),
            'acc_open_past_24mths': np.random.poisson(3, n_samples),
            'avg_cur_bal': np.random.normal(5000, 2000, n_samples),
            'bc_open_to_buy': np.random.normal(2000, 1000, n_samples),
            'bc_util': np.random.normal(30, 15, n_samples),
            'chargeoff_within_12_mths': np.random.poisson(0.1, n_samples),
            'delinq_amnt': np.random.normal(100, 50, n_samples),
            'mo_sin_old_il_acct': np.random.normal(60, 30, n_samples),
            'mo_sin_old_rev_tl_op': np.random.normal(48, 24, n_samples),
            'mo_sin_rcnt_rev_tl_op': np.random.normal(12, 6, n_samples),
            'mo_sin_rcnt_tl': np.random.normal(18, 9, n_samples),
            'mort_acc': np.random.poisson(1, n_samples),
            'mths_since_recent_bc': np.random.normal(6, 3, n_samples),
            'mths_since_recent_bc_dlq': np.random.normal(12, 6, n_samples),
            'mths_since_recent_inq': np.random.normal(3, 1.5, n_samples),
            'mths_since_recent_revol_delinq': np.random.normal(24, 12, n_samples),
            'num_accts_ever_120_pd': np.random.poisson(0.5, n_samples),
            'num_actv_bc_tl': np.random.poisson(2, n_samples),
            'num_actv_rev_tl': np.random.poisson(3, n_samples),
            'num_bc_sats': np.random.poisson(5, n_samples),
            'num_bc_tl': np.random.poisson(4, n_samples),
            'num_il_tl': np.random.poisson(2, n_samples),
            'num_op_rev_tl': np.random.poisson(3, n_samples),
            'num_rev_accts': np.random.poisson(4, n_samples),
            'num_rev_tl_bal_gt_0': np.random.poisson(3, n_samples),
            'num_sats': np.random.poisson(6, n_samples),
            'num_tl_120dpd_2m': np.random.poisson(0.1, n_samples),
            'num_tl_30dpd': np.random.poisson(0.2, n_samples),
            'num_tl_90g_dpd_24m': np.random.poisson(0.1, n_samples),
            'num_tl_op_past_12m': np.random.poisson(2, n_samples),
            'pct_tl_nvr_dlq': np.random.normal(95, 5, n_samples),
            'percent_bc_gt_75': np.random.normal(20, 10, n_samples),
            'pub_rec_bankruptcies': np.random.poisson(0.1, n_samples),
            'tax_liens': np.random.poisson(0.05, n_samples),
            'tot_hi_cred_lim': np.random.normal(25000, 10000, n_samples),
            'total_bal_ex_mort': np.random.normal(15000, 5000, n_samples),
            'total_bc_limit': np.random.normal(10000, 4000, n_samples),
            'total_il_high_credit_limit': np.random.normal(5000, 2000, n_samples),
            'revol_util_joint': np.random.normal(50, 20, n_samples),
            'sec_app_fico_range_low': np.random.normal(700, 50, n_samples),
            'sec_app_fico_range_high': np.random.normal(750, 50, n_samples),
            'sec_app_earliest_cr_line': '2000-01-01',
            'sec_app_inq_last_6mths': np.random.poisson(1, n_samples),
            'sec_app_mort_acc': np.random.poisson(1, n_samples),
            'sec_app_open_acc': np.random.poisson(3, n_samples),
            'sec_app_revol_util': np.random.normal(30, 15, n_samples),
            'sec_app_open_act_il': np.random.poisson(1, n_samples),
            'sec_app_num_rev_accts': np.random.poisson(2, n_samples),
            'sec_app_chargeoff_within_12_mths': np.random.poisson(0.1, n_samples),
            'sec_app_collections_12_mths_ex_med': np.random.poisson(0.1, n_samples),
            'sec_app_mths_since_last_major_derog': np.random.normal(24, 12, n_samples),
            'hardship_flag': np.random.choice(['N', 'Y'], n_samples, p=[0.95, 0.05]),
            'hardship_type': np.random.choice(['debt_consolidation', 'medical', 'other'], n_samples),
            'hardship_reason': np.random.choice(['debt_consolidation', 'medical', 'other'], n_samples),
            'hardship_status': np.random.choice(['ACTIVE', 'PAID_OFF', 'CANCELLED'], n_samples),
            'deferral_term': np.random.normal(3, 1, n_samples),
            'hardship_amount': np.random.normal(5000, 2000, n_samples),
            'hardship_start_date': '2020-01-01',
            'hardship_end_date': '2020-04-01',
            'payment_plan_start_date': '2020-01-01',
            'hardship_length': np.random.normal(3, 1, n_samples),
            'hardship_dpd': np.random.poisson(0.5, n_samples),
            'hardship_loan_status': np.random.choice(['Current', 'Paid Off'], n_samples),
            'orig_projected_additional_accrued_interest': np.random.normal(100, 50, n_samples),
            'hardship_payoff_balance_amount': np.random.normal(10000, 3000, n_samples),
            'hardship_last_payment_amount': np.random.normal(300, 100, n_samples),
            'disbursement_method': np.random.choice(['Cash', 'DirectPay'], n_samples),
            'debt_settlement_flag': np.random.choice(['N', 'Y'], n_samples, p=[0.95, 0.05]),
            'debt_settlement_flag_date': '2020-01-01',
            'settlement_status': np.random.choice(['Settled', 'Not Settled'], n_samples),
            'settlement_amount': np.random.normal(5000, 2000, n_samples),
            'settlement_percentage': np.random.normal(80, 10, n_samples),
            'settlement_term': np.random.normal(12, 3, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save to temporary file
        self.dataset_path = os.path.join(self.temp_dir, "smallLendingClub.csv")
        df.to_csv(self.dataset_path, index=False)
    
    def test_zero_malicious_percentage(self):
        """Test with 0% malicious clients (should complete without errors)."""
        cmd = [
            "python", self.script_path,
            "--dataset", self.dataset_path,
            "--epochs", "5",
            "--n_train_clients", "5",
            "--n_total_clients", "5",
            "--malicious-percentages", "0",
            "--output-dir", self.temp_dir,
            "--epsilon", "1.0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        
        # Check that results file exists
        results_file = os.path.join(self.temp_dir, "lending_club_results.pkl")
        self.assertTrue(os.path.exists(results_file), "Results file not created")
        
        # Load and verify results
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        self.assertIn("malicious_0.0pct", results)
        exp_results = results["malicious_0.0pct"]
        
        # Should have no malicious clients detected
        self.assertEqual(exp_results["true_malicious"], [])
        
        # Check that experiment completed
        self.assertGreater(len(exp_results["epoch_results"]), 0)
        self.assertIn("final_metrics", exp_results)
    
    def test_ten_malicious_percentage(self):
        """Test with 10% malicious clients (should complete and detect some attacks)."""
        cmd = [
            "python", self.script_path,
            "--dataset", self.dataset_path,
            "--epochs", "5",
            "--n_train_clients", "5",
            "--n_total_clients", "5",
            "--malicious-percentages", "10",
            "--output-dir", self.temp_dir,
            "--epsilon", "1.0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Should complete successfully
        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        
        # Check that results file exists
        results_file = os.path.join(self.temp_dir, "lending_club_results.pkl")
        self.assertTrue(os.path.exists(results_file), "Results file not created")
        
        # Load and verify results
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        self.assertIn("malicious_10.0pct", results)
        exp_results = results["malicious_10.0pct"]
        
        # Should have some malicious clients
        self.assertGreater(len(exp_results["true_malicious"]), 0)
        
        # Check that experiment completed
        self.assertGreater(len(exp_results["epoch_results"]), 0)
        self.assertIn("final_metrics", exp_results)
        
        # Check that detection metrics are present
        final_metrics = exp_results["final_metrics"]
        self.assertIn("avg_detection_accuracy", final_metrics)
        self.assertIn("avg_detection_f1", final_metrics)
    
    def test_script_help(self):
        """Test that the script shows help when run with --help."""
        cmd = ["python", self.script_path, "--help"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Should show help and exit with code 0
        self.assertEqual(result.returncode, 0)
        self.assertIn("Federated Learning Attack Detection", result.stdout)
        self.assertIn("--dataset", result.stdout)
        self.assertIn("--epochs", result.stdout)


def run_smoke_tests():
    """Run all smoke tests."""
    print("Running Federated Learning Attack Detection Smoke Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAttackDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestSmokeTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)



