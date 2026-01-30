#!/usr/bin/env python3
"""
Verify that all quantitative claims in prose match the source-of-truth metrics data.

Usage:
    python scripts/verify_claims.py experiments/01-topology-vs-correlation
"""

import json
import sys
from pathlib import Path


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_nested_value(data, path):
    """
    Extract value from nested dict using path like "predictions[1].measured"
    """
    keys = path.replace('[', '.').replace(']', '').split('.')
    value = data
    for key in keys:
        if key.isdigit():
            value = value[int(key)]
        else:
            value = value[key]
    return value


def compare_values(expected, actual, comparison, tolerance):
    """Compare two values according to specified comparison type."""
    if comparison == "equals":
        return expected == actual
    elif comparison == "approximately_equals":
        return abs(expected - actual) <= tolerance
    elif comparison == "greater_than":
        return actual > expected
    elif comparison == "less_than":
        return actual < expected
    elif comparison == "greater_equal":
        return actual >= expected
    elif comparison == "less_equal":
        return actual <= expected
    else:
        raise ValueError(f"Unknown comparison type: {comparison}")


def verify_claims(experiment_dir):
    """Verify all claims in claims.json against metrics."""
    experiment_path = Path(experiment_dir)
    
    # Load claims manifest
    claims_file = experiment_path / "claims.json"
    if not claims_file.exists():
        print(f"ERROR: No claims.json found in {experiment_dir}")
        return False
    
    claims_data = load_json(claims_file)
    
    # Load metrics (source of truth)
    metrics_file = experiment_path / "output" / "metrics.json"
    if not metrics_file.exists():
        print(f"ERROR: No metrics.json found in {experiment_dir}/output/")
        return False
    
    metrics_data = load_json(metrics_file)
    
    # Verify each claim
    all_passed = True
    claims = claims_data.get("claims", [])
    
    print(f"\nVerifying {len(claims)} claims from {experiment_dir}...\n")
    
    for i, claim in enumerate(claims, 1):
        claim_id = claim.get("id", f"claim_{i}")
        description = claim.get("description", "No description")
        metric_path = claim["metric_path"]
        expected = claim["expected_value"]
        tolerance = claim.get("tolerance", 0)
        comparison = claim.get("comparison", "equals")
        
        try:
            actual = get_nested_value(metrics_data, metric_path)
            passed = compare_values(expected, actual, comparison, tolerance)
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {claim_id}")
            print(f"      {description}")
            print(f"      Expected: {expected} | Actual: {actual}")
            
            if not passed:
                all_passed = False
                print(f"      Tolerance: ±{tolerance} | Comparison: {comparison}")
            print()
            
        except (KeyError, IndexError, TypeError) as e:
            print(f"✗ ERROR | {claim_id}")
            print(f"      Could not extract metric: {metric_path}")
            print(f"      Error: {e}\n")
            all_passed = False
    
    if all_passed:
        print("="*60)
        print("ALL CLAIMS VERIFIED ✓")
        print("="*60)
        return True
    else:
        print("="*60)
        print("VERIFICATION FAILED ✗")
        print("="*60)
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/verify_claims.py EXPERIMENT_DIR")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    success = verify_claims(experiment_dir)
    sys.exit(0 if success else 1)
