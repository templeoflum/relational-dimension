"""
Phase 1: Calibration logic.

Tests the measurement method on known systems (1D, 2D, 3D lattices)
to verify it can recover ground truth dimensions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from lattice_generation import generate_calibration_system
from distance_transforms import apply_transform, list_transforms
from dimension_extraction import full_dimension_analysis


@dataclass
class CalibrationResult:
    """Result from a single calibration test."""
    system_type: str
    true_dimension: int
    d_topo: Optional[float]
    d_corr: Optional[float]
    delta: Optional[float]
    valid: bool
    method_diff_topo: float
    method_diff_corr: float
    transform_used: str


@dataclass
class CalibrationGateStatus:
    """Status of all calibration gates."""
    c1_2d_calibration: bool
    c2_3d_calibration: bool
    c3_1d_calibration: bool
    c4_method_agreement: bool
    c5_reproducibility: bool
    all_passed: bool
    details: Dict


def run_single_calibration(system_type: str, transform: str,
                           n_nodes: int = 100,
                           agreement_threshold: float = 0.5) -> CalibrationResult:
    """
    Run calibration on a single system.

    Args:
        system_type: '1d', '2d', or '3d'
        transform: Name of distance transform to use
        n_nodes: Number of nodes
        agreement_threshold: Method agreement threshold

    Returns:
        CalibrationResult
    """
    # Generate system
    system = generate_calibration_system(system_type, n_nodes)

    # Apply distance transform to correlation matrix
    D_corr = apply_transform(system['C'], transform)
    D_topo = system['D_topo']

    # Extract dimensions
    result = full_dimension_analysis(D_topo, D_corr, agreement_threshold)

    return CalibrationResult(
        system_type=system_type,
        true_dimension=system['metadata']['true_dimension'],
        d_topo=result['d_topo'],
        d_corr=result['d_corr'],
        delta=result['delta'],
        valid=result['valid'],
        method_diff_topo=result['topology']['method_diff'],
        method_diff_corr=result['correlation']['method_diff'],
        transform_used=transform
    )


def run_calibration_batch(system_type: str, transform: str,
                          n_replications: int = 50,
                          n_nodes: int = 100) -> Dict:
    """
    Run multiple calibration replications.

    Args:
        system_type: '1d', '2d', or '3d'
        transform: Distance transform name
        n_replications: Number of replications
        n_nodes: Nodes per system

    Returns:
        Aggregated results dictionary
    """
    results = []
    valid_deltas = []
    method_diffs = []

    for i in range(n_replications):
        result = run_single_calibration(system_type, transform, n_nodes)
        results.append(result)

        if result.valid and result.delta is not None:
            valid_deltas.append(result.delta)
            method_diffs.append(max(result.method_diff_topo, result.method_diff_corr))

    n_valid = len(valid_deltas)
    exclusion_rate = 1 - (n_valid / n_replications)

    if n_valid > 0:
        delta_mean = np.mean(valid_deltas)
        delta_std = np.std(valid_deltas)
        max_method_diff = max(method_diffs)
    else:
        delta_mean = None
        delta_std = None
        max_method_diff = None

    return {
        'system_type': system_type,
        'transform': transform,
        'n_replications': n_replications,
        'n_valid': n_valid,
        'exclusion_rate': exclusion_rate,
        'delta_mean': delta_mean,
        'delta_std': delta_std,
        'max_method_diff': max_method_diff,
        'all_deltas': valid_deltas,
        'results': results
    }


def find_best_transform(n_nodes: int = 100, n_replications: int = 20) -> Tuple[str, Dict]:
    """
    Find the transform that achieves best calibration.

    Tests all transforms on all system types and selects
    the one with lowest max |delta|.

    Returns:
        Tuple of (best_transform_name, results_dict)
    """
    transforms = list_transforms()
    system_types = ['1d', '2d', '3d']

    results_by_transform = {}

    for transform in transforms:
        transform_results = {}
        max_abs_delta = 0

        for system_type in system_types:
            batch = run_calibration_batch(system_type, transform,
                                          n_replications, n_nodes)
            transform_results[system_type] = batch

            if batch['delta_mean'] is not None:
                max_abs_delta = max(max_abs_delta, abs(batch['delta_mean']))
            else:
                max_abs_delta = float('inf')

        transform_results['max_abs_delta'] = max_abs_delta
        results_by_transform[transform] = transform_results

    # Find best transform
    best_transform = min(results_by_transform.keys(),
                         key=lambda t: results_by_transform[t]['max_abs_delta'])

    return best_transform, results_by_transform


def evaluate_calibration_gates(results: Dict,
                               delta_threshold: float = 0.1,
                               method_threshold: float = 0.3,
                               std_threshold: float = 0.05) -> CalibrationGateStatus:
    """
    Evaluate all calibration gates.

    Args:
        results: Results from calibration batch runs for 1d, 2d, 3d
        delta_threshold: Maximum |delta| for calibration (C1-C3)
        method_threshold: Maximum method disagreement (C4)
        std_threshold: Maximum std for reproducibility (C5)

    Returns:
        CalibrationGateStatus with pass/fail for each gate
    """
    # C1: 2D lattice calibration
    r2d = results.get('2d', {})
    c1 = (r2d.get('delta_mean') is not None and
          abs(r2d.get('delta_mean', 1)) < delta_threshold)

    # C2: 3D lattice calibration
    r3d = results.get('3d', {})
    c2 = (r3d.get('delta_mean') is not None and
          abs(r3d.get('delta_mean', 1)) < delta_threshold)

    # C3: 1D chain calibration
    r1d = results.get('1d', {})
    c3 = (r1d.get('delta_mean') is not None and
          abs(r1d.get('delta_mean', 1)) < delta_threshold)

    # C4: Method agreement across all
    max_method_diffs = [
        r.get('max_method_diff', 1) for r in [r1d, r2d, r3d]
        if r.get('max_method_diff') is not None
    ]
    c4 = len(max_method_diffs) > 0 and max(max_method_diffs) < method_threshold

    # C5: Reproducibility
    stds = [
        r.get('delta_std', 1) for r in [r1d, r2d, r3d]
        if r.get('delta_std') is not None
    ]
    c5 = len(stds) > 0 and max(stds) < std_threshold

    all_passed = c1 and c2 and c3 and c4 and c5

    details = {
        'c1_delta_2d': r2d.get('delta_mean'),
        'c2_delta_3d': r3d.get('delta_mean'),
        'c3_delta_1d': r1d.get('delta_mean'),
        'c4_max_method_diff': max(max_method_diffs) if max_method_diffs else None,
        'c5_max_std': max(stds) if stds else None,
        'thresholds': {
            'delta': delta_threshold,
            'method': method_threshold,
            'std': std_threshold
        }
    }

    return CalibrationGateStatus(
        c1_2d_calibration=c1,
        c2_3d_calibration=c2,
        c3_1d_calibration=c3,
        c4_method_agreement=c4,
        c5_reproducibility=c5,
        all_passed=all_passed,
        details=details
    )


def run_full_calibration(n_nodes: int = 100,
                         n_replications: int = 50) -> Dict:
    """
    Run complete Phase 1 calibration.

    1. Find best transform
    2. Run full calibration with best transform
    3. Evaluate all gates

    Returns:
        Complete calibration results
    """
    print("Phase 1: Calibration")
    print("=" * 50)

    # Step 1: Find best transform
    print("\nStep 1: Finding best distance transform...")
    best_transform, transform_results = find_best_transform(n_nodes, n_replications=10)
    print(f"  Best transform: {best_transform}")

    # Step 2: Full calibration with best transform
    print(f"\nStep 2: Running full calibration with {best_transform}...")
    full_results = {}
    for system_type in ['1d', '2d', '3d']:
        print(f"  Testing {system_type}...", end=" ")
        batch = run_calibration_batch(system_type, best_transform,
                                       n_replications, n_nodes)
        full_results[system_type] = batch
        if batch['delta_mean'] is not None:
            print(f"delta = {batch['delta_mean']:.4f} (std = {batch['delta_std']:.4f})")
        else:
            print("FAILED - no valid results")

    # Step 3: Evaluate gates
    print("\nStep 3: Evaluating calibration gates...")
    gate_status = evaluate_calibration_gates(full_results)

    print(f"  C1 (2D calibration): {'PASS' if gate_status.c1_2d_calibration else 'FAIL'}")
    print(f"  C2 (3D calibration): {'PASS' if gate_status.c2_3d_calibration else 'FAIL'}")
    print(f"  C3 (1D calibration): {'PASS' if gate_status.c3_1d_calibration else 'FAIL'}")
    print(f"  C4 (Method agreement): {'PASS' if gate_status.c4_method_agreement else 'FAIL'}")
    print(f"  C5 (Reproducibility): {'PASS' if gate_status.c5_reproducibility else 'FAIL'}")
    print(f"\n  ALL GATES: {'PASS' if gate_status.all_passed else 'FAIL'}")

    return {
        'best_transform': best_transform,
        'transform_comparison': transform_results,
        'full_results': full_results,
        'gate_status': gate_status,
        'calibration_passed': gate_status.all_passed
    }
