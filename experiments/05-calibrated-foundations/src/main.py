"""
Main orchestrator for Experiment 05: Calibrated Foundations.

Runs all three phases with strict gating.
"""

import os
import json
from datetime import datetime
from dataclasses import asdict

from calibration import run_full_calibration
from compression_tests import run_full_compression_tests
from effect_size import run_effect_size_analysis
from diagnostics import (
    generate_phase1_outputs,
    generate_phase2_outputs,
    generate_phase3_outputs
)


def save_json(data: dict, path: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert dataclasses and numpy types
    def convert(obj):
        if hasattr(obj, '__dict__'):
            return {k: convert(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2, default=str)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Experiment 05: Calibrated Foundations")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(exp_dir, 'output')
    reports_dir = os.path.join(exp_dir, 'reports')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Results accumulator
    results = {
        'experiment': '05-calibrated-foundations',
        'timestamp': datetime.now().isoformat(),
        'phases': {}
    }

    # =========================================
    # PHASE 1: CALIBRATION (BLOCKING)
    # =========================================
    print("\n" + "=" * 60)
    calibration_results = run_full_calibration(n_nodes=100, n_replications=50)

    # Save Phase 1 results
    save_json(calibration_results, os.path.join(output_dir, 'phase1', 'calibration_results.json'))

    gate_status = calibration_results.get('gate_status')
    gate_dict = {
        'c1_2d_calibration': gate_status.c1_2d_calibration,
        'c2_3d_calibration': gate_status.c2_3d_calibration,
        'c3_1d_calibration': gate_status.c3_1d_calibration,
        'c4_method_agreement': gate_status.c4_method_agreement,
        'c5_reproducibility': gate_status.c5_reproducibility,
        'all_passed': gate_status.all_passed,
        'details': gate_status.details
    }
    save_json(gate_dict, os.path.join(output_dir, 'phase1', 'calibration_gate_status.json'))

    # Generate Phase 1 outputs
    generate_phase1_outputs(calibration_results, reports_dir)

    results['phases']['phase1'] = {
        'status': 'passed' if gate_status.all_passed else 'failed',
        'gates': gate_dict,
        'best_transform': calibration_results.get('best_transform')
    }

    # CHECK BLOCKING GATE
    if not gate_status.all_passed:
        print("\n" + "=" * 60)
        print("PHASE 1 FAILED - STOPPING EXPERIMENT")
        print("=" * 60)
        print("\nCalibration gates did not pass. Cannot proceed to Phase 2.")
        print("The measurement method does not reliably recover known dimensions.")

        results['final_status'] = 'calibration_failed'
        results['phases']['phase2'] = {'status': 'skipped', 'reason': 'Phase 1 failed'}
        results['phases']['phase3'] = {'status': 'skipped', 'reason': 'Phase 1 failed'}

        save_json(results, os.path.join(output_dir, 'final_results.json'))

        print(f"\nResults saved to {output_dir}")
        return results

    # =========================================
    # PHASE 2: COMPRESSION TESTS
    # =========================================
    print("\n" + "=" * 60)
    best_transform = calibration_results.get('best_transform')
    compression_results = run_full_compression_tests(
        transform=best_transform,
        n_replications=50,
        side=10
    )

    # Save Phase 2 results
    save_json(compression_results, os.path.join(output_dir, 'phase2', 'compression_results.json'))

    # Generate Phase 2 outputs
    generate_phase2_outputs(compression_results, reports_dir)

    results['phases']['phase2'] = {
        'status': 'completed',
        'predictions': compression_results.get('predictions'),
        'passed_count': compression_results.get('passed_count')
    }

    # =========================================
    # PHASE 3: EFFECT SIZE ESTIMATION
    # =========================================
    print("\n" + "=" * 60)
    effect_analysis = run_effect_size_analysis(compression_results)

    # Save Phase 3 results
    save_json(effect_analysis, os.path.join(output_dir, 'phase3', 'effect_size_analysis.json'))

    # Generate Phase 3 outputs
    generate_phase3_outputs(effect_analysis, reports_dir)

    results['phases']['phase3'] = {
        'status': 'completed',
        'predictions': effect_analysis.get('predictions'),
        'main_effect_size': effect_analysis.get('main_effect_size'),
        'n_required_power80': effect_analysis.get('n_required_power80')
    }

    # =========================================
    # FINAL SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Count predictions
    phase1_passed = gate_status.all_passed
    phase2_predictions = compression_results.get('predictions', {})
    phase2_passed = sum(1 for p in phase2_predictions.values() if p.get('passed', False))
    phase3_predictions = effect_analysis.get('predictions', {})
    phase3_passed = sum(1 for p in phase3_predictions.values() if p.get('passed', False))

    total_predictions = 5 + 4 + 2  # P1.1-P1.5, P2.1-P2.4, P3.1-P3.2
    total_passed = (5 if phase1_passed else 0) + phase2_passed + phase3_passed

    print(f"\nPhase 1 (Calibration): {'PASSED' if phase1_passed else 'FAILED'}")
    print(f"  - All 5 calibration gates: {'PASS' if phase1_passed else 'FAIL'}")

    print(f"\nPhase 2 (Compression): {phase2_passed}/4 predictions passed")
    for pred_id, pred in phase2_predictions.items():
        status = 'PASS' if pred.get('passed') else 'FAIL'
        print(f"  - {pred_id}: {status}")

    print(f"\nPhase 3 (Effect Size): {phase3_passed}/2 predictions passed")
    for pred_id, pred in phase3_predictions.items():
        status = 'PASS' if pred.get('passed') else 'FAIL'
        print(f"  - {pred_id}: {status}")

    print(f"\nTOTAL: {total_passed}/{total_predictions} predictions passed")

    # Determine overall status
    if phase1_passed and phase2_passed >= 2 and effect_analysis.get('main_effect_size', 0) > 0.3:
        final_status = 'success'
        print("\nSTATUS: SUCCESS - Ready to proceed to Experiment 06")
    elif phase1_passed:
        final_status = 'partial_success'
        print("\nSTATUS: PARTIAL SUCCESS - Calibration passed but effect is weak")
    else:
        final_status = 'failed'
        print("\nSTATUS: FAILED - Calibration did not pass")

    results['final_status'] = final_status
    results['summary'] = {
        'total_predictions': total_predictions,
        'total_passed': total_passed,
        'phase1_passed': phase1_passed,
        'phase2_passed': phase2_passed,
        'phase3_passed': phase3_passed
    }

    # Save final results
    save_json(results, os.path.join(output_dir, 'final_results.json'))

    print(f"\nResults saved to {output_dir}")
    print(f"Completed: {datetime.now().isoformat()}")

    return results


if __name__ == '__main__':
    main()
