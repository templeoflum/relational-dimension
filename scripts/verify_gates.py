#!/usr/bin/env python3
"""
Universal gate checker for machine and inspection gates.

Usage:
    python scripts/verify_gates.py check experiments/01-topology-vs-correlation
    python scripts/verify_gates.py check all
    python scripts/verify_gates.py inspect experiments/01-topology-vs-correlation paper_pdf
    python scripts/verify_gates.py status experiments/01-topology-vs-correlation
"""

import json
import sys
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime


def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256.update(block)
    return sha256.hexdigest()


def load_gates(experiment_dir):
    """Load verification_gates.json."""
    gates_file = Path(experiment_dir) / "verification_gates.json"
    if not gates_file.exists():
        return {"machine_gates": [], "inspection_gates": []}
    
    with open(gates_file, 'r') as f:
        return json.load(f)


def save_gates(experiment_dir, gates_data):
    """Save verification_gates.json."""
    gates_file = Path(experiment_dir) / "verification_gates.json"
    with open(gates_file, 'w') as f:
        json.dump(gates_data, f, indent=2)


def check_machine_gates(experiment_dir):
    """Check all machine gates."""
    print("Checking machine gates...")
    all_passed = True
    
    # Gate 1: Claims verification
    print("  [1/2] Claims verification...", end=" ")
    result = subprocess.run(
        ["python", "scripts/verify_claims.py", str(experiment_dir)],
        capture_output=True
    )
    if result.returncode == 0:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        all_passed = False
    
    # Gate 2: Test suite
    print("  [2/2] Test suite...", end=" ")
    result = subprocess.run(
        ["python", "-m", "pytest", str(experiment_dir)],
        capture_output=True
    )
    if result.returncode == 0:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        all_passed = False
    
    return all_passed


def check_inspection_gates(experiment_dir):
    """Check all inspection gates."""
    print("\nChecking inspection gates...")
    gates_data = load_gates(experiment_dir)
    inspection_gates = gates_data.get("inspection_gates", [])
    
    if not inspection_gates:
        print("  No inspection gates defined.")
        return True
    
    all_passed = True
    for gate in inspection_gates:
        gate_id = gate["gate_id"]
        filepath = Path(experiment_dir) / gate["artifact_path"]
        
        print(f"  {gate_id}...", end=" ")
        
        if not filepath.exists():
            print("✗ ARTIFACT MISSING")
            all_passed = False
            continue
        
        current_hash = compute_file_hash(filepath)
        
        if "inspection" not in gate:
            print("⚠ NOT INSPECTED")
            all_passed = False
            continue
        
        inspection = gate["inspection"]
        if inspection["artifact_hash"] != current_hash:
            print("✗ HASH MISMATCH (artifact changed, re-inspection required)")
            all_passed = False
        else:
            inspector = inspection.get("inspector", "unknown")
            timestamp = inspection.get("timestamp", "unknown")
            print(f"✓ PASS (inspected by {inspector} at {timestamp})")
    
    return all_passed


def check_gates(experiment_dir):
    """Check all gates for an experiment."""
    print(f"\n{'='*60}")
    print(f"Checking gates for: {experiment_dir}")
    print(f"{'='*60}\n")
    
    machine_passed = check_machine_gates(experiment_dir)
    inspection_passed = check_inspection_gates(experiment_dir)
    
    print(f"\n{'='*60}")
    if machine_passed and inspection_passed:
        print("ALL GATES PASSED ✓")
        print(f"{'='*60}\n")
        return True
    else:
        print("GATE CHECK FAILED ✗")
        print(f"{'='*60}\n")
        return False


def record_inspection(experiment_dir, gate_id, inspector="human"):
    """Record that an inspection gate has been passed."""
    gates_data = load_gates(experiment_dir)
    inspection_gates = gates_data.get("inspection_gates", [])
    
    # Find the gate
    gate = None
    for g in inspection_gates:
        if g["gate_id"] == gate_id:
            gate = g
            break
    
    if not gate:
        print(f"ERROR: No inspection gate with id '{gate_id}' found.")
        return False
    
    # Compute current hash
    filepath = Path(experiment_dir) / gate["artifact_path"]
    if not filepath.exists():
        print(f"ERROR: Artifact {filepath} does not exist.")
        return False
    
    current_hash = compute_file_hash(filepath)
    
    # Record inspection
    gate["inspection"] = {
        "artifact_hash": current_hash,
        "timestamp": datetime.now().isoformat(),
        "inspector": inspector
    }
    
    save_gates(experiment_dir, gates_data)
    print(f"✓ Inspection recorded for {gate_id}")
    return True


def show_status(experiment_dir):
    """Show gate status for an experiment."""
    gates_data = load_gates(experiment_dir)
    
    print(f"\nGate status for: {experiment_dir}\n")
    
    print("Machine Gates:")
    print("  [1] Claims verification")
    print("  [2] Test suite")
    
    print("\nInspection Gates:")
    inspection_gates = gates_data.get("inspection_gates", [])
    if not inspection_gates:
        print("  None defined")
    else:
        for gate in inspection_gates:
            gate_id = gate["gate_id"]
            artifact = gate["artifact_path"]
            if "inspection" in gate:
                status = "✓ Inspected"
            else:
                status = "⚠ Not inspected"
            print(f"  {gate_id}: {artifact} - {status}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/verify_gates.py check EXPERIMENT_DIR")
        print("  python scripts/verify_gates.py check all")
        print("  python scripts/verify_gates.py inspect EXPERIMENT_DIR GATE_ID")
        print("  python scripts/verify_gates.py status EXPERIMENT_DIR")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "check":
        if len(sys.argv) != 3:
            print("Usage: python scripts/verify_gates.py check EXPERIMENT_DIR")
            sys.exit(1)
        
        experiment_dir = sys.argv[2]
        
        if experiment_dir == "all":
            # Check all experiments
            experiments_dir = Path("experiments")
            all_passed = True
            for exp_dir in sorted(experiments_dir.iterdir()):
                if exp_dir.is_dir():
                    passed = check_gates(str(exp_dir))
                    if not passed:
                        all_passed = False
            sys.exit(0 if all_passed else 1)
        else:
            success = check_gates(experiment_dir)
            sys.exit(0 if success else 1)
    
    elif command == "inspect":
        if len(sys.argv) != 4:
            print("Usage: python scripts/verify_gates.py inspect EXPERIMENT_DIR GATE_ID")
            sys.exit(1)
        
        experiment_dir = sys.argv[2]
        gate_id = sys.argv[3]
        success = record_inspection(experiment_dir, gate_id)
        sys.exit(0 if success else 1)
    
    elif command == "status":
        if len(sys.argv) != 3:
            print("Usage: python scripts/verify_gates.py status EXPERIMENT_DIR")
            sys.exit(1)
        
        experiment_dir = sys.argv[2]
        show_status(experiment_dir)
        sys.exit(0)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
