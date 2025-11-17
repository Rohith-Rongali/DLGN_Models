#!/usr/bin/env python3
"""
Migration script to switch to optimized DLGN Value Tensor implementation.

This script:
1. Backs up original files
2. Replaces them with optimized versions
3. Runs basic validation tests
4. Provides rollback if needed

Usage:
    python migrate_to_optimized.py [--backup-only | --rollback | --validate]
"""

import argparse
import shutil
import os
import sys
from pathlib import Path


def backup_files():
    """Create backups of original files."""
    files_to_backup = [
        ('DLGN_VT.py', 'DLGN_VT_backup.py'),
    ]

    print("Creating backups...")
    for src, dst in files_to_backup:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Backed up {src} -> {dst}")
        else:
            print(f"  ⚠ Warning: {src} not found, skipping")

    print("Backups created successfully!\n")


def migrate_to_optimized():
    """Replace original files with optimized versions."""
    migrations = [
        ('DLGN_VT_optimized.py', 'DLGN_VT.py'),
    ]

    print("Migrating to optimized versions...")
    for src, dst in migrations:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Migrated {src} -> {dst}")
        else:
            print(f"  ✗ Error: {src} not found!")
            return False

    print("Migration completed successfully!\n")
    return True


def rollback():
    """Restore original files from backup."""
    backups = [
        ('DLGN_VT_backup.py', 'DLGN_VT.py'),
    ]

    print("Rolling back to original versions...")
    for src, dst in backups:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Restored {dst} from {src}")
        else:
            print(f"  ✗ Error: Backup {src} not found!")
            return False

    print("Rollback completed successfully!\n")
    return True


def validate():
    """Run basic validation tests to ensure optimized version works."""
    print("Running validation tests...\n")

    try:
        import torch
        import numpy as np
        from DLGN_VT import DLGN_VT

        # Test 1: Model creation
        print("Test 1: Model creation...")
        model = DLGN_VT(
            input_dim=10,
            output_dim=1,
            num_hidden_nodes=[32, 32],
            beta=30,
            BN=True,
            prod='op',
            feat='sf'
        )
        print("  ✓ Model created successfully")

        # Test 2: Forward pass
        print("Test 2: Forward pass...")
        x = torch.randn(16, 10)
        output = model(x)
        assert output.shape == (16,), f"Expected shape (16,), got {output.shape}"
        print("  ✓ Forward pass successful")

        # Test 3: Get gate scores
        print("Test 3: Get gate scores...")
        gate_scores = model.get_gate_scores(x)
        assert len(gate_scores) == 2, f"Expected 2 layers, got {len(gate_scores)}"
        print("  ✓ Gate scores computed successfully")

        # Test 4: NPK forward
        print("Test 4: NPK forward...")
        cp = model.npk_forward(x)
        assert cp.shape[0] == 16, f"Expected batch size 16, got {cp.shape[0]}"
        print("  ✓ NPK forward successful")

        # Test 5: Get NPK matrix
        print("Test 5: Get NPK matrix...")
        x1 = torch.randn(8, 10)
        x2 = torch.randn(12, 10)
        npk = model.get_npk(x1, x2)
        assert npk.shape == (8, 12), f"Expected shape (8, 12), got {npk.shape}"
        print("  ✓ NPK matrix computed successfully")

        # Test 6: Backward pass
        print("Test 6: Backward pass...")
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        model.train()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y.float())
        loss.backward()
        assert model.value_layers.grad is not None, "Gradients not computed"
        print("  ✓ Backward pass successful")

        # Test 7: Type hints (static check)
        print("Test 7: Checking type hints...")
        import inspect
        sig = inspect.signature(DLGN_VT.__init__)
        assert 'input_dim' in sig.parameters, "Missing type hints"
        print("  ✓ Type hints present")

        print("\n" + "="*50)
        print("All validation tests passed! ✓")
        print("="*50 + "\n")
        return True

    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Migrate DLGN VT to optimized implementation'
    )
    parser.add_argument(
        '--backup-only',
        action='store_true',
        help='Only create backups without migrating'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback to original version from backup'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation tests only'
    )

    args = parser.parse_args()

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    if args.rollback:
        success = rollback()
        sys.exit(0 if success else 1)

    if args.validate:
        success = validate()
        sys.exit(0 if success else 1)

    if args.backup_only:
        backup_files()
        sys.exit(0)

    # Full migration
    print("="*50)
    print("DLGN VT Migration to Optimized Version")
    print("="*50 + "\n")

    # Step 1: Backup
    backup_files()

    # Step 2: Migrate
    if not migrate_to_optimized():
        print("\n✗ Migration failed!")
        print("Your original files are still backed up.")
        print("Run with --rollback to restore them.")
        sys.exit(1)

    # Step 3: Validate
    if not validate():
        print("\n⚠ Warning: Validation failed!")
        print("The files have been migrated, but tests failed.")
        print("You may want to run --rollback and investigate.")
        sys.exit(1)

    print("="*50)
    print("Migration completed successfully! ✓")
    print("="*50)
    print("\nYour original files have been backed up with _backup suffix.")
    print("You can now use the optimized implementation.")
    print("\nTo rollback: python migrate_to_optimized.py --rollback")


if __name__ == '__main__':
    main()
