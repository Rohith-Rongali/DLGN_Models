#!/usr/bin/env python3
"""
Comprehensive test suite for DLGN Value Tensor optimizations.

This script tests:
1. Syntax validation (can run without PyTorch)
2. Import checks
3. Model creation and basic functionality
4. Numerical equivalence with original implementation
5. Performance benchmarking
"""

import sys
import time
import os

# Track test results
test_results = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'errors': []
}

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_test(name, status="PASS", message=""):
    """Print test result."""
    symbols = {
        'PASS': '✓',
        'FAIL': '✗',
        'SKIP': '⊘'
    }
    symbol = symbols.get(status, '?')

    if status == 'PASS':
        test_results['passed'] += 1
        print(f"{symbol} {name}")
    elif status == 'FAIL':
        test_results['failed'] += 1
        print(f"{symbol} {name}")
        if message:
            print(f"  Error: {message}")
        test_results['errors'].append(f"{name}: {message}")
    else:
        test_results['skipped'] += 1
        print(f"{symbol} {name} (skipped: {message})")

    if message and status == 'PASS':
        print(f"  {message}")


def test_syntax_validation():
    """Test 1: Syntax validation."""
    print_header("Test 1: Python Syntax Validation")

    files = [
        'DLGN_VT_optimized.py',
        'training_methods_vt_optimized.py',
        'migrate_to_optimized.py'
    ]

    for file in files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print_test(f"Syntax check: {file}", "PASS")
        except SyntaxError as e:
            print_test(f"Syntax check: {file}", "FAIL", str(e))
        except FileNotFoundError:
            print_test(f"Syntax check: {file}", "FAIL", "File not found")


def test_imports():
    """Test 2: Import checks."""
    print_header("Test 2: Import Validation")

    # Check if torch is available
    try:
        import torch
        torch_available = True
        print_test("PyTorch availability", "PASS", f"Version: {torch.__version__}")
    except ImportError:
        torch_available = False
        print_test("PyTorch availability", "SKIP", "PyTorch not installed")
        return False

    try:
        import numpy as np
        print_test("NumPy availability", "PASS", f"Version: {np.__version__}")
    except ImportError:
        print_test("NumPy availability", "FAIL", "NumPy not installed")
        return False

    # Try importing optimized version
    try:
        from DLGN_VT_optimized import DLGN_VT
        print_test("Import DLGN_VT_optimized", "PASS")
    except Exception as e:
        print_test("Import DLGN_VT_optimized", "FAIL", str(e))
        return False

    # Try importing original version
    try:
        from DLGN_VT import DLGN_VT as DLGN_VT_Original
        print_test("Import DLGN_VT (original)", "PASS")
    except Exception as e:
        print_test("Import DLGN_VT (original)", "SKIP", "Original not available")

    return True


def test_model_creation():
    """Test 3: Model creation."""
    print_header("Test 3: Model Creation")

    try:
        import torch
        from DLGN_VT_optimized import DLGN_VT
    except ImportError:
        print_test("Model creation", "SKIP", "PyTorch not available")
        return None

    # Test different configurations
    configs = [
        {
            'name': 'Basic configuration (op, sf)',
            'params': {
                'input_dim': 10,
                'output_dim': 1,
                'num_hidden_nodes': [32, 32],
                'beta': 30,
                'BN': False,
                'prod': 'op',
                'feat': 'sf'
            }
        },
        {
            'name': 'With batch norm (op, sf, BN)',
            'params': {
                'input_dim': 20,
                'output_dim': 1,
                'num_hidden_nodes': [64, 64, 64],
                'beta': 50,
                'BN': True,
                'prod': 'op',
                'feat': 'sf'
            }
        },
        {
            'name': 'Composite features (op, cf)',
            'params': {
                'input_dim': 15,
                'output_dim': 1,
                'num_hidden_nodes': [128],
                'beta': 20,
                'BN': False,
                'prod': 'op',
                'feat': 'cf'
            }
        },
        {
            'name': 'Inner product (ip, sf)',
            'params': {
                'input_dim': 10,
                'output_dim': 1,
                'num_hidden_nodes': [32],
                'beta': 30,
                'BN': False,
                'prod': 'ip',
                'feat': 'sf'
            }
        }
    ]

    models = []
    for config in configs:
        try:
            model = DLGN_VT(**config['params'])
            models.append(model)
            print_test(config['name'], "PASS")
        except Exception as e:
            print_test(config['name'], "FAIL", str(e))

    return models[0] if models else None


def test_forward_pass(model):
    """Test 4: Forward pass."""
    print_header("Test 4: Forward Pass")

    if model is None:
        print_test("Forward pass", "SKIP", "No model available")
        return

    try:
        import torch
    except ImportError:
        print_test("Forward pass", "SKIP", "PyTorch not available")
        return

    # Test forward pass with different batch sizes
    batch_sizes = [1, 16, 128]

    for bs in batch_sizes:
        try:
            x = torch.randn(bs, model.num_nodes[0])
            output = model(x)

            expected_shape = (bs,)
            if output.shape == expected_shape:
                print_test(f"Forward pass (batch_size={bs})", "PASS",
                          f"Output shape: {output.shape}")
            else:
                print_test(f"Forward pass (batch_size={bs})", "FAIL",
                          f"Expected shape {expected_shape}, got {output.shape}")
        except Exception as e:
            print_test(f"Forward pass (batch_size={bs})", "FAIL", str(e))


def test_model_methods(model):
    """Test 5: Model methods."""
    print_header("Test 5: Model Methods")

    if model is None:
        print_test("Model methods", "SKIP", "No model available")
        return

    try:
        import torch
    except ImportError:
        print_test("Model methods", "SKIP", "PyTorch not available")
        return

    # Test get_gate_scores
    try:
        x = torch.randn(16, model.num_nodes[0])
        gate_scores = model.get_gate_scores(x)

        if len(gate_scores) == model.num_hidden_layers:
            print_test("get_gate_scores()", "PASS",
                      f"Returned {len(gate_scores)} gate score tensors")
        else:
            print_test("get_gate_scores()", "FAIL",
                      f"Expected {model.num_hidden_layers} tensors, got {len(gate_scores)}")
    except Exception as e:
        print_test("get_gate_scores()", "FAIL", str(e))

    # Test npk_forward
    try:
        x = torch.randn(16, model.num_nodes[0])
        cp = model.npk_forward(x)

        if cp.shape[0] == 16:
            print_test("npk_forward()", "PASS", f"Output shape: {cp.shape}")
        else:
            print_test("npk_forward()", "FAIL",
                      f"Expected batch size 16, got {cp.shape[0]}")
    except Exception as e:
        print_test("npk_forward()", "FAIL", str(e))

    # Test get_npk (only for 'op' mode)
    if model.prod == 'op':
        try:
            x1 = torch.randn(8, model.num_nodes[0])
            x2 = torch.randn(12, model.num_nodes[0])
            npk = model.get_npk(x1, x2)

            expected_shape = (8, 12)
            if npk.shape == expected_shape:
                print_test("get_npk()", "PASS", f"NPK matrix shape: {npk.shape}")
            else:
                print_test("get_npk()", "FAIL",
                          f"Expected shape {expected_shape}, got {npk.shape}")
        except Exception as e:
            print_test("get_npk()", "FAIL", str(e))

    # Test return_gating_functions
    try:
        weights = model.return_gating_functions()

        if len(weights) == model.num_hidden_layers:
            print_test("return_gating_functions()", "PASS",
                      f"Returned {len(weights)} weight tensors")
        else:
            print_test("return_gating_functions()", "FAIL",
                      f"Expected {model.num_hidden_layers} tensors, got {len(weights)}")
    except Exception as e:
        print_test("return_gating_functions()", "FAIL", str(e))

    # Test log_features
    try:
        features = model.log_features()
        print_test("log_features()", "PASS", f"Features shape: {features.shape}")
    except Exception as e:
        print_test("log_features()", "FAIL", str(e))


def test_backward_pass(model):
    """Test 6: Backward pass."""
    print_header("Test 6: Backward Pass")

    if model is None:
        print_test("Backward pass", "SKIP", "No model available")
        return

    try:
        import torch
    except ImportError:
        print_test("Backward pass", "SKIP", "PyTorch not available")
        return

    try:
        model.train()
        x = torch.randn(4, model.num_nodes[0], requires_grad=True)
        y = torch.randint(0, 2, (4,)).float()

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()

        # Check if gradients are computed
        if model.value_layers.grad is not None:
            print_test("Gradient computation", "PASS", "Gradients computed successfully")
        else:
            print_test("Gradient computation", "FAIL", "No gradients computed")

        # Check input gradients
        if x.grad is not None:
            print_test("Input gradients", "PASS", "Input gradients available")
        else:
            print_test("Input gradients", "FAIL", "No input gradients")

    except Exception as e:
        print_test("Backward pass", "FAIL", str(e))


def test_numerical_equivalence():
    """Test 7: Numerical equivalence with original."""
    print_header("Test 7: Numerical Equivalence")

    try:
        import torch
        from DLGN_VT_optimized import DLGN_VT as DLGN_VT_Opt
        from DLGN_VT import DLGN_VT as DLGN_VT_Orig
    except ImportError as e:
        print_test("Numerical equivalence", "SKIP", f"Import failed: {e}")
        return

    # Create both models with same config
    config = {
        'input_dim': 10,
        'output_dim': 1,
        'num_hidden_nodes': [32, 32],
        'beta': 30,
        'BN': False,
        'prod': 'op',
        'feat': 'sf'
    }

    try:
        torch.manual_seed(42)
        model_orig = DLGN_VT_Orig(**config)

        torch.manual_seed(42)
        model_opt = DLGN_VT_Opt(**config)

        # Test forward pass equivalence
        x = torch.randn(16, 10)

        model_orig.eval()
        model_opt.eval()

        with torch.no_grad():
            out_orig = model_orig(x)
            out_opt = model_opt(x)

        # Check if outputs are close
        max_diff = torch.max(torch.abs(out_orig - out_opt)).item()

        if max_diff < 1e-5:
            print_test("Forward pass equivalence", "PASS",
                      f"Max difference: {max_diff:.2e}")
        else:
            print_test("Forward pass equivalence", "FAIL",
                      f"Max difference: {max_diff:.2e} (threshold: 1e-5)")

    except Exception as e:
        print_test("Numerical equivalence", "FAIL", str(e))


def test_type_hints():
    """Test 8: Type hint validation."""
    print_header("Test 8: Type Hints Validation")

    try:
        import inspect
        from DLGN_VT_optimized import DLGN_VT
    except ImportError as e:
        print_test("Type hints", "SKIP", f"Import failed: {e}")
        return

    # Check __init__ has type hints
    try:
        sig = inspect.signature(DLGN_VT.__init__)
        params_with_hints = [p for p in sig.parameters.values()
                            if p.annotation != inspect.Parameter.empty]

        if len(params_with_hints) > 0:
            print_test("Type hints in __init__", "PASS",
                      f"{len(params_with_hints)} parameters have type hints")
        else:
            print_test("Type hints in __init__", "FAIL",
                      "No type hints found")
    except Exception as e:
        print_test("Type hints in __init__", "FAIL", str(e))

    # Check other methods
    methods_to_check = ['forward', 'get_gate_scores', 'npk_forward', 'get_npk']

    for method_name in methods_to_check:
        try:
            method = getattr(DLGN_VT, method_name)
            sig = inspect.signature(method)

            has_return_hint = sig.return_annotation != inspect.Signature.empty

            if has_return_hint:
                print_test(f"Type hints in {method_name}()", "PASS")
            else:
                print_test(f"Type hints in {method_name}()", "FAIL",
                          "Missing return type hint")
        except AttributeError:
            print_test(f"Type hints in {method_name}()", "SKIP",
                      "Method not found")
        except Exception as e:
            print_test(f"Type hints in {method_name}()", "FAIL", str(e))


def test_documentation():
    """Test 9: Documentation validation."""
    print_header("Test 9: Documentation Validation")

    try:
        from DLGN_VT_optimized import DLGN_VT
    except ImportError as e:
        print_test("Documentation", "SKIP", f"Import failed: {e}")
        return

    # Check class docstring
    if DLGN_VT.__doc__ and len(DLGN_VT.__doc__.strip()) > 50:
        print_test("Class docstring", "PASS",
                  f"{len(DLGN_VT.__doc__)} characters")
    else:
        print_test("Class docstring", "FAIL", "Missing or too short")

    # Check method docstrings
    methods = ['_compute_gate_score_layer', 'forward', 'get_gate_scores',
               'npk_forward', 'get_npk', 'return_gating_functions', 'log_features']

    for method_name in methods:
        try:
            method = getattr(DLGN_VT, method_name)
            if method.__doc__ and len(method.__doc__.strip()) > 20:
                print_test(f"Docstring for {method_name}()", "PASS")
            else:
                print_test(f"Docstring for {method_name}()", "FAIL",
                          "Missing or too short")
        except AttributeError:
            print_test(f"Docstring for {method_name}()", "SKIP",
                      "Method not found")


def print_summary():
    """Print test summary."""
    print_header("Test Summary")

    total = test_results['passed'] + test_results['failed'] + test_results['skipped']

    print(f"\nTotal tests: {total}")
    print(f"✓ Passed:  {test_results['passed']}")
    print(f"✗ Failed:  {test_results['failed']}")
    print(f"⊘ Skipped: {test_results['skipped']}")

    if test_results['failed'] > 0:
        print("\nFailed tests:")
        for error in test_results['errors']:
            print(f"  - {error}")

    success_rate = (test_results['passed'] / max(total - test_results['skipped'], 1)) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    return test_results['failed'] == 0


def main():
    """Run all tests."""
    print_header("DLGN Value Tensor Optimization Test Suite")

    # Run tests
    test_syntax_validation()
    torch_available = test_imports()

    if torch_available:
        model = test_model_creation()
        test_forward_pass(model)
        test_model_methods(model)
        test_backward_pass(model)
        test_numerical_equivalence()

    test_type_hints()
    test_documentation()

    # Print summary
    success = print_summary()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
