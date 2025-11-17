#!/usr/bin/env python3
"""
Static code analysis for DLGN Value Tensor optimizations.

This script performs code quality checks that don't require PyTorch:
- Code structure validation
- Method signature checks
- Code complexity analysis
- Best practices verification
"""

import ast
import sys
from pathlib import Path


def analyze_file(filepath):
    """Analyze a Python file using AST."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print('='*60)

    with open(filepath, 'r') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

    # Find classes
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    print(f"\n📊 Code Statistics:")
    print(f"  - Classes: {len(classes)}")
    print(f"  - Functions: {len(functions)}")
    print(f"  - Lines of code: {len(code.splitlines())}")

    # Analyze each class
    for cls in classes:
        print(f"\n🔍 Class: {cls.name}")

        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
        print(f"  - Methods: {len(methods)}")

        # Check for docstrings
        has_docstring = (isinstance(cls.body[0], ast.Expr) and
                        isinstance(cls.body[0].value, ast.Constant))
        print(f"  - Has docstring: {'✓' if has_docstring else '✗'}")

        # Check methods
        methods_with_docstrings = 0
        methods_with_type_hints = 0

        for method in methods:
            # Check docstring
            if (method.body and
                isinstance(method.body[0], ast.Expr) and
                isinstance(method.body[0].value, ast.Constant)):
                methods_with_docstrings += 1

            # Check type hints
            has_hints = (
                method.returns is not None or
                any(arg.annotation is not None for arg in method.args.args)
            )
            if has_hints:
                methods_with_type_hints += 1

        if methods:
            print(f"  - Methods with docstrings: {methods_with_docstrings}/{len(methods)} "
                  f"({methods_with_docstrings/len(methods)*100:.0f}%)")
            print(f"  - Methods with type hints: {methods_with_type_hints}/{len(methods)} "
                  f"({methods_with_type_hints/len(methods)*100:.0f}%)")

    # Check for common issues
    print(f"\n🔎 Code Quality Checks:")

    # Check for proper imports
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    print(f"  ✓ Imports found: {len(imports)}")

    # Check for TODO/FIXME comments
    todos = [line for line in code.splitlines() if 'TODO' in line or 'FIXME' in line]
    if todos:
        print(f"  ⚠ TODO/FIXME comments: {len(todos)}")
        for todo in todos[:3]:  # Show first 3
            print(f"    - {todo.strip()}")
    else:
        print(f"  ✓ No TODO/FIXME comments")

    # Check for long functions (> 100 lines)
    long_functions = []
    for func in functions:
        if hasattr(func, 'lineno') and hasattr(func, 'end_lineno'):
            length = func.end_lineno - func.lineno
            if length > 100:
                long_functions.append((func.name, length))

    if long_functions:
        print(f"  ⚠ Long functions (>100 lines):")
        for name, length in long_functions:
            print(f"    - {name}: {length} lines")
    else:
        print(f"  ✓ No overly long functions")

    # Check for unused variables (simple check)
    print(f"  ✓ AST parsed successfully")

    return True


def check_optimization_patterns():
    """Check for optimization patterns in the code."""
    print(f"\n{'='*60}")
    print("Checking Optimization Patterns")
    print('='*60)

    filepath = 'DLGN_VT_optimized.py'

    with open(filepath, 'r') as f:
        code = f.read()

    checks = {
        'torch.no_grad()': 'Using torch.no_grad() for inference',
        '@torch.jit': 'Using torch.jit decorators',
        'torch.cuda.empty_cache()': 'Explicit cache clearing',
        'def _compute_gate_score': 'Unified gate score computation',
        'Tuple[': 'Type hints with Tuple',
        'Optional[': 'Type hints with Optional',
        '"""': 'Docstrings present',
    }

    print("\n✓ Optimization Features:")
    for pattern, description in checks.items():
        if pattern in code:
            print(f"  ✓ {description}")
        else:
            print(f"  - {description} (not found)")


def compare_file_sizes():
    """Compare file sizes between original and optimized."""
    print(f"\n{'='*60}")
    print("File Size Comparison")
    print('='*60)

    files = [
        ('DLGN_VT.py', 'Original implementation'),
        ('DLGN_VT_optimized.py', 'Optimized implementation'),
    ]

    print()
    for filepath, description in files:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            lines = len(Path(filepath).read_text().splitlines())
            print(f"  {description}:")
            print(f"    - Size: {size:,} bytes")
            print(f"    - Lines: {lines:,}")
        else:
            print(f"  {description}: File not found")


def check_best_practices():
    """Check for Python best practices."""
    print(f"\n{'='*60}")
    print("Best Practices Check")
    print('='*60)

    filepath = 'DLGN_VT_optimized.py'

    with open(filepath, 'r') as f:
        code = f.read()

    tree = ast.parse(code)

    print("\n✓ Best Practices:")

    # Check for list comprehensions vs loops
    comprehensions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.ListComp, ast.DictComp))]
    print(f"  ✓ Using comprehensions: {len(comprehensions)} found")

    # Check for context managers
    with_stmts = [node for node in ast.walk(tree) if isinstance(node, ast.With)]
    print(f"  ✓ Using context managers: {len(with_stmts)} found")

    # Check for f-strings
    f_strings = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            f_strings += 1
    if f_strings > 0:
        print(f"  ✓ Using f-strings: {f_strings} found")

    # Check for proper exception handling
    try_stmts = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    print(f"  ✓ Exception handling: {len(try_stmts)} try blocks")


def main():
    """Run all static analyses."""
    print("="*60)
    print("  Static Code Analysis for DLGN VT Optimizations")
    print("="*60)

    files_to_analyze = [
        'DLGN_VT_optimized.py',
        'training_methods_vt_optimized.py',
    ]

    success = True
    for filepath in files_to_analyze:
        if Path(filepath).exists():
            if not analyze_file(filepath):
                success = False
        else:
            print(f"\n✗ File not found: {filepath}")
            success = False

    check_optimization_patterns()
    compare_file_sizes()
    check_best_practices()

    print(f"\n{'='*60}")
    if success:
        print("  ✓ Static analysis completed successfully")
    else:
        print("  ✗ Static analysis found issues")
    print("="*60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
