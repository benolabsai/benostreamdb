#!/usr/bin/env python3
"""Test that all Python functions have comprehensive docstrings."""

import sys
import os

# Add the built library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target', 'release'))

import hyperstreamdb as hdb
import inspect

def check_docstring(func_name, func):
    """Check if a function has a comprehensive docstring."""
    doc = func.__doc__
    if not doc:
        return False, "No docstring"
    
    # Check for required sections
    required_sections = ['Args:', 'Returns:', 'Example:']
    missing = []
    for section in required_sections:
        if section not in doc:
            missing.append(section)
    
    if missing:
        return False, f"Missing sections: {', '.join(missing)}"
    
    # Check for Raises section (should be present for most functions)
    if 'Raises:' not in doc and func_name not in ['auto_detect', 'list_available_backends', 
                                                     'get_stats', 'reset_stats', '__repr__']:
        return False, "Missing Raises: section"
    
    return True, "Complete"

def main():
    """Test all public functions and classes."""
    print("Testing Python API docstrings...\n")
    
    # Test distance functions
    distance_functions = [
        'l2', 'cosine', 'inner_product', 'l1', 'hamming', 'jaccard',
        'l2_batch', 'cosine_batch', 'inner_product_batch', 
        'l1_batch', 'hamming_batch', 'jaccard_batch',
        'l2_sparse', 'cosine_sparse', 'inner_product_sparse',
        'hamming_packed', 'jaccard_packed', 'hamming_auto', 'jaccard_auto'
    ]
    
    print("Distance Functions:")
    print("-" * 80)
    all_passed = True
    for func_name in distance_functions:
        if hasattr(hdb, func_name):
            func = getattr(hdb, func_name)
            passed, msg = check_docstring(func_name, func)
            status = "✓" if passed else "✗"
            print(f"{status} {func_name:30s} {msg}")
            if not passed:
                all_passed = False
                print(f"  Docstring preview: {func.__doc__[:100] if func.__doc__ else 'None'}...")
        else:
            print(f"✗ {func_name:30s} Not found")
            all_passed = False
    
    # Test ComputeContext class
    print("\nComputeContext Class:")
    print("-" * 80)
    if hasattr(hdb, 'ComputeContext'):
        ctx_class = hdb.ComputeContext
        
        # Check class docstring
        if ctx_class.__doc__:
            print(f"✓ ComputeContext class docstring exists")
        else:
            print(f"✗ ComputeContext class docstring missing")
            all_passed = False
        
        # Check methods
        methods = ['auto_detect', 'list_available_backends', 'get_stats', 'reset_stats']
        for method_name in methods:
            if hasattr(ctx_class, method_name):
                method = getattr(ctx_class, method_name)
                passed, msg = check_docstring(method_name, method)
                status = "✓" if passed else "✗"
                print(f"{status} {method_name:30s} {msg}")
                if not passed:
                    all_passed = False
            else:
                print(f"✗ {method_name:30s} Not found")
                all_passed = False
    else:
        print("✗ ComputeContext class not found")
        all_passed = False
    
    # Test SparseVector class
    print("\nSparseVector Class:")
    print("-" * 80)
    if hasattr(hdb, 'SparseVector'):
        sv_class = hdb.SparseVector
        
        # Check class docstring
        if sv_class.__doc__:
            print(f"✓ SparseVector class docstring exists")
        else:
            print(f"✗ SparseVector class docstring missing")
            all_passed = False
        
        # Check methods
        if hasattr(sv_class, 'to_dense'):
            method = sv_class.to_dense
            passed, msg = check_docstring('to_dense', method)
            status = "✓" if passed else "✗"
            print(f"{status} to_dense{' ':23s} {msg}")
            if not passed:
                all_passed = False
    else:
        print("✗ SparseVector class not found")
        all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All docstrings are comprehensive!")
        return 0
    else:
        print("✗ Some docstrings need improvement")
        return 1

if __name__ == '__main__':
    sys.exit(main())
