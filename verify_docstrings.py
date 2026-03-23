#!/usr/bin/env python3
"""
Verify that all Python functions have proper docstrings following NumPy style.
"""

import sys
import inspect

def check_docstring(func_name, func):
    """Check if a function has a proper docstring."""
    doc = func.__doc__
    if not doc:
        print(f"❌ {func_name}: Missing docstring")
        return False
    
    # Check for key sections
    has_args = "Args:" in doc or "Parameters:" in doc
    has_returns = "Returns:" in doc
    has_example = "Example:" in doc or "Examples:" in doc
    
    issues = []
    if not has_args and "(" in str(inspect.signature(func)):
        issues.append("missing Args/Parameters section")
    if not has_returns:
        issues.append("missing Returns section")
    if not has_example:
        issues.append("missing Example section")
    
    if issues:
        print(f"⚠️  {func_name}: {', '.join(issues)}")
        return False
    else:
        print(f"✅ {func_name}: Complete docstring")
        return True

def main():
    try:
        import hyperstreamdb as hdb
    except ImportError:
        print("❌ Cannot import hyperstreamdb. Make sure it's built and installed.")
        sys.exit(1)
    
    print("Checking docstrings for all Python functions...\n")
    
    # List of all functions to check
    functions = [
        # Single-pair distance functions
        ("l2", hdb.l2),
        ("cosine", hdb.cosine),
        ("inner_product", hdb.inner_product),
        ("l1", hdb.l1),
        ("hamming", hdb.hamming),
        ("jaccard", hdb.jaccard),
        
        # Batch distance functions
        ("l2_batch", hdb.l2_batch),
        ("cosine_batch", hdb.cosine_batch),
        ("inner_product_batch", hdb.inner_product_batch),
        ("l1_batch", hdb.l1_batch),
        ("hamming_batch", hdb.hamming_batch),
        ("jaccard_batch", hdb.jaccard_batch),
        
        # Sparse distance functions
        ("l2_sparse", hdb.l2_sparse),
        ("cosine_sparse", hdb.cosine_sparse),
        ("inner_product_sparse", hdb.inner_product_sparse),
        
        # Binary distance functions
        ("hamming_packed", hdb.hamming_packed),
        ("hamming_auto", hdb.hamming_auto),
        ("jaccard_packed", hdb.jaccard_packed),
        ("jaccard_auto", hdb.jaccard_auto),
    ]
    
    classes = [
        ("SparseVector", hdb.SparseVector),
        ("ComputeContext", hdb.ComputeContext),
    ]
    
    all_good = True
    
    print("=== Functions ===")
    for name, func in functions:
        if not check_docstring(name, func):
            all_good = False
    
    print("\n=== Classes ===")
    for name, cls in classes:
        print(f"\n{name}:")
        if not check_docstring(f"{name}.__init__", cls.__init__):
            all_good = False
        
        # Check class methods
        for method_name in dir(cls):
            if not method_name.startswith('_') or method_name in ['__init__', '__repr__']:
                method = getattr(cls, method_name)
                if callable(method):
                    check_docstring(f"{name}.{method_name}", method)
    
    print("\n" + "="*50)
    if all_good:
        print("✅ All docstrings are complete!")
        sys.exit(0)
    else:
        print("⚠️  Some docstrings need improvement")
        sys.exit(1)

if __name__ == "__main__":
    main()
