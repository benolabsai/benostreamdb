
import hyperstreamdb
import pyarrow
import time
import sys
import os

# Ensure verification binary output is flushed
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# URI pointing to the mock Iceberg REST table served by iceberg_rest
# The setup is provided by verify_iceberg_python_delete.sh
URI = "file:///tmp/hdb_test_delete/default/test_delete_table"

def verify_iceberg_read():
    print(f"Connecting to HyperStreamDB Table at {URI}...")
    try:
        # Create Table instance
        table = hyperstreamdb.Table(URI)
        print("Table connected successfully.")
        
        # Read to PyArrow Table
        print("Reading table data...")
        df = table.to_arrow()
        
        row_count = df.num_rows
        print(f"Total rows read: {row_count}")
        
        # Convert to Pandas for easier value checking if available, otherwise iterate arrow
        try:
            import pandas as pd
            pdf = df.to_pandas()
            print("Converted to Pandas DataFrame.")
            
            # Check for deleted rows
            # row_0 (position delete)
            # row_5 (position delete)
            # row_10 (equality delete, category='row_10')
            
            # Assuming 'category' column exists and holds "row_X" strings based on generate_iceberg_manifests.rs
            # The generator creates schema: [category: string, value: float]
            
            categories = pdf['category'].tolist()
            
            if "row_0" in categories:
                print("FAILURE: row_0 found! (Should be deleted by Position Delete)")
                sys.exit(1)
            
            if "row_5" in categories:
                print("FAILURE: row_5 found! (Should be deleted by Position Delete)")
                sys.exit(1)
                
            if "row_10" in categories:
                print("FAILURE: row_10 found! (Should be deleted by Equality Delete)")
                sys.exit(1)
                
            if row_count != 97:
                print(f"FAILURE: Expected 97 rows, got {row_count}")
                sys.exit(1)
                
            print("SUCCESS: Verification Passed! Deleted rows are gone.")
            
        except ImportError:
            print("Pandas not found, checking Arrow table directly.")
            # Basic check if pandas missing
            cat_col = df['category']
            present_cats = set()
            for batch in cat_col.chunks:
                for val in batch:
                    present_cats.add(str(val))
            
            if "row_10" in present_cats:
                 print("FAILURE: row_10 found!")
                 sys.exit(1)
            if row_count != 97:
                 print(f"FAILURE: Expected 97 rows, got {row_count}")
                 sys.exit(1)
            print("SUCCESS: Verification Passed! Deleted rows are gone.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_iceberg_read()
