import hyperstreamdb as hs
import numpy as np
import pandas as pd
import os
import shutil
import time

def test_cascading_device_assignment():
    uri = "file://" + os.path.abspath("test_cascading_db")
    if os.path.exists("test_cascading_db"):
        shutil.rmtree("test_cascading_db")
    
    table = hs.open_table(uri)
    
    # Define schema with three vector columns
    dim = 64
    num_rows = 2000 
    
    # 1. Set Table-level default to CPU
    print("--- Setting default_device to 'cpu' ---")
    table.set_default_device("cpu")
    
    # 2. Add columns:
    # col_default -> Should use CPU
    # col_gpu_override -> Should use MPS
    # col_another_default -> Should use CPU
    
    print("--- Adding col_default (no explicit device) ---")
    table.add_index_columns(["col_default"])
    
    print("--- Adding col_gpu_override with device='mps' ---")
    table.add_index_columns(["col_gpu_override"], device="mps")
    
    print("--- Adding col_another_default (no explicit device) ---")
    table.add_index_columns(["col_another_default"])
    
    # 3. Ingest data
    data = {
        "col_default": np.random.randn(num_rows, dim).astype(np.float32).tolist(),
        "col_gpu_override": np.random.randn(num_rows, dim).astype(np.float32).tolist(),
        "col_another_default": np.random.randn(num_rows, dim).astype(np.float32).tolist(),
        "id": list(range(num_rows))
    }
    
    print(f"Ingesting {num_rows} rows...")
    df = pd.DataFrame(data)
    table.write(df)
    
    print("Flushing...")
    table.flush()
    table.wait_for_background_tasks()
    
    print("Verification complete. Check logs for 'Applying default device' vs 'Applying device override'.")

if __name__ == "__main__":
    test_cascading_device_assignment()
