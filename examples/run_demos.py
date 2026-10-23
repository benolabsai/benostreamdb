import os
import time
import subprocess
import sys
import shutil

def run_notebook(path, delay_per_cell=15):
    print(f"\n{'='*60}")
    print(f"STARTING DEMO: {os.path.basename(path)}")
    print(f"{'='*60}\n")
    
    # Use jupyter nbconvert to execute the notebook
    # We could use the python API for more control, but shell is more robust if dependencies are tricky
    
    jupyter_cmd = shutil.which("jupyter") or [sys.executable, "-m", "jupyter"]
    if isinstance(jupyter_cmd, str):
        cmd = [jupyter_cmd, "nbconvert", "--to", "notebook", "--execute", "--inplace", path]
    else:
        cmd = jupyter_cmd + ["nbconvert", "--to", "notebook", "--execute", "--inplace", path]
    
    start_time = time.time()
    
    try:
        # We simulate the "going thru" by adding an artificial wait if it runs too fast
        # though nbconvert will take its time to actually execute.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while process.poll() is None:
            elapsed = time.time() - start_time
            print(f"Executing {os.path.basename(path)}... ({elapsed:.1f}s elapsed)", end='\r')
            time.sleep(5)
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"\n\nSUCCESS: {os.path.basename(path)} completed.")
        else:
            print(f"\n\nERROR executing {os.path.basename(path)}:")
            print(stderr)
            
        # Configurable pacing (defaults to 0 for instant runs, or set DEMO_PACE_SECONDS=300 for slow walkthrough)
        total_elapsed = time.time() - start_time
        target_time = int(os.environ.get("DEMO_PACE_SECONDS", 0))
        if total_elapsed < target_time:
            wait_time = target_time - total_elapsed
            print(f"Pacing: Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            
    except Exception as e:
        print(f"Failed to run {path}: {e}")

def main():
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
    notebooks = [
        "03_comprehensive_guide.ipynb",
        "01_installation_and_basics.ipynb",
        "02_rag_pipeline.ipynb"
    ]
    
    overall_start = time.time()
    
    for nb in notebooks:
        nb_path = os.path.join(demo_dir, nb)
        if os.path.exists(nb_path):
            run_notebook(nb_path)
        else:
            print(f"Warning: {nb_path} not found.")
            
    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"ALL DEMOS COMPLETED in {overall_elapsed/60:.2f} minutes.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
