import os
import signal
import subprocess
import time
import requests
import tempfile
import shutil

def wait_for_healthy(port=9200, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/_cluster/health", headers={"Authorization": "Bearer test-key"})
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False

def test_process_kill_recovery():
    print("="*60)
    print("Testing External Crash Recovery (SIGKILL)")
    print("="*60)

    # Setup unique data directory
    ts = int(time.time())
    data_dir = os.path.join(tempfile.gettempdir(), f"bsdb_crash_data_{ts}")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    env = os.environ.copy()
    env["BSDB_SEARCH_STORAGE_URI"] = f"file://{data_dir}"
    env["BSDB_SEARCH_BIND"] = "127.0.0.1"
    env["BSDB_SEARCH_PORT"] = "9200"
    env["BSDB_QDRANT_BIND"] = "127.0.0.1"
    env["BSDB_QDRANT_PORT"] = "6333"
    env["BSDB_AUTH_REQUIRED"] = "1"
    env["BSDB_API_KEY"] = "test-key"

    # Start the server
    print("Starting bsdb-search...")
    # Assuming bsdb-search is built in release mode
    bin_path = "../../target/release/bsdb-search"
    if not os.path.exists(bin_path):
        # Fallback to cargo run
        proc = subprocess.Popen(["cargo", "run", "--release", "-p", "benostreamdb-search"], env=env)
    else:
        proc = subprocess.Popen([bin_path], env=env)

    try:
        assert wait_for_healthy(), "Server did not become healthy"
        
        headers = {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }

        print("Writing documents...")
        # Write 5 documents
        for i in range(5):
            r = requests.post(f"http://127.0.0.1:9200/crash_index/_doc/{i}", json={"text": f"Document {i}"}, headers=headers)
            assert r.status_code in (200, 201), f"Failed to write document {i}: {r.text}"
            
        # Refresh to make sure it's searchable
        requests.post("http://127.0.0.1:9200/crash_index/_refresh", headers=headers)

        # Force flush to disk (optional, but guarantees WAL/buffer goes to OS)
        # Actually we just wait a bit
        time.sleep(2)

        print("Sending SIGKILL...")
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        print("Process killed.")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # Restart the server
    print("Restarting bsdb-search...")
    if not os.path.exists(bin_path):
        proc = subprocess.Popen(["cargo", "run", "--release", "-p", "benostreamdb-search"], env=env)
    else:
        proc = subprocess.Popen([bin_path], env=env)

    try:
        assert wait_for_healthy(), "Server did not recover and become healthy"
        
        headers = {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }

        print("Verifying recovered data...")
        # Give it a moment to recover index
        time.sleep(2)
        
        r = requests.post("http://127.0.0.1:9200/crash_index/_search", json={"query": {"match_all": {}}}, headers=headers)
        assert r.status_code == 200, f"Search failed: {r.text}"
        hits = r.json().get("hits", {}).get("hits", [])
        
        assert len(hits) == 5, f"Expected 5 documents, found {len(hits)}"
        print(f"✓ Recovered {len(hits)} documents successfully after SIGKILL")
        print("\nPASSED: Process Kill Recovery Test")
    finally:
        proc.kill()
        proc.wait()
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)

if __name__ == "__main__":
    test_process_kill_recovery()
