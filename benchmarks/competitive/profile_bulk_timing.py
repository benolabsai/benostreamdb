#!/usr/bin/env python3
"""Profile the hypersearch `_bulk` ingest path.

Launches the `hypersearch` release binary with `RUST_LOG=info`, sends a
realistic NDJSON `_bulk` payload (default 10,000 docs, 64-dim embeddings,
~2.9 KB/doc to match the competitive benchmark), and prints the per-phase
timing lines the server logs (`bulk_core timing` / `write_index_docs timing`).

Usage:
    ./venv/bin/python benchmarks/competitive/profile_bulk_timing.py [--docs 10000] [--dim 64]
"""

import argparse
import json
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = REPO_ROOT / "target" / "release" / "hypersearch"
if not BINARY.exists():
    BINARY = REPO_ROOT / "target" / "debug" / "hypersearch"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_docs(n: int, dim: int):
    """Synthetic docs sized like the wiki benchmark payload (~2.9 KB each)."""
    words = (
        "the quick brown fox jumps over the lazy dog science research data "
        "information study history world time people water energy system "
        "method result analysis model theory experiment observation value"
    ).split()
    docs = []
    for i in range(n):
        body = " ".join(random.choice(words) for _ in range(180))
        docs.append(
            {
                "_id": str(i),
                "title": f"Doc {i}",
                "body": body,
                "category": random.choice(["science", "history", "general"]),
                "price": round(random.random() * 100, 2),
                "embedding": [random.random() for _ in range(dim)],
            }
        )
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--requests", type=int, default=3, help="number of _bulk POSTs")
    args = ap.parse_args()

    if not BINARY.exists():
        print(f"binary not found: {BINARY}", file=sys.stderr)
        return 1

    port = free_port()
    tmp = tempfile.mkdtemp(prefix="hsbulkprof-")
    storage_uri = f"file://{tmp}"
    index = "prof-" + uuid.uuid4().hex[:8]

    log_path = Path(tmp) / "hypersearch.log"
    log_file = open(log_path, "wb")

    env = {
        **__import__("os").environ,
        "HYPERSEARCH_BIND": "127.0.0.1",
        "HYPERSEARCH_PORT": str(port),
        "HYPERSEARCH_STORAGE_URI": storage_uri,
        "RUST_LOG": "debug",
    }
    proc = subprocess.Popen(
        [str(BINARY)],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    base = f"http://127.0.0.1:{port}"
    try:
        # Wait for readiness.
        ready = False
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if requests.get(base + "/", timeout=2).status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                time.sleep(0.2)
        if not ready:
            print("hypersearch did not become ready", file=sys.stderr)
            return 1

        docs = make_docs(args.docs, args.dim)
        per_req = max(1, args.docs // args.requests)

        # Warm-up request (first write also pays one-time index creation).
        warm = docs[:per_req]
        payload = "".join(
            json.dumps({"index": {"_index": index, "_id": d["_id"]}}) + "\n"
            + json.dumps(d) + "\n"
            for d in warm
        )
        r = requests.post(
            base + "/_bulk", data=payload, headers={"Content-Type": "application/x-ndjson"}, timeout=300
        )
        print(f"warm-up _bulk ({len(warm)} docs): HTTP {r.status_code}")

        # Measured requests.
        for k in range(args.requests):
            chunk = docs[k * per_req : (k + 1) * per_req]
            payload = "".join(
                json.dumps({"index": {"_index": index, "_id": d["_id"]}}) + "\n"
                + json.dumps(d) + "\n"
                for d in chunk
            )
            t0 = time.time()
            r = requests.post(
                base + "/_bulk",
                data=payload,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=300,
            )
            dt = time.time() - t0
            print(
                f"bulk req {k + 1}/{args.requests}: {len(chunk)} docs, "
                f"{len(payload) / 1e6:.1f} MB, HTTP {r.status_code}, "
                f"client-side {dt * 1000:.0f} ms ({len(chunk) / dt:.0f} docs/s)"
            )

        # Refresh to flush.
        t0 = time.time()
        r = requests.post(base + f"/{index}/_refresh", timeout=300)
        print(f"refresh: HTTP {r.status_code}, {(time.time() - t0) * 1000:.0f} ms")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()

    # Print the timing lines (and, if none, the tail of the whole log for debugging).
    print("\n=== server timing lines ===")
    try:
        all_lines = log_path.read_text(errors="replace").splitlines()
        timing = [
            l
            for l in all_lines
            if "bulk_core timing" in l
            or "write_index_docs timing" in l
            or "write_with_durability_async phase timing" in l
            or "wal_task timing" in l
            or "idx_task timing" in l
            or "wal_writer write timing" in l
        ]
        if timing:
            for line in timing:
                print(line)
        else:
            print(f"(no timing lines; log has {len(all_lines)} lines, last 25:)")
            for line in all_lines[-25:]:
                print(line)
    except FileNotFoundError:
        print("(no log captured)")

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
