#!/usr/bin/env python3
"""Download and verify Cohere-10M dataset with resumable streaming.

Downloads the 768-dimensional dense neural embedding dataset from HuggingFace
(makneeee/cohere_large_10m) with official precomputed exact ground truth:
- base.fbin: 10,000,000 vectors (768 float32, 30,720,000,008 bytes)
- queries.fbin: 1,000 queries (768 float32, 3,072,008 bytes)
- gt_100.fbin: 1,000 queries x 100 nearest neighbours (800,008 bytes)

Default storage location: /home/ralbright/data/datasets/cohere/ (13 TB HDD)
"""

import argparse
import os
import sys
import time
import urllib.request
import numpy as np

DEFAULT_DATA_DIR = "/home/ralbright/data/datasets/cohere"

FILES = {
    "base": {
        "filename": "base.fbin",
        "url": "https://huggingface.co/datasets/makneeee/cohere_large_10m/resolve/main/fbin/base.fbin",
        "bytes": 30_720_000_008,
        "count": 10_000_000,
        "dim": 768,
        "dtype": np.float32,
    },
    "queries": {
        "filename": "queries.fbin",
        "url": "https://huggingface.co/datasets/makneeee/cohere_large_10m/resolve/main/fbin/queries.fbin",
        "bytes": 3_072_008,
        "count": 1_000,
        "dim": 768,
        "dtype": np.float32,
    },
    "gt": {
        "filename": "gt_100.fbin",
        "url": "https://huggingface.co/datasets/makneeee/cohere_large_10m/resolve/main/diskann/gt_100.fbin",
        "bytes": 800_008,
        "count": 1_000,
        "dim": 100,
        "dtype": np.uint32,
    },
}


def download_resumable(url: str, dest_path: str, target_bytes: int, log_interval_mb: int = 100):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    cur_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    if cur_bytes > target_bytes:
        print(f"[{os.path.basename(dest_path)}] File size {cur_bytes:,} > target {target_bytes:,}. Truncating to target.")
        with open(dest_path, "r+b") as f:
            f.truncate(target_bytes)
        cur_bytes = target_bytes

    if cur_bytes == target_bytes:
        print(f"[{os.path.basename(dest_path)}] Already fully downloaded ({target_bytes:,} bytes).")
        return

    print(f"[{os.path.basename(dest_path)}] Downloading {target_bytes - cur_bytes:,} remaining bytes "
          f"(resuming from {cur_bytes:,} / {target_bytes:,} bytes)...")

    # Add standard User-Agent for HuggingFace CDN
    headers = {
        "Range": f"bytes={cur_bytes}-{target_bytes - 1}",
        "User-Agent": "BenoStreamDB-Cohere-Downloader/1.0",
    }
    req = urllib.request.Request(url, headers=headers)

    chunk_size = 4 * 1024 * 1024  # 4 MiB buffer
    log_interval_bytes = log_interval_mb * 1024 * 1024
    t_start = time.time()
    bytes_downloaded_this_session = 0
    bytes_since_last_log = 0

    with urllib.request.urlopen(req) as resp, open(dest_path, "ab") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            chunk_len = len(chunk)
            cur_bytes += chunk_len
            bytes_downloaded_this_session += chunk_len
            bytes_since_last_log += chunk_len

            now = time.time()
            if bytes_since_last_log >= log_interval_bytes or cur_bytes == target_bytes:
                dt_total = now - t_start
                overall_speed_mb = (bytes_downloaded_this_session / (1024 * 1024)) / max(dt_total, 0.001)
                pct = 100.0 * cur_bytes / target_bytes
                cur_gb = cur_bytes / (1024**3)
                target_gb = target_bytes / (1024**3)
                remaining_bytes = target_bytes - cur_bytes
                eta_s = remaining_bytes / max(overall_speed_mb * 1024 * 1024, 1)
                eta_m = eta_s / 60.0
                print(
                    f"[{os.path.basename(dest_path)}] {pct:5.1f}% | "
                    f"{cur_gb:6.2f} / {target_gb:6.2f} GiB | "
                    f"Speed: {overall_speed_mb:5.1f} MB/s | "
                    f"ETA: {eta_m:4.1f} min",
                    flush=True,
                )
                bytes_since_last_log = 0


def verify_file(dest_path: str, meta: dict) -> bool:
    if not os.path.exists(dest_path):
        print(f"[FAIL] {os.path.basename(dest_path)} does not exist.")
        return False
    size = os.path.getsize(dest_path)
    if size != meta["bytes"]:
        print(f"[FAIL] {os.path.basename(dest_path)} size {size:,} != expected {meta['bytes']:,}")
        return False

    with open(dest_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
    count, dim = int(header[0]), int(header[1])
    if count != meta["count"] or dim != meta["dim"]:
        print(f"[FAIL] {os.path.basename(dest_path)} header mismatch: ({count}, {dim}) != expected ({meta['count']}, {meta['dim']})")
        return False

    print(f"[OK] {os.path.basename(dest_path)} verified: {count:,} vectors/queries, dim={dim}, size={size/(1024**3):.2f} GiB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Cohere-10M dataset for BenoStreamDB")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Target directory on storage drive (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--files", choices=["all", "base", "queries", "gt"], default="all",
                        help="Which files to download (default: all)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing files on disk without downloading")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    targets = ["base", "queries", "gt"] if args.files == "all" else [args.files]

    print("=" * 80)
    print("COHERE-10M DATASET DOWNLOADER & VERIFIER")
    print(f"Target Directory: {args.data_dir}")
    print(f"Components:       {targets}")
    print("=" * 80 + "\n", flush=True)

    if not args.verify_only:
        for key in targets:
            info = FILES[key]
            dest = os.path.join(args.data_dir, info["filename"])
            download_resumable(info["url"], dest, info["bytes"])

    print("\n--- Verifying Dataset Integrity ---")
    all_ok = True
    for key in targets:
        info = FILES[key]
        dest = os.path.join(args.data_dir, info["filename"])
        if not verify_file(dest, info):
            all_ok = False

    if all_ok:
        print("\nAll Cohere-10M files verified successfully!")
        print(f"You can now run: python benchmarks/run_neurips_bigann.py --dataset cohere-10m --data-dir {args.data_dir} --stage both")
    else:
        print("\n[WARNING] One or more files failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
