#!/usr/bin/env python3
"""Download and prepare BigANN datasets (100M and 1B) with resumable streaming.

Slices the official NeurIPS BigANN base vectors from Facebook Research CDN:
- base.100M.u8bin (first 100M vectors = 12,800,000,008 bytes with patched count)
- base.1B.u8bin (full 1B vectors = 128,000,000,008 bytes)
- GT.100M and GT.1B ground truth files (8,000,008 bytes each)
"""

import argparse
import os
import struct
import sys
import time
import urllib.request
import numpy as np

URL_BASE_1B = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
URL_GT_100M = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/bigann-100M"
URL_GT_1B = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_1B/bigann-1B"

BYTES_100M = 8 + 100_000_000 * 128  # 12,800,000,008
BYTES_1B = 8 + 1_000_000_000 * 128   # 128,000,000,008
BYTES_GT = 8 + 10_000 * 100 * 4 * 2  # 8,000,008


def download_file_resumable(url: str, dest_path: str, target_bytes: int, log_interval_bytes: int, patch_count: int = None):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    cur_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    if cur_bytes > target_bytes:
        print(f"[{dest_path}] File size {cur_bytes} > target {target_bytes}. Truncating to target.")
        with open(dest_path, "r+b") as f:
            f.truncate(target_bytes)
        cur_bytes = target_bytes

    if cur_bytes == target_bytes:
        print(f"[{dest_path}] Already fully downloaded ({target_bytes:,} bytes).")
        if patch_count is not None:
            _patch_header(dest_path, patch_count)
        _verify_header(dest_path, patch_count)
        return

    print(f"[{dest_path}] Downloading {target_bytes - cur_bytes:,} remaining bytes (resuming from {cur_bytes:,})...")
    req = urllib.request.Request(url, headers={"Range": f"bytes={cur_bytes}-{target_bytes - 1}"})
    
    chunk_size = 4 * 1024 * 1024  # 4 MB chunks
    t_start = time.time()
    t_last_log = t_start
    bytes_downloaded_this_session = 0
    bytes_since_last_log = 0

    with urllib.request.urlopen(req) as resp, open(dest_path, "ab") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            cur_bytes += len(chunk)
            bytes_downloaded_this_session += len(chunk)
            bytes_since_last_log += len(chunk)

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
                t_last_log = now

    if patch_count is not None:
        _patch_header(dest_path, patch_count)
    _verify_header(dest_path, patch_count)


def _patch_header(dest_path: str, count: int):
    with open(dest_path, "r+b") as f:
        f.seek(0)
        old_val = struct.unpack("<I", f.read(4))[0]
        if old_val != count:
            print(f"[{os.path.basename(dest_path)}] Patching header count: {old_val:,} -> {count:,} vectors.")
            f.seek(0)
            f.write(struct.pack("<I", count))


def _verify_header(dest_path: str, expected_count: int = None):
    h = np.fromfile(dest_path, dtype=np.int32, count=2)
    size = os.path.getsize(dest_path)
    print(f"[{os.path.basename(dest_path)}] Verified size: {size:,} bytes | Header: n_vectors={h[0]:,}, dim={h[1]}")
    if expected_count is not None and h[0] != expected_count:
        raise ValueError(f"Header count mismatch: expected {expected_count}, got {h[0]}")


def main():
    parser = argparse.ArgumentParser(description="Download BigANN 100M and 1B datasets")
    parser.add_argument("--data-dir", default="/home/ralbright/data/datasets/bigann", help="Target datasets directory")
    parser.add_argument("--target", choices=["100m", "1b", "all"], default="all", help="Target dataset to download")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # 1. Ground truth files
    gt_100m_path = os.path.join(args.data_dir, "GT.100M")
    gt_1b_path = os.path.join(args.data_dir, "GT.1B")
    
    if args.target in ("100m", "all"):
        if not os.path.exists(gt_100m_path) or os.path.getsize(gt_100m_path) != BYTES_GT:
            print(f"Downloading GT.100M...")
            download_file_resumable(URL_GT_100M, gt_100m_path, BYTES_GT, log_interval_bytes=4 * 1024 * 1024)
        else:
            _verify_header(gt_100m_path)

    if args.target in ("1b", "all"):
        if not os.path.exists(gt_1b_path) or os.path.getsize(gt_1b_path) != BYTES_GT:
            print(f"Downloading GT.1B...")
            download_file_resumable(URL_GT_1B, gt_1b_path, BYTES_GT, log_interval_bytes=4 * 1024 * 1024)
        else:
            _verify_header(gt_1b_path)

    # 2. Base vectors
    if args.target in ("100m", "all"):
        base_100m_path = os.path.join(args.data_dir, "base.100M.u8bin")
        print("\n=== Downloading base.100M.u8bin (100M vectors = 12.8 GB) ===")
        download_file_resumable(
            URL_BASE_1B,
            base_100m_path,
            BYTES_100M,
            log_interval_bytes=500 * 1024 * 1024,  # log every 500 MB
            patch_count=100_000_000,
        )

    if args.target in ("1b", "all"):
        base_1b_path = os.path.join(args.data_dir, "base.1B.u8bin")
        print("\n=== Downloading base.1B.u8bin (1B vectors = 128 GB) ===")
        download_file_resumable(
            URL_BASE_1B,
            base_1b_path,
            BYTES_1B,
            log_interval_bytes=2 * 1024 * 1024 * 1024,  # log every 2 GB
        )

    print("\nAll requested BigANN downloads and verifications complete.")


if __name__ == "__main__":
    main()

