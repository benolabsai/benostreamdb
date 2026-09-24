#!/usr/bin/env python3
"""Download English-Wikipedia `mediawiki_content_current` XML dump chunks.

Resumable and parallel:
* files already present in data/ are skipped,
* in-flight downloads go to <name>.part and resume via HTTP Range on retry,
* chunks download concurrently (Wikimedia throttles per connection),
* transient failures retry with backoff.
"""
import urllib.request
import re
import os
import time
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

UA = "BenoStreamDB/1.0 (https://github.com/benostreamdb) Python-urllib/3.0"


def download_file(url: str, filepath: str, retries: int = 10) -> bool:
    name = os.path.basename(filepath)
    if os.path.exists(filepath):
        print(f"[skip] already complete: {name}")
        return True

    part = filepath + ".part"
    pos = os.path.getsize(part) if os.path.exists(part) else 0

    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": UA}
            if pos > 0:
                headers["Range"] = f"bytes={pos}-"
                print(f"[resume] {name} at {pos / 1e6:.0f} MB (attempt {attempt})")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                resumed = response.status == 206 and pos > 0
                if not resumed:
                    pos = 0
                remaining = int(response.getheader("Content-Length") or 0)
                total = pos + remaining
                mode = "ab" if resumed else "wb"
                last_report = pos
                with open(part, mode) as out_file:
                    while True:
                        buf = response.read(1 << 20)
                        if not buf:
                            break
                        out_file.write(buf)
                        pos += len(buf)
                        if pos - last_report >= 64 * 1024 * 1024:
                            pct = pos * 100 / max(total, 1)
                            print(f"[get] {name}: {pos / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)",
                                  flush=True)
                            last_report = pos
            if total and pos < total:
                raise IOError(f"connection closed early ({pos}/{total} bytes)")
            os.replace(part, filepath)
            print(f"[ok] {name} ({pos / 1e9:.2f} GB)")
            return True
        except Exception as e:
            status = getattr(e, "code", None)
            print(f"[warn] {name}: attempt {attempt} failed: {e}", flush=True)
            if attempt < retries:
                # Wikimedia 429 windows are minutes-long; back off harder there.
                time.sleep((60 if status == 429 else 5) * attempt)
    print(f"[fail] {name}: giving up after {retries} attempts; keep .part for a later resume")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia XML dump chunks.")
    parser.add_argument("--date", type=str, default="2026-09-01",
                        help="Date of the dump to download (e.g. 2026-09-01)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum number of chunks to download (0 for all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel downloads (Wikimedia throttles per connection)")
    args = parser.parse_args()

    base_url = f"https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/{args.date}/xml/bzip2/"

    print(f"Fetching index from {base_url}...")
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as response:
            html = response.read().decode("utf-8")
    except Exception as e:
        print(f"Failed to fetch index: {e}")
        sys.exit(1)

    # All chunk files look like enwiki-<date>-p<start>p<end>.xml.bz2
    pattern = re.compile(rf'href="(enwiki-{args.date}-p\d+p\d+\.xml\.bz2)"')
    files = list(dict.fromkeys(pattern.findall(html)))  # dedupe, keep order

    if not files:
        print(f"No dump files found for date {args.date} at {base_url}")
        sys.exit(1)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)

    present = [f for f in files if os.path.exists(os.path.join(data_dir, f))]
    missing = [f for f in files if f not in present]
    print(f"Found {len(files)} chunk(s): {len(present)} already downloaded, {len(missing)} to fetch.")
    for f in missing:
        print(f"  missing: {f}")

    if args.limit > 0:
        missing = missing[:args.limit]
        print(f"Limiting download to {len(missing)} chunk(s) as requested.")

    if not missing:
        print("Nothing to do — all chunks present.")
        return

    jobs = [(base_url + f, os.path.join(data_dir, f)) for f in missing]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda j: download_file(*j), jobs))

    ok = sum(results)
    if ok == len(results):
        print("Download complete: all chunks present.")
    else:
        print(f"Download finished with {len(results) - ok} failure(s); rerun to resume (.part files kept).")
        sys.exit(2)


if __name__ == "__main__":
    main()
