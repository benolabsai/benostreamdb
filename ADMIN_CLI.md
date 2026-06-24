# HyperStreamDB Admin CLI

`hyperstreamdb-admin` is the official administrative tooling for HyperStreamDB. It provides cluster operators with safe, easy-to-use commands to manage storage, optimize performance, and clean up historical data.

## Installation / Building
If you are compiling from source, the admin CLI is built alongside the main engine:
```bash
cargo build --release --bin hyperstreamdb-admin
```

The compiled binary will be located at `target/release/hyperstreamdb-admin`.

## Commands

### 1. `compact`
Over time, continuous ingestion can lead to fragmentation—creating hundreds or thousands of small data files (often called the "small files problem"). The `compact` command merges these small files into larger, optimized files. This significantly improves query performance by reducing metadata overhead and maximizing the efficiency of vectorized scans.

**Usage:**
```bash
hyperstreamdb-admin compact --uri <TABLE_URI> [OPTIONS]
```

**Options:**
- `--uri`, `-u`: **(Required)** The storage URI of the table you want to compact. (e.g., `file:///tmp/hdb`, `s3://my-bucket/table`)
- `--target-file-size`: **(Optional)** The ideal size for the compacted data files. The CLI accepts human-readable string values like `64MB`, `1GB`, `512K`. The default is `64MB`.

**Example:**
```bash
hyperstreamdb-admin compact --uri "s3://production-data/events_table" --target-file-size 128MB
```

### 2. `vacuum`
HyperStreamDB relies on Multi-Version Concurrency Control (MVCC) and copy-on-write semantics. As data is updated, deleted, or compacted, old files are unlinked from the active manifest but remain on disk to allow time-travel queries and safe rollback.

The `vacuum` command permanently deletes these unreferenced historical files to reclaim physical storage space. 

> [!WARNING]
> Running vacuum is irreversible. You will not be able to time-travel to versions older than the retention limit once vacuum completes.

**Usage:**
```bash
hyperstreamdb-admin vacuum --uri <TABLE_URI> [OPTIONS]
```

**Options:**
- `--uri`, `-u`: **(Required)** The storage URI of the table you want to vacuum.
- `--retain-versions`: **(Optional)** The number of historical manifest versions to keep. Any unreferenced data file not tied to the kept versions will be permanently deleted. Default is `10`.

**Example:**
```bash
hyperstreamdb-admin vacuum --uri "s3://production-data/events_table" --retain-versions 5
```

## E2E Testing
The CLI is strictly tested for regressions. You can run the administrative automated bash test locally using:
```bash
./tests/verify_admin_cli.sh
```
This test spins up a mock dataset, runs a compaction pass, and successfully vacuums the resulting orphaned files.
