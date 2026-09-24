# Soak / stress suite

Long-running tests that intentionally saturate **memory**, **CPU**, and
**disk I/O**. The point is not correctness on the happy path (that is what the
rest of `tests/` is for) but that the engine **degrades gracefully instead of
dying**: under resource pressure it must block, throttle, or return an error —
never OOM-kill, panic, or corrupt data.

## Running

The whole directory is skipped unless `BSDB_STRESS=1`, so the normal
`pytest tests/` run stays fast.

```bash
BSDB_STRESS=1 BSDB_SOAK_SECONDS=120 pytest tests/stress -v
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `BSDB_STRESS` | unset | `1` enables the suite |
| `BSDB_SOAK_SECONDS` | `30` | wall-clock budget for the mixed soak |
| `BSDB_STRESS_RSS_CEILING_GB` | `16` | RSS ceiling asserted by the mixed soak |
| `BSDB_MAX_INGEST_RAM_GB` | — | set by the memory test to force back-pressure |
| `BSDB_INDEX_BUILD_CONCURRENCY` | — | set by the CPU test to bound the build fan-out |

The [`soak` workflow](../../.github/workflows/soak.yml) runs this at
length, together with the Rust soak harness (`cargo test --test soak -- --ignored`).

## What each test proves

| Test | Resource | Assertion |
| --- | --- | --- |
| `test_memory_backpressure.py` | RAM | With a low ingest high-water mark, writes block and resume, data is readable, RSS stays under the ceiling, process alive |
| `test_cpu_saturation.py` | CPU | With index builds gated, concurrent searches still complete within a bounded time |
| `test_disk_io.py` | Disk I/O | Many commit cycles (WAL + manifest + small parquet files) complete and a full read returns exactly the written row count |
| `test_mixed_soak.py` | All | Continuous write/search/read churn for `BSDB_SOAK_SECONDS`: no exception, RSS growth bounded, process alive |

## Relationship to `test_oom.py`

[`test_oom.py`](../../test_oom.py) was the one-off script that originally
surfaced the unbounded-concurrency OOM (it wrote 25 M vectors with no guard).
`test_memory_backpressure.py` is the maintained, assertion-bearing successor: it
exercises the same failure mode but asserts graceful degradation and is part of
CI rather than a script at the repo root.
