# Pre-release soak report

| Field | Value |
|---|---|
| Started | 2026-10-21T02:07:00Z |
| Finished | 2026-10-21T02:29:00Z |
| Soak duration | 600s per test |
| Result | PASSED |

## Soak stats

```
[soak-stats] test=mixed_workload_soak writes=1026 duration_s=600 rows=16000 peak_rss_mb=5821
[soak-stats] test=maintenance_churn_soak rounds=1053 duration_s=600 rows=1000 peak_rss_mb=6162
[soak-stats] test=multi_writer_soak writers=8 ops=24776 duration_s=600 rows=44425
```

## Notes

- All three soaks ran for **600s** (10x the previous 60s run) and passed their
  invariants: no lost or duplicated rows, no torn snapshots, and compaction /
  vacuum / time-travel remained consistent throughout.
- Peak RSS is now **5.8–6.2 GB** for the single-writer soaks, versus ~605 MB in
  the earlier 60s run. The growth tracks soak duration closely enough to warrant
  a dedicated memory-retention investigation; this run does not prove a leak,
  but it no longer supports a "flat memory" claim.
- The multi-writer soak (8 writers) completed 24,776 operations and ended with
  44,425 rows, a valid serialization of the committed writes.
