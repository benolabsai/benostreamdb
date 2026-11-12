# Pre-release soak report

| Field | Value |
|---|---|
| Started | 2026-11-12T21:53:20Z |
| Finished | 2026-11-12T21:56:49Z |
| Soak duration | 60s |
| Result | PASSED |

## Soak stats

```
[soak-stats] test=mixed_workload_soak writes=362 duration_s=60 rows=12000 peak_rss_mb=608
[soak-stats] test=maintenance_churn_soak rounds=53 duration_s=60 rows=1000 peak_rss_mb=601
[soak-stats] test=multi_writer_soak writers=8 ops=2480 duration_s=60 rows=34652 peak_rss_mb=781
```
