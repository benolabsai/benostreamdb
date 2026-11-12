# Pre-release soak report

| Field | Value |
|---|---|
| Started | 2026-11-12T18:10:07Z |
| Finished | 2026-11-12T18:13:25Z |
| Soak duration | 60s |
| Result | PASSED |

## Soak stats

```
[soak-stats] test=mixed_workload_soak writes=354 duration_s=60 rows=24000 peak_rss_mb=587
[soak-stats] test=maintenance_churn_soak rounds=49 duration_s=60 rows=1000 peak_rss_mb=587
[soak-stats] test=multi_writer_soak writers=8 ops=2277 duration_s=60 rows=241924 peak_rss_mb=752
```
