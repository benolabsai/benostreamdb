# Pre-release soak report

| Field | Value |
|---|---|
| Started | 2026-11-12T12:52:22Z |
| Finished | 2026-11-12T12:54:37Z |
| Soak duration | 60s |
| Result | PASSED |

## Soak stats

```
[soak-stats] test=mixed_workload_soak writes=350 duration_s=60 rows=20000 peak_rss_mb=581
[soak-stats] test=maintenance_churn_soak rounds=49 duration_s=60 rows=1000 peak_rss_mb=580
[soak-stats] test=multi_writer_soak writers=8 ops=2020 duration_s=60 rows=248748 peak_rss_mb=704
```
