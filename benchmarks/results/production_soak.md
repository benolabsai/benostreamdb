# Pre-release soak report

| Field | Value |
|---|---|
| Started | 2026-11-12T11:34:36Z |
| Finished | 2026-11-12T11:36:51Z |
| Soak duration | 60s |
| Result | PASSED |

## Soak stats

```
[soak-stats] test=maintenance_churn_soak rounds=47 duration_s=60 rows=1000 peak_rss_mb=558
[soak-stats] test=mixed_workload_soak writes=340 duration_s=60 rows=10000 peak_rss_mb=558
[soak-stats] test=multi_writer_soak writers=8 ops=2002 duration_s=60 rows=233172 peak_rss_mb=701
```
