# Benchmark rollup (2026-10-10T15:44:06-04:00)

Hardware profile: cpus=8 mem=16g — see hardware_profile.txt

| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|
| pgvector | cpu | fashion-mnist-784-euclidean | 1.0 | 835.8 | 1.158 | 1.725 | 11.249 | 166.35 |
| pgvector | cpu | gist-960-euclidean | 0.9988 | 340.4 | 2.98 | 3.95 | 26.002 | 247.29 |
| pgvector | cpu | glove-100-angular | 0.462 | 762.9 | 1.311 | 1.991 | 6.216 | 24.85 |
| pgvector | cpu | glove-200-angular | 0.1758 | 586.0 | 1.685 | 2.851 | 8.615 | 42.27 |
| pgvector | cpu | lastfm-64-dot | 0.996 | 1221.8 | 0.802 | 1.261 | 6.803 | 18.51 |
| pgvector | cpu | mnist-784-euclidean | 1.0 | 746.5 | 1.318 | 2.262 | 13.644 | 166.36 |
| pgvector | cpu | nytimes-256-angular | 0.0908 | 521.3 | 1.875 | 3.236 | 10.346 | 50.91 |
| pgvector | cpu | sift-128-euclidean | 1.0 | 1016.4 | 0.969 | 1.333 | 4.125 | 29.21 |
