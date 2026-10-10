# Competitor: datafusion / sql

| Metric | Value |
|---|---|
| engine | datafusion |
| workload | sql |
| available | True |
| sql | SELECT category, count(*) AS n, avg(value) AS avg_value FROM t GROUP BY category ORDER BY n DESC LIMIT 10 |
| seconds | 0.131 |
| rows | 10 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, os=Linux-7.0.0-34-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
