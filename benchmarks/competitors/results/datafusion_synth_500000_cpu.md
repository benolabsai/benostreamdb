# Competitor: datafusion / sql

| Metric | Value |
|---|---|
| engine | datafusion |
| workload | sql |
| available | True |
| sql | SELECT category, count(*) AS n, avg(value) AS avg_value FROM t GROUP BY category ORDER BY n DESC LIMIT 10 |
| seconds | 0.123 |
| rows | 10 |
| dataset | synth_500000 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-38-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
