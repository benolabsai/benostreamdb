# Competitor: datafusion / sql

| Metric | Value |
|---|---|
| engine | datafusion |
| workload | sql |
| available | True |
| sql | SELECT category, count(*) AS n, avg(value) AS avg_value FROM t GROUP BY category ORDER BY n DESC LIMIT 10 |
| seconds | 0.012 |
| rows | 10 |
| dataset | synth_500000 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=32, ram_gb=121.4, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-38-generic-x86_64-with-glibc2.43, python=3.14.4, containerized=False, gpus=['NVIDIA GeForce RTX 5070 Ti, 16303 MiB', 'NVIDIA GeForce RTX 5070 Ti, 16303 MiB'] |
