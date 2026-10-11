---
name: Bug Report
about: Create a report to help us improve BenoStreamDB (Developer Preview)
title: "[BUG] "
labels: ["bug", "triage"]
assignees: ''

---

### BenoStreamDB Environment
- **BenoStreamDB Version:** (e.g. `0.12.0` — check via `pip show benostreamdb` or CLI)
- **Operating System:** (e.g. Ubuntu 24.04, macOS Sonoma 14.5 Apple Silicon, Windows WSL2)
- **Architecture:** (e.g. `x86_64`, `aarch64` / Apple M-series)
- **Python / Tool Version:** (e.g. Python 3.11.8, Rust 1.80)
- **Installation Method:** (e.g. `pip install benostreamdb`, Docker quickstart, source build)

### Bug Category
- [ ] Installation / Wheel packaging failure
- [ ] Query correctness (unexpected query result)
- [ ] Crash / Panic / Segmentation fault
- [ ] Data integrity / Persistence issue
- [ ] Missing index fallback / Puffin recovery failure
- [ ] Performance anomaly

### Description of the Issue
A clear and concise description of what the bug is.

### Minimal Reproducible Example
Please provide a self-contained, minimal script or SQL snippet that reproduces the issue without requiring external credentials or proprietary data:

```python
import benostreamdb as bsdb

# Minimal reproduction script here
```

### Expected Behavior
A clear description of what you expected to happen.

### Actual Behavior / Output & Stack Trace
```text
Paste terminal output, error message, or panic trace here
```

### Additional Context
Any additional context (e.g., storage backend used: local filesystem, S3, MinIO; memory constraints).
