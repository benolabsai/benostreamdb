# Security Policy

## Supported Versions

BenoStreamDB is a fast-moving project: only the **latest release** is
supported. Older versions are not maintained — please upgrade before reporting
an issue.

| Version        | Supported          |
| -------------- | ------------------ |
| Latest release | :white_check_mark: |
| Older releases | :x:                |

## Reporting a Vulnerability

### Responsible Disclosure

We take the security of BenoStreamDB seriously. If you discover a security vulnerability, please report it responsibly following the guidelines below.

**Do not** open a public GitHub issue for security concerns.

### How to Report

1. **Email**: Send a detailed report to `security@benostreamdb.org`
2. **Encryption**: For sensitive findings, encrypt your report using the PGP key below
3. **Include**: Steps to reproduce, impact assessment, and any proof-of-concept code

### PGP Key

PGP key will be published at [https://benostreamdb.org/.well-known/security.txt](https://benostreamdb.org/.well-known/security.txt) once the public key infrastructure is provisioned.

### Scope

#### In Scope

- SQL injection vulnerabilities in the Python binding layer (`sanitize_sql`, query construction)
- Authentication bypass in Nessie/Trino integration pathways
- Credential leakage in configuration files or environment handling
- Memory safety vulnerabilities in the Rust core engine (use-after-free, buffer overflows)
- Deserialization vulnerabilities in Arrow IPC / FFI boundaries
- Path traversal in table URI resolution
- Privilege escalation in multi-tenant catalog configurations
- Side-channel attacks on vector similarity search
- Malformed input (SQL vector literals, rewriter input, REST request bodies) that
  crashes or panics a server process. These surfaces are fuzzed on every push —
  see `fuzz/` — and a crash is the highest-severity class of finding we accept.
  Production paths are under a no-panic policy (`NO_PANIC_POLICY.md`).

#### Out of Scope

- Vulnerabilities in third-party dependencies (report upstream instead)
- Social engineering or physical attacks
- Denial-of-service attacks against demo/development infrastructure
- Issues requiring physical access to deployment hardware
- Browser-based vulnerabilities in the MinIO web console (upstream MinIO issue)

### Response Expectations

BenoStreamDB is a free, open-source project maintained by a single developer
on a **best-effort basis**. There is no commercial support contract and no
guaranteed response time, but security reports are prioritised over feature
work.

| Milestone              | Target (best effort) |
| ---------------------- | -------------------- |
| Acknowledgment         | As soon as possible, typically within a few days |
| Initial triage         | Within ~1–2 weeks |
| Initial fix            | Prioritised for critical/high severity |
| Coordinated disclosure | Agreed with reporter |

We will, to the best of our ability:
- Acknowledge receipt and confirm the report is being looked at
- Provide an initial severity assessment
- Prioritise a fix for critical/high severity issues
- Keep you informed of progress throughout the resolution process
- Credit you in the security advisory (unless you request anonymity)

### What We Expect

- Good faith reporting with no active exploitation
- No data destruction or disruption to other users
- Allow sufficient time for remediation before any public disclosure
- Provide a clear reproduction path

### What We Commit To

- Acknowledge and review reports as promptly as we can
- Keep you informed of progress and resolution timeline
- Work with you to understand and validate the fix
- Provide appropriate credit in release notes and CVE advisories
- No legal action against good-faith reporters

---

_Last updated: June 2026_
