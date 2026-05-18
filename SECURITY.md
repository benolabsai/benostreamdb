# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| v0.1.x  | :white_check_mark: |
| < v0.1  | :x:                |

## Reporting a Vulnerability

### Responsible Disclosure

We take the security of HyperStreamDB seriously. If you discover a security vulnerability, please report it responsibly following the guidelines below.

**Do not** open a public GitHub issue for security concerns.

### How to Report

1. **Email**: Send a detailed report to `security@hyperstreamdb.io` (placeholder — replace with actual address)
2. **Encryption**: For sensitive findings, encrypt your report using the PGP key below
3. **Include**: Steps to reproduce, impact assessment, and any proof-of-concept code

### PGP Key

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[mPlaceholder: Replace with actual PGP public key fingerprint]
Key ID: 0xXXXXXXXXXXXXXXXX
Fingerprint: XX XX XX XX XX  XX XX XX XX  XX XX  XX XX XX XX  XX XX XX XX
-----END PGP PUBLIC KEY BLOCK-----
```

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

#### Out of Scope

- Vulnerabilities in third-party dependencies (report upstream instead)
- Social engineering or physical attacks
- Denial-of-service attacks against demo/development infrastructure
- Issues requiring physical access to deployment hardware
- Browser-based vulnerabilities in the MinIO web console (upstream MinIO issue)

### Response SLA

| Milestone              | Target     |
| ---------------------- | ---------- |
| Acknowledgment         | **48 hours** |
| Initial triage         | 5 business days |
| Initial fix (CVE draft)| **7 days** |
| Coordinated disclosure | Agreed with reporter |

We will:
- Acknowledge receipt within **48 hours**
- Provide an initial severity assessment within 5 business days
- Work toward an initial fix within **7 days** for critical/high severity issues
- Keep you informed of progress throughout the resolution process
- Credit you in the security advisory (unless you request anonymity)

### Bug Bounty

HyperStreamDB participates in a bug bounty program through [platform placeholder — e.g., HackerOne, Bugcrowd].

- **Critical** (RCE, data exfiltration): Up to $10,000
- **High** (authentication bypass, SQL injection): Up to $5,000
- **Medium** (information disclosure, privilege escalation): Up to $2,000
- **Low** (minor configuration exposure): Up to $500

_Bounty program details and eligibility will be published at the referenced platform once active._

### What We Expect

- Good faith reporting with no active exploitation
- No data destruction or disruption to other users
- Allow sufficient time for remediation before any public disclosure
- Provide a clear reproduction path

### What We Commit To

- Respond within 48 hours of initial report
- Keep you informed of progress and resolution timeline
- Work with you to understand and validate the fix
- Provide appropriate credit in release notes and CVE advisories
- No legal action against good-faith reporters

---

_Last updated: May 2026_
