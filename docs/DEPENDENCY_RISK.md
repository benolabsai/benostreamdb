# Dependency Risk & Advisory Tracking Plan

Status: living document. Owner: BenoStreamDB core.
Companion config: [`audit.toml`](../audit.toml) (cargo-audit), [`deny.toml`](../deny.toml) (cargo-deny).

This document is the **tracking plan** for advisories we have reviewed and
accepted. It exists so the allow-lists in `audit.toml` / `deny.toml` cannot
silently go stale: every ignored advisory has a named owner, a reachability
assessment, a migration path, and a review date.

## Policy

1. **`cargo audit` and `cargo deny` run in CI** (`.github/workflows/`). A new
   advisory fails the build unless it is added here *and* to both allow-lists.
2. **Every ignored advisory must have an entry below** with: ID, crate +
   version, dependency chain, reachability, migration path, and next review.
3. **Review cadence:** every release, and at least monthly. Re-run
   `cargo tree -i <crate>` to confirm the chain still holds.
4. **Prefer removal over acceptance.** If the crate is a *direct* dependency,
   the default is to replace it; acceptance is for *transitive* deps we cannot
   remove without forking an upstream.
5. **"Unmaintained" ≠ "vulnerable".** RUSTSEC flags unmaintained crates so we
   can plan a migration; it is a maintenance risk, not an exploit.

## Accepted advisories

### RUSTSEC-2025-0141 — `bincode` is unmaintained

| Field | Value |
|---|---|
| Crate / version | `bincode` 2.0.1 |
| Kind | Unmaintained (not a vulnerability) |
| Chain | **Direct** dependency of `benostreamdb` (`Cargo.toml`), used in [`src/core/index/hnsw_rs/hnswio.rs`](../src/core/index/hnsw_rs/hnswio.rs) |
| Reachability | Runtime, but only for HNSW point-vector dump/load. The bytes are our own index sidecars (reconstructible from the data), not untrusted input. `config::legacy()` is byte-compatible with bincode 1.3, so older dumps still load. |
| Migration path | Replace with a maintained serializer (`postcard`, `rmp-serde`, or `ciborium`). This changes the on-disk format, so it requires a **format-version bump + index rebuild** — acceptable because HNSW sidecars are reconstructible. No drop-in maintained successor shares bincode's wire format. |
| Status | Accepted, tracked. |
| Next review | Each release; re-check for a maintained fork or a DataFusion/Arrow-provided serializer. |

### RUSTSEC-2026-0346 / RUSTSEC-2026-0349 — `rustls` 0.21.12 (EOL)

| Field | Value |
|---|---|
| Crate / version | `rustls` 0.21.12 (end-of-life) |
| Kind | Vulnerability (low practical impact in this configuration) |
| Chain | **Transitive**: `aws-smithy-http-client` → `hyper-rustls 0.24` → `tokio-rustls 0.24` → `rustls 0.21.12`. The AWS SDK chain is already at the latest compatible release and cannot move to 0.23 without an upstream change. |
| Reachability | Only the AWS SDK's legacy HTTPS client. Both advisories require, respectively, a **custom `ClientSessionStore`** that `Debug`-formats session values (we implement none; rustls's built-in session cache `Debug` output is redacted), and **resumption-based key-share prediction** with an on-path attacker (impact limited to loss of post-quantum confidentiality — no authentication bypass, no key compromise). ECH is not enabled. |
| Fix status | `rustls 0.23.45 → 0.23.46` is applied; 0.23.46 fixes all four rustls advisories (0346/0347/0348/0349). The 0.21 branch is EOL and has no backport. |
| Migration path | Track `aws-smithy-http-client` for a release that drops rustls 0.21 (or move the AWS integration to a rustls-0.23 client). Remove both ignores from `deny.toml` once `rustls 0.21.12` leaves `Cargo.lock`. |
| Status | Accepted, tracked. |
| Next review | Each release; re-run `cargo tree -i rustls@0.21.12`. |

### RUSTSEC-2024-0436 — `paste` is unmaintained

| Field | Value |
|---|---|
| Crate / version | `paste` 1.0.15 |
| Kind | Unmaintained (not a vulnerability) |
| Chain | **Transitive** via `datafusion-common` 52.5.0 (`cargo tree -i paste`) |
| Reachability | **Compile-time only** — `paste` is a proc-macro. It produces no runtime code and has no runtime attack surface. |
| Migration path | None available to us: it is pulled in by DataFusion. Track DataFusion releases for a move to `pastey` (the maintained fork) or removal. |
| Status | Accepted, tracked. |
| Next review | Each DataFusion bump; re-run `cargo tree -i paste`. |

## Other allow-listed advisories

The remaining entries in `audit.toml` / `deny.toml` are reviewed transitive
advisories with no reachable runtime surface in the production paths (proc-macro
only, unreachable code path, or accepted upstream). They are re-reviewed on the
same cadence. Notable ones:

| ID | Crate | Why accepted |
|---|---|---|
| RUSTSEC-2026-0258 | `h2` | Unbounded empty DATA frames (DoS). Deployment is assumed behind a mitigating reverse proxy. |
| RUSTSEC-2026-0195 / -0194 | `quick-xml` | Only reachable in the offline `ingest_wikipedia.rs` utility, not the database paths. |
| RUSTSEC-2023-0071 | `rsa` | Marvin attack; not on a production path. |
| RUSTSEC-2026-0176 / -0177 | `pyo3` | Python-binding edge cases; tracked for the next pyo3 bump. |

## How to add a new accepted advisory

1. Run `cargo audit` and `cargo tree -i <crate>` to get the ID and chain.
2. Add the ID to `audit.toml` **and** `deny.toml` with a one-line reason.
3. Add a full entry to this document (copy the table above).
4. Set a next-review date and note the migration path.
5. If the crate is a direct dependency, open a tracking issue to replace it.
