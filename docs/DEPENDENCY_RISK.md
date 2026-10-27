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

### RUSTSEC-2026-0382 — distrusted emSign intermediates (`webpki-roots`, `rustls-webpki`)

| Field | Value |
|---|---|
| Crate / version | `webpki-roots` 0.26.11 and `rustls-webpki` 0.101.7. The fixed copies (`webpki-roots` 1.0.10 and `rustls-webpki` 0.103.16) **are** in `Cargo.lock`; only the upstream-pinned copies below remain affected. |
| Kind | Vulnerability (certificate path validation) |
| Chain | `webpki-roots 0.26.11` ← `sqlx-core 0.8.6` (the fix is 1.0.10+; the 0.26 line has no patched release). `rustls-webpki 0.101.7` ← `rustls 0.21.12` ← `aws-smithy-http-client 1.1.13` → `hyper-rustls 0.24` → `tokio-rustls 0.24` (rustls 0.21 is EOL; the 0.101 line has no backport — the fix starts at 0.103.16). |
| Reachability | Only legacy TLS clients of upstream crates: the AWS SDK's rustls-0.21 client and `sqlx`'s own connector (we do not use `sqlx`'s TLS on the primary object-store / Flight SQL paths). The advisory requires a certificate chain that includes the distrusted emSign intermediates; the well-known private keys only matter if such a path is presented by a peer. Our own TLS paths (reqwest/object_store/metrics) resolve `rustls-webpki` 0.103.16 and `webpki-roots` 1.0.10. |
| Fix status | `cargo update -p rustls-webpki@0.103.15 -p webpki-roots@1.0.8` applied (→ 0.103.16 / 1.0.10). The remaining 0.101.7 / 0.26.11 copies cannot move without an upstream major bump. |
| Migration path | Track `aws-smithy-http-client` for a release that drops rustls 0.21, and `sqlx` 0.9 for the `webpki-roots` 1.x move. Remove the ignore from `deny.toml` / `audit.toml` once the affected versions leave `Cargo.lock`. |
| Status | Accepted, tracked. |
| Next review | Each release; re-run `cargo tree -i rustls-webpki@0.101.7` and `cargo tree -i webpki-roots@0.26.11`. |

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

### RUSTSEC-2026-0391 / -0398 / -0403 — `zstd` 0.13.3 and `zstd-safe` 7.2.4

| Field | Value |
|---|---|
| Crate / version | `zstd` 0.13.3 (0391) and `zstd-safe` 7.2.4 (0398, 0403). `zstd-sys` was already bumped to the patched `2.1.2+zstd.1.5.7`, clearing RUSTSEC-2026-0410. |
| Kind | Vulnerability (memory disclosure / UB) |
| Chain | **Transitive**: `arrow-ipc 57.3.1`, `parquet 57.3.1`, `datafusion 52.5.0`, `datafusion-datasource 52.5.0` and `compression-codecs 0.4.38` all require `zstd ^0.13` / `zstd-safe ^7.2`. Only `arrow-ipc` **60.0.0** accepts `zstd ^0.14`, i.e. the fix rides on the Arrow 60 / DataFusion 54 migration. |
| Reachability | **Not reachable in this configuration.** <br>• **0391** needs the `with_context` stream encoders/decoders to be reused across logically separate streams (e.g. per-user contexts). We never call them: the only direct `zstd` call sites are [`zstd::decode_all`](../src/core/puffin.rs) in the Puffin blob reader, and the parquet/datafusion paths use the bulk API.<br>• **0398** lives behind `zstd-safe`'s `seekable` feature, which is **not enabled** (`cargo tree -e features -i zstd-safe` shows `arrays,legacy,std,zdict_builder` only).<br>• **0403** requires the `zstd::stream::write::Decoder` push path with a corrupt legacy (v0.4–v0.7) frame; the advisory states `zstd::stream::read::Decoder`, `zstd::decode_all` and the bulk API are unaffected, and those are the only paths in the graph. |
| Fix status | Blocked upstream: no 0.13.x / 7.x backport exists. `cargo update -p zstd -p zstd-safe` resolves to the vulnerable versions. |
| Migration path | Bump the data stack to `arrow`/`parquet` 60 + `datafusion` 54, which moves `zstd` to 0.14 and `zstd-safe` to 8.x. Do **not** add a second `zstd 0.14` copy for our own direct use: it would duplicate the C library without removing the vulnerable copies transitively used by parquet/Arrow. Remove the three ignores once `cargo tree -i zstd` reports 0.14.3+. |
| Status | Accepted, tracked. |
| Next review | Each Arrow/DataFusion bump; re-run `cargo tree -i zstd` and `cargo tree -e features -i zstd-safe`. |

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
