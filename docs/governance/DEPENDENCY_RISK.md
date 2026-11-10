# Dependency Risk & Advisory Tracking Plan

- **Status:** Living document
- **Owner:** BenoStreamDB core
- **Initial tracking plan:** 2026-10-02 (`6213bcdc`)
- **Last verified:** 2026-11-10 (v0.15.1)
- **Companion config:** [`deny.toml`](https://github.com/benolabsai/benostreamdb/blob/main/deny.toml) (`cargo-deny`), [`audit.toml`](https://github.com/benolabsai/benostreamdb/blob/main/audit.toml) (`cargo-audit`)

This document is the **tracking plan** for dependency advisories we have
reviewed and accepted (or resolved). It keeps the allow-lists in
`deny.toml` / `audit.toml` from silently going stale: every advisory has a
named owner, initial review date, last verified date, reachability assessment,
migration path, and resolution record once cleared.

## Policy

1. **`cargo deny check` runs on CI and pre-push gates**
   (`.github/workflows/ci.yml`, `.github/workflows/pr-security-review.yml`, and
   `scripts/pre_release_check.sh`). Any new advisory in the active dependency
   graph fails the build unless it is resolved or documented here and
   allow-listed in `deny.toml` (mirrored in `audit.toml`).
2. **Every ignored advisory must have a dated entry below** with: ID, crate +
   version, dependency chain, reachability, migration path, `Initial review`
   date, `Last verified` date, and `Next review`.
3. **Review cadence:** every release, and at least monthly. Re-run
   `cargo tree --all-features -i <crate>` to confirm the chain still holds.
4. **Prefer removal over acceptance.** If the crate is a *direct* dependency,
   the default is to replace it; acceptance is for *transitive* deps we cannot
   remove without forking an upstream or breaking on-disk format compatibility.
5. **"Unmaintained" ≠ "vulnerable".** RUSTSEC flags unmaintained crates so we
   can plan a migration; it is a maintenance risk, not an exploit.
6. **Keep resolved history.** When an advisory is fixed or pruned from
   `deny.toml`, move it to [Resolved Advisories](#resolved-advisories) with the
   resolution date and commit SHA rather than deleting the audit trail.

---

## Accepted Advisories (Active)

### RUSTSEC-2025-0141 — `bincode` is unmaintained

| Field | Value |
|---|---|
| Crate / version | `bincode` 2.0.1 |
| Kind | Unmaintained (not a vulnerability) |
| Initial review | **2026-09-29** (`1e3df0fa`; formally tracked **2026-10-02**, `6213bcdc`) |
| Last verified | **2026-11-10** |
| Chain | **Direct** dependency of `benostreamdb` (`Cargo.toml`), used in [`hnswio.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/index/hnsw_rs/hnswio.rs) |
| Reachability | Runtime, but only for internal HNSW point-vector dump/load. The bytes are reconstructible Puffin index sidecars authored by the engine itself, not untrusted input. `config::legacy()` preserves byte-compatibility with bincode 1.3 for older dumps. |
| Migration path | Replace with a maintained serializer (`postcard`, `rmp-serde`, or `ciborium`). This changes the wire format, requiring a **format-version bump + index rebuild** — acceptable because HNSW sidecars are rebuildable. |
| Status | Accepted, tracked. |
| Next review | Next format-version bump or v0.16.x; re-check maintained serializers. |

### RUSTSEC-2026-0346 / RUSTSEC-2026-0349 — `rustls` 0.21.12 (EOL)

| Field | Value |
|---|---|
| Crate / version | `rustls` 0.21.12 (end-of-life); `rustls` 0.23.46 is also present and patched |
| Kind | Vulnerability (low practical impact in this configuration) |
| Initial review | **2026-10-14** (`c34eec87`) |
| Partial resolution | **2026-10-14** (`c34eec87`) — bumped `rustls 0.23.45 → 0.23.46`, fixing all four advisories (`-0346/-0347/-0348/-0349`) for primary TLS paths and resolving `-0347/-0348`. |
| Last verified | **2026-11-10** |
| Chain | **Transitive**: `aws-smithy-http-client 1.1.13` → `hyper-rustls 0.24.2` → `tokio-rustls 0.24` → `rustls 0.21.12`. Upstream AWS SDK chain cannot move to 0.23 without an upstream major bump. |
| Reachability | Only the AWS SDK's legacy HTTPS client. Both advisories require, respectively, a **custom `ClientSessionStore`** that `Debug`-formats session values (we implement none; built-in cache `Debug` output is redacted), and **resumption-based key-share prediction** with an on-path attacker (post-quantum confidentiality only; ECH is not enabled). Primary HTTP/object-store/metrics paths use `rustls` 0.23.46. |
| Migration path | Track `aws-smithy-http-client` for a release that drops `rustls` 0.21 (or move AWS SDK integration to a `rustls` 0.23 client). Remove both ignores from `deny.toml` / `audit.toml` once `rustls 0.21.12` leaves `Cargo.lock`. |
| Status | Accepted (upstream-blocked). |
| Next review | Each release; re-run `cargo tree --all-features -i rustls@0.21.12`. |

### RUSTSEC-2026-0382 (and `-0098` / `-0099` / `-0104`) — `rustls-webpki` 0.101.7 & `webpki-roots` 0.26.11

| Field | Value |
|---|---|
| Crate / version | `webpki-roots` 0.26.11 (`0382`) and `rustls-webpki` 0.101.7 (`0098`, `0099`, `0104`, `0382`). Fixed copies (`webpki-roots` 1.0.10 and `rustls-webpki` 0.103.16) **are** in `Cargo.lock`. |
| Kind | Vulnerability (certificate path validation, name constraints, CRL panic) |
| Initial review | **2026-09-24** (`6a0e94da` for `0098/0099/0104`); **2026-10-20** (`5ef7b25b` for `0382`) |
| Partial resolution | **2026-10-20** (`5ef7b25b`) — bumped `rustls-webpki 0.103.15 → 0.103.16` and `webpki-roots 1.0.8 → 1.0.10`. **2026-10-23** (`6ab58081`) — `RUSTSEC-2026-0381` resolved and pruned. |
| Last verified | **2026-11-10** |
| Chain | `webpki-roots 0.26.11` ← `sqlx-core 0.8.6` (0.26 line has no patched release; fix is 1.0.10+). `rustls-webpki 0.101.7` ← `rustls 0.21.12` ← `aws-smithy-http-client 1.1.13` → `hyper-rustls 0.24.2` → `tokio-rustls 0.24` (0.101 line has no backport; fix starts at 0.103.16). |
| Reachability | Only legacy TLS clients of upstream crates (`aws-smithy-http-client` legacy connector and `sqlx-core` non-production SQLite path). Primary TLS paths (`reqwest` / `object_store` / metrics) resolve fixed `rustls-webpki` 0.103.16 and `webpki-roots` 1.0.10. |
| Migration path | Track `aws-smithy-http-client` for `rustls` 0.23 and `sqlx` 0.9 for `webpki-roots` 1.x. Remove ignores from `deny.toml` / `audit.toml` once `0.101.7` and `0.26.11` leave the graph. |
| Status | Accepted (upstream-blocked). |
| Next review | Each release; re-run `cargo tree --all-features -i rustls-webpki@0.101.7` and `-i webpki-roots@0.26.11`. |

### RUSTSEC-2024-0436 — `paste` is unmaintained

| Field | Value |
|---|---|
| Crate / version | `paste` 1.0.15 |
| Kind | Unmaintained (not a vulnerability) |
| Initial review | **2026-09-24** (`a9093f88`; documented **2026-10-02**, `6213bcdc`) |
| Last verified | **2026-11-10** |
| Chain | **Transitive** via `datafusion-common` 52.5.0 (`cargo tree -i paste`) |
| Reachability | **Compile-time only** — `paste` is a proc-macro. It emits no runtime code and has no runtime attack surface. |
| Migration path | Upstream-owned by Apache DataFusion. Track DataFusion releases for a move to `pastey` or removal. |
| Status | Accepted, tracked. |
| Next review | Next DataFusion upgrade; re-run `cargo tree -i paste`. |

### RUSTSEC-2026-0391 / -0398 / -0403 — `zstd` 0.13.3 and `zstd-safe` 7.2.4

| Field | Value |
|---|---|
| Crate / version | `zstd` 0.13.3 (`0391`) and `zstd-safe` 7.2.4 (`0398`, `0403`). Bundled `zstd-sys` is already on patched `2.1.2+zstd.1.5.7`. |
| Kind | Vulnerability (memory disclosure / UB) |
| Initial review | **2026-10-27** (`498cbc1f`) |
| Partial resolution | **2026-10-27** (`498cbc1f`) — `zstd-sys` bumped to `2.1.2+zstd.1.5.7`, clearing `RUSTSEC-2026-0410`. |
| Last verified | **2026-11-10** |
| Chain | **Transitive**: `arrow-ipc 57.3.1`, `parquet 57.3.1`, `datafusion 52.5.0`, `datafusion-datasource 52.5.0`, and `compression-codecs 0.4.38` all require `zstd ^0.13` / `zstd-safe ^7.2`. Only `arrow-ipc 60.0.0+` accepts `zstd ^0.14`. |
| Reachability | **Not reachable in this configuration.**<br>• **0391** requires reusing `with_context` stream encoders/decoders across logically separate streams; we call only [`puffin.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/puffin.rs) in Puffin readers and bulk APIs in Parquet/DataFusion.<br>• **0398** sits behind `zstd-safe`'s `seekable` feature, which is **disabled** (`cargo tree -e features -i zstd-safe` shows `arrays,legacy,std,zdict_builder` only).<br>• **0403** requires `zstd::stream::write::Decoder` with a corrupt legacy (v0.4–v0.7) frame; bulk APIs and `decode_all` are unaffected. |
| Fix status | Blocked upstream on the Apache Arrow 57 / DataFusion 52 stack (no `0.13.x` / `7.x` backports). |
| Migration path | Upgrade the data stack to `arrow`/`parquet` 60 + `datafusion` 54 (`zstd` 0.14 + `zstd-safe` 8.x). Do **not** pull a duplicate `zstd 0.14` C library alongside `0.13`. |
| Status | Accepted (unreachable, upstream-blocked). |
| Next review | Next Arrow / DataFusion major upgrade; re-run `cargo tree -i zstd` and `cargo tree -e features -i zstd-safe`. |

### RUSTSEC-2026-0435 — `thrift` 0.17.0 unbounded memory allocation

| Field | Value |
|---|---|
| Crate / version | `thrift` 0.17.0 |
| Kind | Vulnerability (unbounded allocation in binary protocol `read_bytes`) |
| Initial review | **2026-11-08** (`133c2621`) |
| Last verified | **2026-11-10** |
| Chain | **Transitive**: `thrift 0.17.0` ← `parquet 57.3.1` ← (`benostreamdb`, `datafusion 52.5.0`, `datafusion-common 52.5.0`, `datafusion-datasource-parquet 52.5.0`) |
| Reachability | Reachable only when decoding Parquet Thrift file footers/metadata. Segment Parquet files live inside trusted Iceberg table storage authored by authenticated writers/engines, not arbitrary untrusted Thrift network streams. |
| Fix status | Blocked upstream by `parquet 57.3.1` / `datafusion 52.5.0`. |
| Migration path | Track Arrow (`parquet`) and DataFusion releases for a fix or removal of `thrift 0.17.0`; drop ignore from `deny.toml` / `audit.toml` once cleared. |
| Status | Accepted (upstream-blocked). |
| Next review | Next Arrow / DataFusion major upgrade; re-run `cargo tree -i thrift`. |

### RUSTSEC-2026-0459 — `libsqlite3-sys` 0.30.1 (CVE-2026-119375)

| Field | Value |
|---|---|
| Crate / version | `libsqlite3-sys` 0.30.1 |
| Kind | Vulnerability (uninitialized page-cache memory in SQLite `-wal` recovery) |
| Initial review | **2026-11-09** (`5ac9838a`) |
| Last verified | **2026-11-10** |
| Chain | **Transitive**: `libsqlite3-sys 0.30.1` ← `sqlx-sqlite 0.8.6` ← `sqlx 0.8.6` ← `benostreamdb` |
| Reachability | **Not reachable on adversarial inputs.** Exploitation requires opening an untrusted external SQLite database + `-wal` sidecar file. BenoStreamDB uses `sqlx-sqlite` strictly for local trusted state and never opens user-supplied SQLite files. |
| Fix status | Blocked upstream: `sqlx-sqlite 0.8.6` pins `libsqlite3-sys < 0.31`; the fix ships in `libsqlite3-sys >= 0.39.0`. |
| Migration path | Upgrade `sqlx` (`0.9+`) or drop the `sqlite` feature from `sqlx` if no catalog path needs it. |
| Status | Accepted (unreachable on untrusted inputs). |
| Next review | Next `sqlx` bump or catalog cleanup; re-run `cargo tree -i libsqlite3-sys`. |

---

## Other Allow-Listed Advisories (Active)

These transitive advisories remain in [`deny.toml`](https://github.com/benolabsai/benostreamdb/blob/main/deny.toml) and [`audit.toml`](https://github.com/benolabsai/benostreamdb/blob/main/audit.toml) with no reachable attack surface in supported server/library paths:

| ID | Crate / Version | Chain & Reachability | Initial Review | Last Verified |
|---|---|---|---|---|
| `RUSTSEC-2026-0258` | `h2` 0.3.27 | Unbounded empty `DATA` frames via legacy hyper 0.14 (`aws-smithy-runtime`, `tonic 0.11` OTLP, `axum 0.6`). Primary HTTP/gRPC listeners use `h2` 0.4.19; deployments sit behind a reverse proxy. | 2026-09-24 (`6a0e94da`) | 2026-11-10 |
| `RUSTSEC-2026-0194` / `-0195` | `quick-xml` 0.31.0 & 0.38.4 | `0.31.0` is used only in the offline `ingest_wikipedia.rs` utility; `0.38.4` is transitive via `object_store 0.12.5` talking to trusted S3/Azure/GCS endpoints (not untrusted third-party XML). | 2026-09-24 (`6a0e94da`) | 2026-11-10 |
| `RUSTSEC-2024-0437` | `protobuf` 2.28.0 | Transitive via `prometheus 0.13.4` for outbound metrics serialization only; no untrusted protobuf parsing. | 2026-09-24 (`6a0e94da`) | 2026-11-10 |
| `RUSTSEC-2026-0176` / `-0177` | `pyo3` 0.26.0 | Enabled under `--features python` (`arrow-pyarrow 57.3.1`); upgrade to `0.29+` requires coordinated `arrow-pyarrow` bump. | 2026-09-24 (`6a0e94da`) | 2026-11-10 |
| `RUSTSEC-2026-0190` | `anyhow` 1.0.102 | Unsoundness in `Error::downcast_mut()`. Verified 0 calls to `downcast_mut()` across `src/`, `tests/`, `benchmarks/`, and `examples/`. | 2026-09-24 (`6a0e94da`) | 2026-11-10 |
| `RUSTSEC-2026-0221` *(audit warning)* | `event-listener` 5.4.1 | Transitive via `async-broadcast 0.7.2` (`volo`), `async-lock 3.4.2` (`moka`), `sqlx-core 0.8.6`. Flagged by `cargo audit`; no `!Send` closure tags passed across thread boundaries. | 2026-09-24 (`6a0e94da`) | 2026-11-10 |
| `RUSTSEC-2025-0134` *(unmaintained)* | `rustls-pemfile` 2.2.0 | Transitive via `object_store 0.12.5`; tracked for upstream `object_store` upgrade. | 2026-09-24 (`6a0e94da`) | 2026-11-10 |

---

## Resolved Advisories

Historical advisories that were remediated or cleared from the active dependency graph:

| ID | Crate | Initial Review | Resolved | Resolution Summary & Commit |
|---|---|---|---|---|
| `RUSTSEC-2026-0097` | `rand` | 2026-09-24 (`6a0e94da`) | **2026-10-14** | No longer triggered in active graph; pruned from `deny.toml` (`c34eec87`). |
| `RUSTSEC-2026-0347` / `-0348` | `rustls` | 2026-10-14 (`c34eec87`) | **2026-10-14** | Bumped `rustls 0.23.45 → 0.23.46` (`c34eec87`). |
| `RUSTSEC-2026-0381` | `rustls-webpki` / `webpki-roots` | 2026-10-20 (`5ef7b25b`) | **2026-10-23** | Cleared after lockfile resolution (`6ab58081`); removed from both allow-lists. |
| `RUSTSEC-2026-0385` & yanked dep | `ctutils`, `clipboard-win` | 2026-10-23 (`6ab58081`) | **2026-10-23** | Updated `ctutils → 0.4.5` and `clipboard-win → 5.4.3` (`6ab58081`). |
| `RUSTSEC-2026-0300` / `-0408` / `-0409` / `-0410` | `skiplist`, `zstd-sys` | 2026-10-27 (`498cbc1f`) | **2026-10-27** | Bumped `skiplist → 1.3.4` (`dev-dependencies`) and `zstd-sys → 2.1.2+zstd.1.5.7` (`498cbc1f`). |
| `RUSTSEC-2026-0204` | `crossbeam-epoch` | 2026-09-24 (`6a0e94da`) | **2026-10-14** | Upgraded to `crossbeam-epoch` 0.9.21; pruned from `deny.toml` (`c34eec87`). |
| `RUSTSEC-2026-0233` / `-0234` / `-0235` / `RUSTSEC-2023-0071` | `rkyv`, `rsa` | 2026-09-24 (`6a0e94da`) | **2026-10-14** | Absent from compiled feature graph (`rkyv` removed, `rsa` unused); pruned from `deny.toml` (`c34eec87`). |

---

## How to Add or Resolve an Entry

1. Run `cargo deny check advisories` and `cargo tree --all-features -i <crate>` to get the exact advisory ID, crate version, and dependency chain.
2. Add the ID to [`deny.toml`](https://github.com/benolabsai/benostreamdb/blob/main/deny.toml) and [`audit.toml`](https://github.com/benolabsai/benostreamdb/blob/main/audit.toml) with a concise inline rationale.
3. Add a dated entry to this document recording **Initial review**, **Last verified**, reachability, and migration path.
4. When a fix lands, remove the ID from `deny.toml` / `audit.toml` and move it to [Resolved Advisories](#resolved-advisories) with the resolution date and commit SHA.
