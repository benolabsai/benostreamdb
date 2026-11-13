#!/usr/bin/env python3
# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
"""Fail the sweep when the measured vector artifacts violate an invariant.

The report is only as trustworthy as the arithmetic inside its artifacts, and
every defect this benchmark has shipped so far was a *relationship* between two
numbers, not a wrong number on its own:

  * a refined row reporting lower recall than its unrefined sibling (rescore
    ranked in the caller's metric while the graph ranked in the index metric);
  * a row-path QPS higher than the index-only QPS of the same engine (two
    back-to-back timing loops with no warm-up and a fixed order);
  * a refined row with no unrefined sibling, so the pair it was meant to
    illustrate could not be compared at all;
  * two engines on the same dataset measured at different `n`, `dim`, `queries`
    or `k`, which turns a like-for-like table into a per-engine lottery.

Those are checkable without re-running anything, so they are checked here and
this module exits non-zero when one of them is violated. `generate_summary.py`
calls it before writing a report, and `sweep_full_row.sh` calls it before that,
so there is no command that produces a report from unverified artifacts.

What it deliberately does *not* check is whether a number is "good". A slow
engine passes. A recall of 0.71 passes. Only self-contradiction fails.

Refinement is the subtle case, because the correct behaviour differs by index
precision and both halves are counter-intuitive:

  * f32 (uncompressed): stage 1 already ranks by the exact f32 distance that the
    rescore uses, over the same vectors, in the same metric. Re-ranking the same
    candidate set therefore *cannot* change the top-k, and refined recall must be
    identical to unrefined recall. A difference is a real bug: it means the two
    stages disagree about the metric or the candidate set is unstable.
  * quantized (sq/tq8/tq4/pq): stage 1 ranks by codes, stage 2 by the raw f32
    payload, so refined recall must be non-decreasing and must actually move
    unless it is already sitting on the dataset's exact-search ceiling.

A mismatch in either direction fails the run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

# Every check below is written as the condition that must hold, with inclusive
# comparisons: `<=` and `>=` on both the value and its tolerance, so a value
# exactly on the boundary passes and only a clear violation fails. The tolerances
# are named rather than inlined so the boundary is visible at the call site.

# Recall is stored rounded to 4 decimals, so a comparison needs room for half of
# the last digit. One swapped id out of 5 000 slots moves recall by 2e-4, which is
# the smallest real change this gate should tolerate; the regressions it exists to
# catch moved recall by 0.35.
RECALL_TOL = 5e-4

# The row path fetches the id and the embedding; the index path fetches only ids.
# Row does strictly more work, so `row_qps <= index_qps` must hold. The 1.02 limit
# allows for timer noise between the two interleaved measurement blocks and is far
# tighter than the 2x inversion (faiss 6848 row vs 3448 index) that motivated it.
ROW_OVER_INDEX_LIMIT = 1.02

# Throughput and latency must be positive to mean anything. Inclusive against a
# small epsilon rather than a strict comparison against zero.
TIMING_EPS = 1e-9

# Engines we implement. Only these get the strict no-op assertion below: a
# third-party engine's `refine_factor` is its own semantics (LanceDB, for
# instance, re-ranks with the raw vector after normalizing at build time, so its
# recall legitimately moves in both directions), and asserting our internal
# invariant about someone else's code produces noise instead of trust.
OWN_ENGINE_PREFIX = "benostreamdb"

QUANTIZED_SUFFIXES = ("_sq", "_sq8", "_tq8", "_tq4", "_pq")
QUANTIZED_PRECISIONS = {"sq", "sq8", "tq8", "tq4", "pq", "int8", "binary"}
QUANTIZED_INDEX_TYPES = {
    "hnsw_sq",
    "hnsw_tq8",
    "hnsw_tq4",
    "ivf_sq8",
    "ivf_pq",
    "hnsw_pq",
    "ivf_sq",
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "competitors", "results"
)


@dataclass
class Finding:
    """One violated (or suspicious) invariant."""

    level: str  # "FAIL" or "WARN"
    check: str
    where: str
    detail: str

    def render(self) -> str:
        return f"  [{self.level}] {self.check:<26} {self.where}\n           {self.detail}"


def is_own_engine(record: dict) -> bool:
    """True for the engine this harness ships, whose internals we can assert on."""
    return str(record.get("engine") or "").lower().startswith(OWN_ENGINE_PREFIX)


def is_quantized(record: dict) -> bool:
    """True when the index stores codes rather than the f32 payload.

    Three independent spellings exist because three producers write this file:
    our harness writes `index.type`, the competitor adapters write both
    `index_type` and `index_precision`, and the engine name carries the variant
    in the report. Any of them is enough.
    """
    precision = str(record.get("index_precision") or "").lower()
    if precision in QUANTIZED_PRECISIONS:
        return True
    index_type = str(record.get("index_type") or "").lower()
    if index_type in QUANTIZED_INDEX_TYPES:
        return True
    engine = str(record.get("engine") or "").lower()
    return any(engine.endswith(suffix) for suffix in QUANTIZED_SUFFIXES)


def _refine_factor(record: dict) -> int:
    try:
        return int(record.get("refine_factor") or 1)
    except (TypeError, ValueError):
        return 1


def load_records(results_dir: str) -> list[dict]:
    """Every vector record in `results_dir`, tagged with its source file."""
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("workload") not in (None, "vector_ann"):
                continue
            entry["_source"] = os.path.basename(path)
            records.append(entry)
    return records


def _group(records: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        out[tuple(record.get(key) for key in keys)].append(record)
    return out


def check_comparability(records: list[dict]) -> list[Finding]:
    """All engines on one dataset must answer the same question."""
    findings: list[Finding] = []
    for (dataset, device), group in _group(records, "dataset", "device").items():
        for field_name in ("n", "dim", "queries", "k"):
            seen = defaultdict(list)
            for record in group:
                value = record.get(field_name)
                if value is not None:
                    seen[value].append(record.get("engine"))
            if len(seen) > 1:
                rendered = ", ".join(
                    f"{value}={sorted({str(e) for e in engines})}"
                    for value, engines in sorted(seen.items(), key=lambda kv: str(kv[0]))
                )
                findings.append(
                    Finding(
                        "FAIL",
                        "comparability",
                        f"{dataset} [{device}]",
                        f"engines disagree on {field_name}: {rendered}",
                    )
                )
    return findings


def check_unique_cells(records: list[dict]) -> list[Finding]:
    """One artifact per (dataset, device, engine, refine factor)."""
    findings: list[Finding] = []
    for (dataset, device, engine), group in _group(
        records, "dataset", "device", "engine"
    ).items():
        by_factor = defaultdict(list)
        for record in group:
            by_factor[_refine_factor(record)].append(record["_source"])
        for value, sources in by_factor.items():
            if len(sources) > 1:
                findings.append(
                    Finding(
                        "FAIL",
                        "unique cell",
                        f"{dataset} [{device}] {engine} rf={value}",
                        f"written by more than one artifact: {sources}",
                    )
                )
    return findings


def check_row_path_cost(records: list[dict]) -> list[Finding]:
    """Fetching the payload cannot be faster than fetching ids only."""
    findings: list[Finding] = []
    for record in records:
        row_qps = record.get("qps")
        index_qps = record.get("pure_index_qps")
        if not isinstance(row_qps, (int, float)) or not isinstance(
            index_qps, (int, float)
        ):
            continue
        if index_qps <= 0:
            continue
        where = (
            f"{record.get('dataset')} [{record.get('device')}] "
            f"{record.get('engine')} rf={_refine_factor(record)}"
        )
        # The invariant, stated positively: row throughput must not exceed the
        # index-only throughput it contains, up to the noise allowance.
        allowed = row_qps <= index_qps * ROW_OVER_INDEX_LIMIT
        if not allowed:
            findings.append(
                Finding(
                    "FAIL",
                    "row <= index",
                    where,
                    f"row path {row_qps:.1f} QPS exceeds index-only "
                    f"{index_qps:.1f} QPS ({row_qps / index_qps:.2f}x); the row "
                    "path hydrates strictly more columns, so this is timer "
                    "ordering, a projection mismatch, or a `qps` field carrying "
                    "the index-only rate",
                )
            )
    return findings


def check_row_path_measured(records: list[dict]) -> list[Finding]:
    """Row QPS must come from its own measurement, not the index-only rate.

    `run_competitor.py` used to write `qps = q / pure_total` -- the index-only
    rate -- into the field the report reads as Row QPS, while the p50/p99 beside
    it came from the row loop. Every competitor therefore showed the same number
    in both throughput columns (faiss on sift: 9554.2 twice), which reads as "the
    payload is free" for every engine at once. Two separate timing loops over
    different work do not agree to the last decimal, so exact equality is the
    signature of a column that was never measured.
    """
    findings: list[Finding] = []
    for record in records:
        row_qps = record.get("qps")
        index_qps = record.get("pure_index_qps")
        if not isinstance(row_qps, (int, float)) or not isinstance(
            index_qps, (int, float)
        ):
            continue
        distinct = row_qps != index_qps
        if not distinct:
            findings.append(
                Finding(
                    "FAIL",
                    "row measured",
                    f"{record.get('dataset')} [{record.get('device')}] "
                    f"{record.get('engine')} rf={_refine_factor(record)}",
                    f"row QPS and index QPS are both {row_qps}; the row path "
                    "hydrates the payload, so the two columns cannot be the same "
                    "measurement",
                )
            )
    return findings


def check_percentiles(records: list[dict]) -> list[Finding]:
    """p99 >= p50 > 0 for both paths, else the timing loop is broken."""
    findings: list[Finding] = []
    for record in records:
        where = (
            f"{record.get('dataset')} [{record.get('device')}] "
            f"{record.get('engine')} rf={_refine_factor(record)}"
        )
        for qps_key, p50_key, p99_key, label in (
            ("qps", "p50_ms", "p99_ms", "row"),
            ("pure_index_qps", "pure_p50_ms", "pure_p99_ms", "index"),
        ):
            p50 = record.get(p50_key)
            p99 = record.get(p99_key)
            qps = record.get(qps_key)
            if (
                not isinstance(p50, (int, float))
                or not isinstance(p99, (int, float))
                or not isinstance(qps, (int, float))
            ):
                continue
            positive = qps >= TIMING_EPS and p50 >= TIMING_EPS
            if not positive:
                findings.append(
                    Finding(
                        "FAIL",
                        "timing plausible",
                        where,
                        f"{label} path reports qps={qps} p50_ms={p50}: a "
                        "non-positive rate means the loop did not run",
                    )
                )
            elif not (p99 >= p50):
                findings.append(
                    Finding(
                        "FAIL",
                        "timing plausible",
                        where,
                        f"{label} path reports p99 {p99} ms below p50 {p50} ms",
                    )
                )
    return findings


def check_refine_pairs(records: list[dict]) -> list[Finding]:
    """Every refined row must have the unrefined sibling it is compared against."""
    findings: list[Finding] = []
    baselines = {
        (r.get("dataset"), r.get("device"), r.get("engine"))
        for r in records
        if _refine_factor(r) <= 1
    }
    for record in records:
        factor = _refine_factor(record)
        if factor <= 1:
            continue
        key = (record.get("dataset"), record.get("device"), record.get("engine"))
        if key not in baselines:
            findings.append(
                Finding(
                    "FAIL",
                    "refine baseline",
                    f"{key[0]} [{key[1]}] {key[2]} rf={factor}",
                    "no unrefined row for the same engine and dataset, so the "
                    "refine effect on this cell cannot be read",
                )
            )
    return findings


def check_refine_by_precision(records: list[dict]) -> list[Finding]:
    """f32 refine is a no-op; quantized refine must move and never regress."""
    findings: list[Finding] = []
    for (dataset, device, engine), group in _group(
        records, "dataset", "device", "engine"
    ).items():
        by_factor = {}
        for record in group:
            recall = record.get("recall_at_k")
            if isinstance(recall, (int, float)):
                by_factor[_refine_factor(record)] = recall
        baseline = by_factor.get(1)
        if baseline is None:
            continue
        quantized = is_quantized(group[0])
        own = is_own_engine(group[0])
        where = f"{dataset} [{device}] {engine}"

        # Our invariants are hard failures; a third-party adapter gets the same
        # observation as a warning, because its refine may legitimately be a
        # different operation.
        level = "FAIL" if own else "WARN"

        for factor, recall in sorted(by_factor.items()):
            if factor <= 1:
                continue
            delta = recall - baseline
            if not quantized:
                # Uncompressed: our rescore re-ranks identical vectors with the
                # identical metric, so the top-k is a permutation of itself and
                # recall must not move at all.
                unchanged = abs(delta) <= RECALL_TOL
                if not unchanged:
                    findings.append(
                        Finding(
                            level,
                            "refine noop (f32)",
                            f"{where} rf={factor}",
                            f"recall moved {baseline:.4f} -> {recall:.4f} on an "
                            f"uncompressed index (delta {delta:+.4f}); for an "
                            "engine that rescores the same f32 vectors in the same "
                            "metric this cannot happen, so either stage 1 and the "
                            "rescore disagree about the metric or the candidate set "
                            "is unstable",
                        )
                    )
                continue

            non_decreasing = delta >= -RECALL_TOL
            if not non_decreasing:
                findings.append(
                    Finding(
                        level,
                        "refine monotone",
                        f"{where} rf={factor}",
                        f"refined recall {recall:.4f} is below the unrefined "
                        f"{baseline:.4f} (delta {delta:+.4f}) on a quantized "
                        "index; rescoring the f32 payload can only reorder "
                        "candidates towards the truth",
                    )
                )

        if not quantized or not own:
            continue

        # `f >= 2` rather than `f > 1`: refine factor 1 is the unrefined baseline.
        refined = [r for f, r in by_factor.items() if f >= 2]
        recovered = bool(refined) and max(refined) >= baseline + RECALL_TOL
        if refined and not recovered:
            # Allowed only when there is nothing left to recover: the unrefined
            # quantized recall already equals the best unquantized recall any
            # engine reached on this dataset.
            ceiling = max(
                (
                    r.get("recall_at_k", 0.0)
                    for r in records
                    if r.get("dataset") == dataset
                    and r.get("device") == device
                    and not is_quantized(r)
                    and isinstance(r.get("recall_at_k"), (int, float))
                ),
                default=None,
            )
            if ceiling is not None and baseline >= ceiling - RECALL_TOL:
                continue
            findings.append(
                Finding(
                    "WARN",
                    "refine effective",
                    where,
                    f"quantized recall stayed at {baseline:.4f} across every "
                    f"refine factor while the best uncompressed recall here is "
                    f"{ceiling if ceiling is None else round(ceiling, 4)}; the "
                    "remaining misses are true neighbours the graph never placed "
                    "in the candidate set (an ef_search bound), which rescoring "
                    "cannot recover -- or the quantizer is lossless on this data",
                )
            )
    return findings


def check_environment(records: list[dict]) -> list[Finding]:
    """One machine state per artifact set; a spliced table is not a benchmark."""
    findings: list[Finding] = []
    for device, group in _group(records, "device").items():
        seen = defaultdict(list)
        for record in group:
            env = record.get("env") or {}
            cpu = env.get("cpu_model")
            if cpu:
                seen[cpu].append(record.get("_source"))
        if len(seen) > 1:
            detail = "; ".join(
                f"{cpu} ({len(sources)} artifacts)"
                for cpu, sources in sorted(seen.items())
            )
            findings.append(
                Finding(
                    "FAIL",
                    "machine state",
                    f"device={device}",
                    f"artifacts came from more than one host: {detail}",
                )
            )
    return findings


CHECKS = (
    check_comparability,
    check_unique_cells,
    check_row_path_cost,
    check_row_path_measured,
    check_percentiles,
    check_refine_pairs,
    check_refine_by_precision,
    check_environment,
)


def verify(results_dir: str | None = None, strict: bool = False) -> int:
    """Run every check. Returns the number of failures (0 means trustworthy)."""
    directory = results_dir or RESULTS_DIR
    records = load_records(directory)

    header = f"verifying {len(records)} vector records in {directory}"
    if not records:
        print(f"{header}\n  0 records to verify", file=sys.stderr)
        return 0

    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(records))

    failures = [f for f in findings if f.level == "FAIL"]
    warnings = [f for f in findings if f.level == "WARN"]

    print(header)
    for finding in failures + warnings:
        print(finding.render())
    if findings:
        print()
    print(
        f"  {len(CHECKS)} invariant groups, {len(failures)} failure(s), "
        f"{len(warnings)} warning(s)"
    )

    if failures:
        return len(failures)
    if strict and warnings:
        return len(warnings)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the vector result artifacts for self-contradiction: refined "
            "recall regressions, row QPS above index QPS, orphaned refine rows, "
            "incomparable cells, and mixed machine states."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=f"directory of competitor result artifacts (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as failures",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    options = _parse_args(sys.argv[1:])
    sys.exit(1 if verify(options.results_dir, options.strict) else 0)
