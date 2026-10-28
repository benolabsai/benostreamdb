#!/usr/bin/env python3
"""Machine-state fingerprint, calibration probes, and the session ledger.

Why this exists
---------------
Every number in `benchmarks/results/BENCHMARK_REPORT.md` is measured on one host,
so the *machine state* is part of the measurement: DIMM timings, CPU voltage /
frequency limits, and the kernel all move every participant in the comparison
(BenoStreamDB and faiss, pgvector, OpenSearch, ...) at once. A later run on a
different state is not a regression or a win — it is a different experiment.

Two consequences this module enforces:

1. a sweep records the state it ran under, plus cheap calibration probes, so a
   future reader can tell *why* a column moved; and
2. `generate_summary.py` refuses to build a table from sections measured in
   different sessions. Partial refreshes are the failure mode we care about: a
   fresh CPU-side run spliced into stored competitor rows (or into the GPU
   section, which does not move with host DIMM timings at all) reads as a real
   delta when it is an artefact of scheduling the sweep in two sittings.

Usage
-----
    # before/after each section of a sweep (artifacts are relative to benchmarks/)
    python benchmarks/session_provenance.py capture --section vector \
        --artifact competitors/results/vector_matrix.json

    python benchmarks/session_provenance.py verify     # exit 1 on mixed sessions
    python benchmarks/session_provenance.py show

`BSDB_BENCH_SESSION_LEDGER` overrides the ledger path (tests, CI).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from typing import Any, Iterable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEDGER = os.path.join(BASE_DIR, "results", "session.json")

# Fields that make two measurements comparable. Hostname, PID and timestamps are
# deliberately absent: they differ between runs that measure the same thing.
COMPARABLE_KEYS = (
    "cpu_model",
    "cpu_cores",
    "mem_total_bytes",
    "kernel",
    "governor",
    "cpu_freq_limits",
    "dimm_config",
    "engine_version",
)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Low-level probes. Every probe is best-effort: a missing permission, file or
# module yields None rather than failing the sweep (a benchmark run must not
# depend on being root).
# ---------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _read_first_line(path: str) -> str | None:
    value = _read(path)
    return value.splitlines()[0].strip() if value else None


def _run(argv: list[str], timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _cpu_model() -> str | None:
    text = _read("/proc/cpuinfo") or ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _mem_total_bytes() -> int | None:
    text = _read("/proc/meminfo") or ""
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def _cpu_freq_limits() -> dict[str, Any] | None:
    base = "/sys/devices/system/cpu/cpu0/cpufreq"
    driver = _read_first_line(os.path.join(base, "scaling_driver"))
    minimum = _read_first_line(os.path.join(base, "scaling_min_freq"))
    maximum = _read_first_line(os.path.join(base, "scaling_max_freq"))
    # intel_pstate exposes the hardware limits; amd-pstate/acpi expose only the
    # policy limits, which is why both are recorded when present.
    hw_min = _read_first_line(os.path.join(base, "cpuinfo_min_freq"))
    hw_max = _read_first_line(os.path.join(base, "cpuinfo_max_freq"))
    if not any((driver, minimum, maximum, hw_min, hw_max)):
        return None
    return {
        "driver": driver,
        "policy_min_khz": int(minimum) if minimum and minimum.isdigit() else None,
        "policy_max_khz": int(maximum) if maximum and maximum.isdigit() else None,
        "hw_min_khz": int(hw_min) if hw_min and hw_min.isdigit() else None,
        "hw_max_khz": int(hw_max) if hw_max and hw_max.isdigit() else None,
        "boost_disabled": _read_first_line(os.path.join(base, "boost")) == "0"
        or _read_first_line("/sys/devices/system/cpu/intel_pstate/no_turbo") == "1",
    }


def _package_power_limits() -> dict[str, Any] | None:
    """RAPL package limits. Frequently unavailable (AMD without `rapl`): None."""
    base = "/sys/class/powercap/intel-rapl:0"
    limit_uw = _read_first_line(os.path.join(base, "constraint_0_power_limit_uw"))
    max_uw = _read_first_line(os.path.join(base, "constraint_0_max_power_uw"))
    if limit_uw is None and max_uw is None:
        return None
    return {
        "constraint_0_power_limit_w": int(limit_uw) / 1e6 if limit_uw else None,
        "constraint_0_max_power_w": int(max_uw) / 1e6 if max_uw else None,
    }


def _dimm_config() -> str | None:
    """DIMM speed and configured timings, as a single comparable string.

    `dmidecode` needs root; `lshw` usually works unprivileged but reports less.
    When neither is available this is None and the report says so — the point is
    to never *assume* the memory subsystem did not change.
    """
    dmidecode = _run(["dmidecode", "-t", "memory"], timeout=10.0)
    if dmidecode:
        speeds = sorted(
            {
                match.group(1).strip()
                for match in re.finditer(r"Configured Memory Speed:\s*(.+)", dmidecode)
            }
        )
        speeds += sorted(
            {
                match.group(1).strip()
                for match in re.finditer(r"Speed:\s*(\d+ MT/s|Unknown)", dmidecode)
            }
        )
        parts = sorted(
            {
                " ".join(match.groups())
                for match in re.finditer(r"(?:Configured )?Memory Speed:\s*(.+)", dmidecode)
            }
        )
        if parts:
            return "; ".join(parts)
        if speeds:
            return "; ".join(dict.fromkeys(speeds))
    # `lshw` without root only reports the total size ("130GiB System memory"),
    # which is not a timing signal — accepting it would make two different memory
    # configurations look identical. Only a line that states a clock or a size per
    # module counts; otherwise record "unknown" so the reader knows to check.
    lshw = _run(["lshw", "-C", "memory"], timeout=10.0)
    if lshw:
        modules = [
            line.strip()
            for line in lshw.splitlines()
            if re.search(r"(clock:|size:\s*\d+\s*(MiB|GiB)\s*$)", line) and "System memory" not in line
        ]
        if any("clock:" in line for line in modules):
            return " | ".join(dict.fromkeys(modules))
    return None


def _engine_version() -> str | None:
    """The workspace version: the single source of truth for release identity."""
    text = _read(os.path.join(BASE_DIR, "..", "Cargo.toml")) or ""
    workspace = re.search(
        r"\[workspace\.package\][^\[]*?^version\s*=\s*\"([^\"]+)\"",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if workspace:
        return workspace.group(1)
    any_version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return any_version.group(1) if any_version else None


def fingerprint() -> dict[str, Any]:
    """The state a measurement is only valid within."""
    return {
        "cpu_model": _cpu_model(),
        "cpu_cores": os.cpu_count(),
        "mem_total_bytes": _mem_total_bytes(),
        "kernel": platform.release(),
        "governor": _read_first_line(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "cpu_freq_limits": _cpu_freq_limits(),
        "package_power_limits": _package_power_limits(),
        "dimm_config": _dimm_config(),
        "engine_version": _engine_version(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }


def fingerprint_id(fp: dict[str, Any] | None = None) -> str:
    """Stable short id over the comparable fields (order-independent)."""
    subject = {key: (fp or fingerprint()).get(key) for key in COMPARABLE_KEYS}
    encoded = json.dumps(subject, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Calibration: cheap, deterministic probes whose only job is to move when the
# machine state moves. They are not benchmarks of the engine.
# ---------------------------------------------------------------------------

CALIBRATION_RANDOM_GATHER_BYTES = 64 * 1024 * 1024
CALIBRATION_TRIAD_BYTES = 64 * 1024 * 1024
CALIBRATION_MATMUL_SIZE = 1024


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _random_gather_ns(np: Any) -> float:
    """Latency proxy: random access into a 64 MiB buffer, ns per access."""
    elements = CALIBRATION_RANDOM_GATHER_BYTES // 8
    data = np.ones(elements, dtype=np.float64)
    rng = np.random.default_rng(20260101)
    index = rng.integers(0, elements, size=2_000_000)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        # `data[index]` is a gather: latency bound, unlike a sequential sum.
        total = float(data[index].sum())
        elapsed = time.perf_counter() - start
        if total <= 0:
            raise RuntimeError("calibration gather returned no data")
        samples.append(elapsed / index.size * 1e9)
    return _median(samples)


def _triad_gbps(np: Any) -> float:
    """Bandwidth proxy: a[i] = b[i] + 3.0 * c[i] over 64 MiB, GB/s."""
    elements = CALIBRATION_TRIAD_BYTES // 8
    a = np.empty(elements, dtype=np.float64)
    b = np.ones(elements, dtype=np.float64)
    c = np.full(elements, 2.0, dtype=np.float64)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        np.add(b, 3.0 * c, out=a)
        elapsed = time.perf_counter() - start
        # Read b, read c, write a: 3 streams of 8 bytes per element.
        samples.append(elements * 8 * 3 / elapsed / 1e9)
    return _median(samples)


def _matmul_gflops(np: Any) -> float:
    """Sustained-clock proxy: fixed dense matmul, GFLOPS."""
    side = CALIBRATION_MATMUL_SIZE
    rng = np.random.default_rng(20260101)
    left = rng.random((side, side))
    right = rng.random((side, side))
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        np.dot(left, right)
        elapsed = time.perf_counter() - start
        samples.append(2 * side**3 / elapsed / 1e9)
    return _median(samples)


def calibration() -> dict[str, Any]:
    """Run the probes; each is optional so a numpy-less host still records state."""
    result: dict[str, Any] = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import numpy as np  # noqa: PLC0415 - optional, benchmarks-only dependency
    except ImportError:
        result["available"] = False
        result["reason"] = "numpy is not installed"
        return result

    result["available"] = True
    probes = {
        "random_gather_ns_per_access": _random_gather_ns,
        "triad_gbs": _triad_gbps,
        "matmul_gflops": _matmul_gflops,
    }
    for name, probe in probes.items():
        try:
            result[name] = round(float(probe(np)), 4)
        except Exception as error:  # noqa: BLE001 - a probe must never fail a sweep
            result[name] = None
            result.setdefault("probe_errors", {})[name] = str(error)
    return result


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def ledger_path() -> str:
    return os.environ.get("BSDB_BENCH_SESSION_LEDGER", DEFAULT_LEDGER)


def load_ledger(path: str | None = None) -> dict[str, Any]:
    target = path or ledger_path()
    try:
        with open(target, "r", encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"schema": SCHEMA_VERSION, "sessions": {}, "sections": {}}
    ledger.setdefault("schema", SCHEMA_VERSION)
    ledger.setdefault("sessions", {})
    ledger.setdefault("sections", {})
    return ledger


def save_ledger(ledger: dict[str, Any], path: str | None = None) -> str:
    target = path or ledger_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = f"{target}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
    return target


def stamp(
    section: str,
    artifacts: Iterable[str] = (),
    note: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Record that `section` was measured now, with this machine state."""
    ledger = load_ledger(path)
    state = fingerprint()
    session_id = fingerprint_id(state)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    session = ledger["sessions"].setdefault(
        session_id,
        {"fingerprint": state, "first_seen": now, "sections": [], "calibrations": []},
    )
    session["fingerprint"] = state
    session["last_seen"] = now
    if section not in session["sections"]:
        session["sections"].append(section)

    # One calibration sample per stamp: the first and last therefore bracket the
    # sweep, which is what exposes drift (thermal, power-limit, background load)
    # inside a single session as well as between sessions.
    sample = calibration()
    samples = session.setdefault("calibrations", [])
    samples.append(sample)
    del samples[:-100]

    existing = ledger["sections"].get(section, {})
    merged_artifacts = list(dict.fromkeys(list(existing.get("artifacts", [])) + list(artifacts)))
    ledger["sections"][section] = {
        "session": session_id,
        "captured_at": now,
        "artifacts": merged_artifacts,
        "note": note or existing.get("note"),
        "calibration": sample,
    }
    save_ledger(ledger, path)
    return ledger["sections"][section]


def verify(ledger: dict[str, Any] | None = None) -> tuple[str | None, list[tuple[str, str]]]:
    """Return (primary session id, sections measured in a different session).

    The primary session is the one that measured the most sections; ties keep the
    earliest `first_seen`. Unstamped sections are not mismatches: historical
    artifacts predate provenance, and the report says so explicitly.
    """
    ledger = ledger or load_ledger()
    sections = ledger.get("sections", {})
    counts: dict[str, int] = {}
    for entry in sections.values():
        session = entry.get("session")
        if session:
            counts[session] = counts.get(session, 0) + 1
    if not counts:
        return None, []

    sessions = ledger.get("sessions", {})

    def ordering(session_id: str) -> tuple[int, str]:
        return (-counts[session_id], sessions.get(session_id, {}).get("first_seen", ""))

    primary = sorted(counts, key=ordering)[0]
    mismatches = [
        (name, entry["session"])
        for name, entry in sorted(sections.items())
        if entry.get("session") and entry["session"] != primary
    ]
    return primary, mismatches


def describe_fingerprint(fp: dict[str, Any]) -> list[str]:
    """Human-readable machine-state lines for the report header."""
    memory = fp.get("mem_total_bytes")
    lines = [
        f"CPU: {fp.get('cpu_model')} ({fp.get('cpu_cores')} threads)",
        f"Memory: {f'{memory / 2**30:.1f} GiB' if memory else 'unknown'} — "
        f"DIMMs: {fp.get('dimm_config') or 'unknown (needs root: dmidecode -t memory)'}",
        f"Kernel: {fp.get('kernel')} — governor: {fp.get('governor') or 'unknown'}",
    ]
    limits = fp.get("cpu_freq_limits") or {}
    if limits:
        parts = [
            f"driver {limits.get('driver') or 'unknown'}",
            f"policy {limits.get('policy_min_khz')}–{limits.get('policy_max_khz')} kHz",
            f"hw {limits.get('hw_min_khz')}–{limits.get('hw_max_khz')} kHz",
            "boost disabled" if limits.get("boost_disabled") else "boost enabled",
        ]
        lines.append("CPU frequency: " + ", ".join(parts))
    power = fp.get("package_power_limits") or {}
    if power:
        lines.append(
            "Package power limit: "
            f"{power.get('constraint_0_power_limit_w')} W "
            f"(max {power.get('constraint_0_max_power_w')} W)"
        )
    return lines


PROBE_LABELS = (
    ("random_gather_ns_per_access", "random gather (ns/access)"),
    ("triad_gbs", "triad (GB/s)"),
    ("matmul_gflops", "matmul fp64 1024³ (GFLOPS)"),
)


def describe_calibration(cal: dict[str, Any] | None) -> list[tuple[str, str]]:
    """(probe, value) rows for a single calibration sample."""
    cal = cal or {}
    if not cal.get("available"):
        return [("unavailable", cal.get("reason", "not measured"))]
    return [
        (label, "n/a" if cal.get(key) is None else str(cal.get(key)))
        for key, label in PROBE_LABELS
    ]


def describe_calibration_pair(
    samples: list[dict[str, Any]] | None,
) -> list[tuple[str, str, str]]:
    """(probe, first, last) rows: the drift envelope across a sweep."""
    samples = [s for s in (samples or []) if s.get("available")]
    if not samples:
        return []
    first, last = samples[0], samples[-1]
    rows = []
    for key, label in PROBE_LABELS:
        before, after = first.get(key), last.get(key)
        if before is None and after is None:
            continue
        delta = ""
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
            change = (after - before) / before * 100.0
            if abs(change) >= 1.0:
                delta = f" ({change:+.1f}%)"
        rows.append((label, "n/a" if before is None else str(before), "n/a" if after is None else f"{after}{delta}"))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="stamp a measured section with the current state")
    capture.add_argument("--section", required=True)
    capture.add_argument("--artifact", action="append", default=[])
    capture.add_argument("--note")
    capture.add_argument("--ledger")

    verify_cmd = sub.add_parser("verify", help="exit 1 if sections were measured in different sessions")
    verify_cmd.add_argument("--ledger")

    show = sub.add_parser("show", help="print the current state and the recorded sections")
    show.add_argument("--ledger")

    args = parser.parse_args(argv)

    if args.command == "capture":
        entry = stamp(args.section, args.artifact, args.note, args.ledger)
        print(
            f"stamped section '{args.section}' in session {entry['session']} "
            f"({len(entry['artifacts'])} artifact(s))"
        )
        return 0

    if args.command == "verify":
        ledger = load_ledger(args.ledger)
        primary, mismatches = verify(ledger)
        if primary is None:
            print("no sessions recorded (artifacts predate provenance stamping)")
            return 0
        print(f"primary session: {primary}")
        if mismatches:
            print("sections measured in a different session:", file=sys.stderr)
            for name, session in mismatches:
                print(f"  {name}: {session}", file=sys.stderr)
            return 1
        stamped = len(ledger.get("sections", {}))
        print(f"{stamped} section(s), all in the primary session")
        return 0

    ledger = load_ledger(args.ledger)
    primary, mismatches = verify(ledger)
    state = (ledger.get("sessions", {}).get(primary, {}) or {}).get("fingerprint") or fingerprint()
    print(f"session: {primary or 'none recorded'}")
    for line in describe_fingerprint(state):
        print(f"  {line}")
    for name, _value in describe_calibration(
        (ledger.get("sections", {}).get(primary, {}) or {}).get("calibration")
    ):
        print(f"  calibration {name}: {_value}")
    for name, entry in sorted(ledger.get("sections", {}).items()):
        marker = "" if not mismatches or entry.get("session") == primary else "  <-- different session"
        print(f"  section {name}: {entry.get('session')}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
