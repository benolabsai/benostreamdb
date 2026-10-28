#!/usr/bin/env python3
"""Run the whole benchmark sweep that `BENCHMARK_REPORT.md` summarizes.

A report is only readable when every section in it was measured in the same
machine state (see `docs/BENCHMARKING.md` §1.1 and `session_provenance.py`). This
driver enforces that workflow:

1. capture the session fingerprint and calibration samples;
2. run each section's command, streaming its output;
3. stamp the section with the artifacts it produced;
4. regenerate the report, which **refuses** to build if any stamped section came
   from a different session.

Sections and their commands are explicit and printed by `--dry-run`. Nothing is
inferred: if a harness changes its flags, override it on the command line rather
than editing the sequence in place, so the change is visible in the commit.

Usage
-----
    python benchmarks/run_sweep.py --dry-run
    python benchmarks/run_sweep.py --cpus 8 --mem 16g
    python benchmarks/run_sweep.py --only vector_matrix,graph
    python benchmarks/run_sweep.py --cmd 'vector_ann=python benchmarks/ann_benchmarks/run.py --datasets glove-100-angular'

Rough wall clock on the reference host (8 CPU / 16 GB envelope): GPU cdist ~2 min,
vector ANN self-characterization ~20 min per dataset, competitor vector matrix
~15 min per dataset (eight datasets, CPU+GPU), graph/sql/BEIR ~10 min each,
filters ~10 min, production/soak ~30 min. A full sweep is a multi-hour,
unattended run — that is the point of the "one session" rule.
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

try:
    import session_provenance
except ImportError:  # imported as `benchmarks.run_sweep`
    from benchmarks import session_provenance  # type: ignore[no-redef]

VECTOR_DATASETS = (
    "sift-128-euclidean",
    "glove-100-angular",
    "glove-200-angular",
    "nytimes-256-angular",
    "lastfm-64-dot",
    "fashion-mnist-784-euclidean",
    "mnist-784-euclidean",
    "gist-960-euclidean",
)

GRAPH_ALGORITHMS = ("pagerank", "connected_components", "shortest_path")


@dataclass
class Section:
    """One report section: how to measure it and what it writes."""

    name: str
    commands: list[list[str]]
    artifacts: list[str] = field(default_factory=list)

    def resolved(self) -> list[str]:
        return [" ".join(shlex.quote(part) for part in command) for command in self.commands]


def _docker_bench() -> str:
    return os.path.join("benchmarks", "competitors", "docker_bench.sh")


def build_sections(cpus: str, mem: str, envelope: list[str] | None = None) -> list[Section]:
    """The sweep, in report order.

    `envelope` is `--cpus`/`--mem`; it is passed to every containerized section so
    all engines run under the same limits (the like-for-like rule).
    """
    env = envelope or ["--cpus", cpus, "--mem", mem]
    docker = _docker_bench()

    vector_commands = [
        [docker, "--both", *env, "--dataset", dataset, "--limit", "20000", "--queries", "500"]
        for dataset in VECTOR_DATASETS
    ]
    graph_commands = [
        [
            docker,
            "--workload",
            "graph",
            *env,
            "--graph-engines",
            "networkx neo4j cugraph benostreamdb",
            "--algorithm",
            algorithm,
        ]
        for algorithm in GRAPH_ALGORITHMS
    ]

    return [
        Section(
            "gpu_cdist",
            [[sys.executable, "benchmarks/gpu_cdist_bench.py"]],
            ["results/gpu_cdist.json"],
        ),
        Section(
            "vector_ann",
            [[sys.executable, "benchmarks/ann_benchmarks/run.py", "--limit", "20000", "--queries", "500"]],
            ["ann_benchmarks/results/*"],
        ),
        Section("vector_matrix", vector_commands, ["competitors/results/*.json"]),
        Section("graph", graph_commands, ["graph/results/*"]),
        Section(
            "sql",
            [[docker, "--workload", "sql", *env, "--sql-engines",
              "duckdb datafusion clickhouse benostreamdb", "--sql-rows", "500000"]],
            ["sql/results/*"],
        ),
        Section(
            "beir",
            [[sys.executable, "benchmarks/beir/run.py"]],
            ["beir/results/*"],
        ),
        Section(
            "filters",
            [[sys.executable, "benchmarks/filters/run.py"]],
            ["results/production_filters.*"],
        ),
        Section(
            "production",
            # Both workloads: `concurrent` writes the QPS/latency ladder (with
            # `--sweep` for the concurrency series), `maintenance` writes the
            # compaction/vacuum record. The old single command omitted the required
            # `--workload` flag and exited 2 without measuring anything.
            [
                [sys.executable, "benchmarks/production_workload.py",
                 "--workload", "concurrent", "--sweep"],
                [sys.executable, "benchmarks/production_workload.py",
                 "--workload", "maintenance"],
            ],
            ["results/production_*"],
        ),
    ]


def _expand_artifacts(patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(os.path.join(BASE_DIR, pattern))))
    return [os.path.relpath(path, BASE_DIR) for path in found]


def _fresh_artifacts(patterns: list[str], started: float, tolerance: float = 5.0) -> list[str]:
    """Artifacts the section actually wrote, i.e. modified since it started.

    Stamping a section whose command exited 0 but wrote nothing is worse than
    failing it: the report would claim those numbers came from this session while
    they are the previous run's. Filesystem mtime is the only signal the harness
    exposes, so every produced artifact must be newer than the section start
    (minus a small tolerance for coarse timestamps).
    """
    fresh: list[str] = []
    for relative in _expand_artifacts(patterns):
        try:
            modified = os.path.getmtime(os.path.join(BASE_DIR, relative))
        except OSError:
            continue
        if modified >= started - tolerance:
            fresh.append(relative)
    return fresh


def run_sections(
    sections: list[Section],
    keep_going: bool,
    dry_run: bool,
) -> int:
    failures: list[tuple[str, int]] = []
    for index, section in enumerate(sections, start=1):
        print(f"\n=== [{index}/{len(sections)}] {section.name}", flush=True)
        for command in section.resolved():
            print(f"    $ {command}", flush=True)
        if dry_run:
            continue

        started = time.time()
        section_failed = False
        for command in section.commands:
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if completed.returncode != 0:
                section_failed = True
                failures.append((section.name, completed.returncode))
                break
        elapsed = time.time() - started

        if section_failed:
            print(f"    {section.name}: FAILED after {elapsed:.0f}s", flush=True)
            if not keep_going:
                return 1
            continue

        fresh = _fresh_artifacts(section.artifacts, started)
        if not fresh:
            # Exit 0 with no new artifact means the command did nothing useful
            # (wrong flags, missing engine, output path the report does not read).
            # Never stamp that: it would attribute the previous run's files to this
            # session, which is exactly the splice the guard exists to prevent.
            failures.append((section.name, -1))
            print(
                f"    {section.name}: NO FRESH ARTIFACTS after {elapsed:.0f}s "
                f"(looked for {', '.join(section.artifacts)}) — not stamped",
                flush=True,
            )
            if not keep_going:
                return 1
            continue

        entry = session_provenance.stamp(section.name, fresh)
        print(
            f"    {section.name}: ok in {elapsed:.0f}s, stamped session "
            f"{entry['session']} with {len(fresh)} artifact(s)",
            flush=True,
        )

    if failures:
        print("\nfailed sections:", file=sys.stderr)
        for name, code in failures:
            print(f"  {name} (exit {code})", file=sys.stderr)
        return 1
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full benchmark sweep and regenerate the report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cpus", default="8", help="container CPU envelope (default 8)")
    parser.add_argument("--mem", default="16g", help="container memory envelope (default 16g)")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated section names (default: every section)",
    )
    parser.add_argument(
        "--cmd",
        action="append",
        default=[],
        metavar="SECTION=COMMAND",
        help="override a section's command(s); repeatable, use '&&' to chain",
    )
    parser.add_argument("--keep-going", action="store_true", help="continue after a failing section")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="stamp the sections but do not regenerate BENCHMARK_REPORT.md",
    )
    parser.add_argument(
        "--allow-mixed-sessions",
        action="store_true",
        help="regenerate the report even if sections span sessions (historical only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    sections = build_sections(args.cpus, args.mem)

    for override in args.cmd:
        name, _, command = override.partition("=")
        if not command:
            print(f"--cmd expects SECTION=COMMAND, got {override!r}", file=sys.stderr)
            return 2
        chained = [shlex.split(part) for part in command.split("&&") if part.strip()]
        for section in sections:
            if section.name == name:
                section.commands = chained
                break
        else:
            print(f"unknown section in --cmd: {name}", file=sys.stderr)
            return 2

    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - {section.name for section in sections}
        if unknown:
            print(f"unknown section(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        sections = [section for section in sections if section.name in wanted]

    if args.dry_run:
        state = session_provenance.fingerprint()
        print(f"session: {session_provenance.fingerprint_id(state)}")
        for line in session_provenance.describe_fingerprint(state):
            print(f"  {line}")
        print()

    code = run_sections(sections, keep_going=args.keep_going, dry_run=args.dry_run)
    if code != 0 or args.dry_run or args.no_report:
        return code

    print("\n=== regenerating the report", flush=True)
    completed = subprocess.run(
        [sys.executable, "benchmarks/generate_summary.py"]
        + (["--allow-mixed-sessions"] if args.allow_mixed_sessions else []),
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
