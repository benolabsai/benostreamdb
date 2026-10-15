#!/usr/bin/env python3
"""Teardown helper: undo a published BenoStreamDB release, version by version.

Given an engine version (e.g. ``0.12.0`` or ``v0.12.0``) this walks *every*
distribution the repository publishes and undoes it:

* **PyPI**     -> *delete* the release via ``pypi-cleanup``. PyPI has no yank
                 API or CLI, so delete is the only scriptable action. Deleting
                 also frees the version number, so the same version can be
                 re-published cleanly afterwards.
* **crates.io** -> *yank* via ``cargo yank``. A crates.io version can never be
                  deleted, so a yank is the strongest available undo (it is
                  reversible with ``cargo yank --undo``).

Nothing destructive happens unless ``--apply`` is passed. The default mode is a
read-only plan which also queries both registries to report whether each
release actually exists, so you never touch something that isn't there.

Two hard limits of the underlying tools, handled explicitly
----------------------------------------------------------
1. ``pypi-cleanup`` **refuses to delete a project's last remaining release**,
   because that would leave the project with no versions. It cannot be
   overridden except with ``--delete-project`` (which destroys the name and its
   trusted-publisher configuration). Those cases are detected up front and
   reported as a MANUAL step that must be done with the PyPI web UI, which
   *does* allow deleting the only release while keeping the project.
2. ``pypi-cleanup`` authenticates via ``getpass()``, which hijacks the
   controlling terminal and ignores piped stdin. This script therefore passes
   the token through ``PYPI_CLEANUP_PASSWORD``, which the tool reads first.

Version mapping
---------------
Two distributions carry an INDEPENDENT version line and are resolved from the
repository rather than from the engine tag:

* ``dbt-benostreamdb``   <- dbt-benostreamdb/dbt/adapters/benostreamdb/__version__.py
* ``benostream-gpu-ann`` <- benostream-gpu-ann/Cargo.toml

Override with ``--dbt-version`` / ``--gpu-ann-version``, or drop them from the
run with ``--no-independent``.

Authentication
--------------
* ``PYPI_API_TOKEN``       - PyPI API token. Used for the ``twine upload``
                             bootstrap of NEW project names, which must be
                             ACCOUNT-scoped (a project-scoped token cannot
                             target a project that does not exist yet).
                             IMPORTANT: an API token CANNOT log in to the PyPI
                             *website*, and pypi-cleanup deletes through that
                             web login -- verified: the upload API accepts the
                             token (HTTP 400 on a probe) while
                             POST /account/login/ rejects it ("Error processing
                             form"). A token alone therefore cannot delete a
                             release.
* ``PYPI_PASSWORD``        - your real PyPI account password. THIS is what
                             pypi-cleanup needs in order to log in and delete.
* ``PYPI_TOTP_SECRET``     - optional base32 TOTP seed. When set, the current
                             6-digit code is fed to pypi-cleanup's 2FA prompt so
                             the flow stays non-interactive (RFC 6238, no extra
                             dependency).
* ``CARGO_REGISTRY_TOKEN`` - crates.io token. Required to yank.

Nothing is needed for a dry run.

Usage
-----
    # plan only (no credentials needed)
    python scripts/release_teardown.py --version 0.12.0

    # do it (0.12.0 engine releases only)
    PYPI_API_TOKEN=pypi-xxxx CARGO_REGISTRY_TOKEN=yyyy \\
        python scripts/release_teardown.py --version v0.12.0 --no-independent --apply

    # one registry only
    python scripts/release_teardown.py -v 0.12.0 --pypi-only  --apply
    python scripts/release_teardown.py -v 0.12.0 --crates-only --apply
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent

# PyPI distribution names this repository publishes. Kept in sync with the
# `[project] name` fields under pyproject.toml / contrib / server / dbt-benostreamdb.
PYPI_DISTS = [
    "benostreamdb",         # engine (maturin / PyO3)
    "benostreamdb-search",  # contrib/benostreamdb-search
    "benostreamdb-flight",  # server/flight_sql
    "benostreamdb-mcp",     # contrib/benostreamdb-mcp
    "dbt-benostreamdb",     # INDEPENDENT version (dbt convention)
]

# crates.io crate names. benostream-gpu-ann is versioned independently.
CRATES = [
    "benostream-gpu-ann",   # INDEPENDENT version
    "benostreamdb",
    "benostreamdb-search",
]

DBT_VERSION_FILE = REPO_ROOT / "dbt-benostreamdb" / "dbt" / "adapters" / "benostreamdb" / "__version__.py"
GPU_ANN_CARGO = REPO_ROOT / "benostream-gpu-ann" / "Cargo.toml"

USER_AGENT = "beno-release-teardown (+https://github.com/benolabsai/benostreamdb)"


# --------------------------------------------------------------------------- #
# Small HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #

def _get(url: str, accept: str = "application/json") -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": accept,
        # PyPI's CDN can still serve a stale 200 for a version that was just
        # deleted, which makes a completed teardown look like a no-op. Ask
        # intermediaries to revalidate rather than trust their copy.
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception as exc:  # network hiccup: report, don't crash the plan
        print(f"  ! network error for {url}: {exc}", file=sys.stderr)
        return 0, b""


def _get_json(url: str) -> dict | None:
    status, body = _get(url)
    if status != 200 or not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def pypi_release_exists(dist: str, version: str) -> bool:
    """True when the exact version exists on PyPI (404 => all files deleted)."""
    status, _ = _get(f"https://pypi.org/pypi/{dist}/{version}/json")
    return status == 200


def pypi_project_status(dist: str) -> int:
    status, _ = _get(f"https://pypi.org/pypi/{dist}/json")
    return status


def pypi_all_versions(dist: str) -> set[str]:
    """Every version PyPI knows for this project (empty set if unknown)."""
    data = _get_json(f"https://pypi.org/pypi/{dist}/json")
    if not data:
        return set()
    return set((data.get("releases") or {}).keys())


def is_last_release(dist: str, version: str) -> bool:
    """True when deleting ``version`` would leave the project with no releases."""
    versions = pypi_all_versions(dist)
    return bool(versions) and versions == {version}


def crate_version_state(crate: str, version: str) -> str:
    """'missing' | 'yanked' | 'live' — queried straight from crates.io."""
    data = _get_json(f"https://crates.io/api/v1/crates/{crate}/{version}")
    if not data or "version" not in data:
        return "missing"
    return "yanked" if data["version"].get("yanked") else "live"


def pypi_release_url(dist: str, version: str) -> str:
    return f"https://pypi.org/manage/project/{dist}/release/{version}/"


# --------------------------------------------------------------------------- #
# Version resolution
# --------------------------------------------------------------------------- #

def normalize_version(raw: str) -> str:
    return raw.strip().lstrip("vV")


def _grep_version(path: Path, pattern: str) -> str | None:
    if not path.is_file():
        return None
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def read_dbt_version() -> str | None:
    return _grep_version(DBT_VERSION_FILE, r'version\s*=\s*"([^"]+)"')


def read_gpu_ann_version() -> str | None:
    return _grep_version(GPU_ANN_CARGO, r'(?m)^version\s*=\s*"([^"]+)"')


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #

def ensure_pypi_cleanup(apply: bool) -> str | None:
    """Return the pypi-cleanup executable, installing it when --apply is set."""
    exe = shutil.which("pypi-cleanup")
    if exe:
        return exe
    if not apply:
        return None
    print("  installing pypi-cleanup ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "pypi-cleanup"],
        check=False,
    )
    return shutil.which("pypi-cleanup")


def totp_now(secret: str) -> str:
    """RFC 6238 TOTP (SHA-1, 6 digits, 30s step) with no third-party dependency."""
    cleaned = re.sub(r"[\s-]", "", secret).upper()
    pad = "=" * ((8 - len(cleaned) % 8) % 8)
    key = base64.b32decode(cleaned + pad)
    counter = struct.pack(">Q", int(time.time()) // 30)
    digest = hmac.new(key, counter, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return f"{code % 1_000_000:06d}"


def pypi_delete(dist: str, version: str, username: str, password: str,
                totp_secret: str | None, exe: str) -> bool:
    """Delete exactly ``version`` of ``dist`` via pypi-cleanup.

    pypi-cleanup deletes by logging in to the PyPI *website*, which does NOT
    accept API tokens -- only the real account password (plus a 2FA code when
    enabled). Two details make that work non-interactively:

    * the password goes through ``PYPI_CLEANUP_PASSWORD`` (read before getpass),
      because getpass() would otherwise hijack the controlling terminal and
      ignore anything piped to stdin;
    * when a TOTP seed is supplied, the current code is piped to stdin, which is
      where pypi-cleanup's ``input("Authentication code: ")`` reads from.
    """
    version_re = "^" + re.escape(version) + "$"
    cmd = [exe, "-u", username, "-p", dist, "-r", version_re, "--do-it", "-y"]
    env = {**os.environ, "PYPI_CLEANUP_PASSWORD": password}
    stdin_text = f"{totp_now(totp_secret)}\n" if totp_secret else None
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=300,
                              env=env, input=stdin_text)
    except subprocess.TimeoutExpired:
        print(f"  ! pypi-cleanup timed out for {dist} {version}")
        return False

    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return True

    print(f"  ! pypi-cleanup failed for {dist} {version} (exit {proc.returncode})")
    if "ALL AVAILABLE RELEASED VERSIONS" in output:
        print("    refused: this is the project's ONLY release (see MANUAL ACTION below)")
    elif "Login for user" in output:
        print("    login rejected. pypi-cleanup needs the REAL PyPI PASSWORD")
        print("    (PYPI_PASSWORD) -- an API token cannot log in to the website.")
        if not totp_secret:
            print("    if the account has 2FA, also set PYPI_TOTP_SECRET.")
    for line in output.strip().splitlines()[-10:]:
        print(f"      {line}")
    print(f"    manual fallback: {pypi_release_url(dist, version)}")
    return False


def cargo_yank(crate: str, version: str, token: str) -> bool:
    """Yank a crate version. `cargo yank` takes the crate POSITIONALLY —
    there is no `-p/--package` flag (confirmed on cargo 1.93.0)."""
    cmd = ["cargo", "yank", "--version", version, "--registry", "crates-io",
           "--token", token, crate]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"  ! cargo yank timed out for {crate} {version}")
        return False
    if proc.returncode == 0:
        return True
    print(f"  ! cargo yank failed for {crate} {version} (exit {proc.returncode})")
    for line in ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-12:]:
        print(f"      {line}")
    return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_plan(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Return (registry, name, version) targets, in teardown order."""
    engine = normalize_version(args.version)
    targets: list[tuple[str, str, str]] = []

    if not args.crates_only:
        for dist in PYPI_DISTS:
            if dist == "dbt-benostreamdb":
                if args.no_independent:
                    continue
                version = normalize_version(args.dbt_version or read_dbt_version() or "")
                if not version:
                    print("  ~ skipping dbt-benostreamdb (could not resolve its version; "
                          "pass --dbt-version)", file=sys.stderr)
                    continue
            else:
                version = engine
            targets.append(("pypi", dist, version))

    if not args.pypi_only:
        # Reverse dependency order: dependents first, core last.
        for crate in CRATES:
            if crate == "benostream-gpu-ann":
                if args.no_independent:
                    continue
                version = normalize_version(args.gpu_ann_version or read_gpu_ann_version() or "")
                if not version:
                    print("  ~ skipping benostream-gpu-ann (could not resolve its version; "
                          "pass --gpu-ann-version)", file=sys.stderr)
                    continue
            else:
                version = engine
            targets.append(("crates", crate, version))

    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete (PyPI) and yank (crates.io) a BenoStreamDB release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", "-v", required=True,
                        help="engine version to tear down, e.g. 0.12.0 or v0.12.0")
    parser.add_argument("--apply", action="store_true",
                        help="actually perform the deletes/yanks (default: plan only)")
    parser.add_argument("--pypi-only", action="store_true", help="skip crates.io")
    parser.add_argument("--crates-only", action="store_true", help="skip PyPI")
    parser.add_argument("--no-independent", action="store_true",
                        help="skip dbt-benostreamdb and benostream-gpu-ann, which "
                             "carry their own version lines")
    parser.add_argument("--dbt-version", help="override dbt-benostreamdb's version")
    parser.add_argument("--gpu-ann-version", help="override benostream-gpu-ann's version")
    args = parser.parse_args(argv)

    if args.pypi_only and args.crates_only:
        parser.error("--pypi-only and --crates-only are mutually exclusive")

    engine = normalize_version(args.version)
    mode = "APPLY (destructive)" if args.apply else "PLAN (read-only)"
    print(f"release teardown — {mode}")
    print(f"engine version: {engine}")
    print()

    targets = build_plan(args)

    # ---- plan + preflight ------------------------------------------------- #
    print(f"{'registry':<8} {'name':<22} {'version':<10} state")
    print("-" * 68)
    work: list[tuple[str, str, str]] = []
    manual: list[tuple[str, str, str]] = []
    for registry, name, version in targets:
        if registry == "pypi":
            if not pypi_release_exists(name, version):
                state = ("absent (project does not exist)"
                         if pypi_project_status(name) == 404 else "absent (no such version)")
            elif is_last_release(name, version):
                state = "present (ONLY release -> web UI)"
                manual.append((registry, name, version))
            else:
                state = "present"
                work.append((registry, name, version))
        else:
            state = crate_version_state(name, version)
            if state == "live":
                work.append((registry, name, version))
            elif state == "yanked":
                state = "already yanked"
        print(f"{registry:<8} {name:<22} {version:<10} {state}")

    print()
    if not work and not manual:
        print("Nothing to do — every target is already absent or yanked.")
        return 0

    if work:
        print(f"{len(work)} target(s) can be automated:")
        for registry, name, version in work:
            if registry == "pypi":
                print(f"  DELETE pypi:{name}=={version}")
            else:
                print(f"  YANK   crates.io:{name}@{version}")
        print()

    if manual:
        # pypi-cleanup refuses to delete a project's last release, and
        # --delete-project would destroy the name + trusted-publisher config.
        print(f"{len(manual)} target(s) REQUIRE THE WEB UI (pypi-cleanup cannot "
              f"delete a project's only release):")
        for _, name, version in manual:
            print(f"  MANUAL pypi:{name}=={version}  ->  {pypi_release_url(name, version)}")
        print()

    if not args.apply:
        print("Re-run with --apply to perform the automatable actions.")
        return 0

    # ---- credentials ------------------------------------------------------ #
    pypi_password = os.environ.get("PYPI_PASSWORD", "").strip()
    pypi_api_token = os.environ.get("PYPI_API_TOKEN", "").strip()
    pypi_username = os.environ.get("PYPI_USERNAME", "").strip() or "__token__"
    pypi_totp = os.environ.get("PYPI_TOTP_SECRET", "").strip() or None
    cargo_token = os.environ.get("CARGO_REGISTRY_TOKEN", "").strip()

    pypi_exe = None
    if any(r == "pypi" for r, _, _ in work):
        if not pypi_password:
            print("error: PYPI_PASSWORD is not set - skipping PyPI deletes.", file=sys.stderr)
            if pypi_api_token:
                print("       An API token cannot log in to the PyPI website, which is how",
                      file=sys.stderr)
                print("       pypi-cleanup deletes. Use the web UI links, or set PYPI_PASSWORD.",
                      file=sys.stderr)
            work = [(r, n, v) for r, n, v in work if r != "pypi"]
        else:
            if pypi_username == "__token__":
                print("note: set PYPI_USERNAME to your real PyPI username when using a password")
            pypi_exe = ensure_pypi_cleanup(args.apply)
            if not pypi_exe:
                print("error: pypi-cleanup is unavailable (pip install pypi-cleanup).",
                      file=sys.stderr)
                work = [(r, n, v) for r, n, v in work if r != "pypi"]

    if any(r == "crates" for r, _, _ in work) and not cargo_token:
        print("error: CARGO_REGISTRY_TOKEN is not set — skipping crates.io yanks.",
              file=sys.stderr)
        work = [(r, n, v) for r, n, v in work if r != "crates"]

    # ---- execute ---------------------------------------------------------- #
    failures = 0
    for registry, name, version in work:
        if registry == "pypi":
            print(f"deleting pypi:{name}=={version} ...")
            ok = pypi_delete(name, version, pypi_username, pypi_password, pypi_totp,
                             pypi_exe or "pypi-cleanup")
        else:
            print(f"yanking crates.io:{name}@{version} ...")
            ok = cargo_yank(name, version, cargo_token)
        print("  ok" if ok else "  FAILED")
        if not ok:
            failures += 1

    # ---- verify ----------------------------------------------------------- #
    print()
    print("verification:")
    print(f"{'registry':<8} {'name':<22} {'version':<10} state")
    print("-" * 68)
    for registry, name, version in work:
        if registry == "pypi":
            state = "still present" if pypi_release_exists(name, version) else "gone"
        else:
            state = crate_version_state(name, version)
        print(f"{registry:<8} {name:<22} {version:<10} {state}")

    if manual:
        print()
        print("MANUAL ACTION REQUIRED — open the link(s) and click 'Delete release':")
        for _, name, version in manual:
            print(f"  {pypi_release_url(name, version)}")
        print("  (the web UI allows deleting the last release; the project name and its")
        print("   trusted-publisher config are retained)")

    print()
    if failures:
        print(f"{failures} automated action(s) failed — see the output above.")
        return 1
    print("Teardown finished (see MANUAL ACTION REQUIRED above, if any).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
