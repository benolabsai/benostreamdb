#!/usr/bin/env python3
"""AI-assisted security review of a pull request / branch diff.

A **local** tool: it reads the standard OpenAI-compatible environment variables
from your shell (no CI secrets required) and asks an OpenRouter
chat-completions endpoint to look for *injected* vulnerabilities — backdoors,
weakened or removed checks, credential exfiltration, unsafe deserialization,
command/SQL/path injection, disabled security controls, and supply-chain
changes.

Usage (local)::

    export OPENAI_API_KEY=sk-or-...            # your OpenRouter key
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # default
    export OPENAI_MODEL=openai/gpt-4o-mini     # default
    python scripts/ai_pr_review.py             # reviews `git diff origin/main...HEAD`

    # or review a specific diff / PR:
    PR_DIFF_FILE=/tmp/pr.diff python scripts/ai_pr_review.py
    GITHUB_REPOSITORY=owner/repo PR_NUMBER=123 GITHUB_TOKEN=... \
        python scripts/ai_pr_review.py

Environment:
  - ``OPENAI_API_KEY``  required to run
  - ``OPENAI_BASE_URL`` default ``https://openrouter.ai/api/v1``
  - ``OPENAI_MODEL``    default ``openai/gpt-4o-mini`` (cheap; raise for depth)
  - ``OPENAI_SITE_URL`` / ``OPENAI_APP_NAME`` optional OpenRouter attribution
  - ``PR_DIFF_FILE``    read the diff from a file instead of git/GitHub
  - ``PR_BASE_REF``     base ref for the local diff (default ``origin/main``)
  - ``GITHUB_REPOSITORY`` / ``PR_NUMBER`` / ``GITHUB_TOKEN``  optional: fetch the
    PR diff and post the findings as a PR comment

Exit codes: 0 = clean, 1 = high-severity finding or an error.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Keep the prompt within a cheap token budget; the diff is truncated to fit.
MAX_DIFF_CHARS = 60_000

SYSTEM_PROMPT = """You are a security reviewer for a Rust/Arrow/Iceberg database.
Review the pull request diff ONLY for security-relevant changes, especially
deliberately injected vulnerabilities:
- backdoors, hidden network calls, telemetry or data exfiltration
- weakened or removed authentication, authorization, or input validation
- disabled security controls (TLS verification, signature checks, sandboxing)
- command/SQL/path injection, unsafe deserialization, unbounded allocation
- hardcoded secrets, credentials, tokens, or suspicious URLs
- supply-chain changes (new dependencies, build scripts, CI permission changes)
Ignore style, formatting, and non-security refactors.
Respond with a JSON object of the form:
{"findings": [{"severity": "high|medium|low", "file": "...", "line": "...",
"issue": "...", "recommendation": "..."}], "summary": "..."}.
If nothing security-relevant changed, return
{"findings": [], "summary": "No security-relevant changes."}."""


def _request(method: str, url: str, *, token: str | None = None,
             data: bytes | None = None, accept: str = "application/vnd.github+json",
             extra_headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, method=method, data=data)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", accept)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (extra_headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _fetch_pr_diff(repo: str, pr: str, token: str) -> str:
    raw = _request(
        "GET",
        f"https://api.github.com/repos/{repo}/pulls/{pr}",
        token=token,
        accept="application/vnd.github.v3.diff",
    )
    return raw.decode("utf-8", "replace")


def _local_diff() -> str:
    """Diff the current branch against ``PR_BASE_REF`` (default ``origin/main``)."""
    base = os.environ.get("PR_BASE_REF", "origin/main")
    for args in (["git", "diff", f"{base}...HEAD"], ["git", "diff", "HEAD~1"]):
        try:
            out = subprocess.run(args, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if out.stdout.strip():
            return out.stdout
    return ""


def _review(diff: str, api_base: str, model: str, api_key: str) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"PR diff:\n\n{diff}"},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    headers = {"Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in api_base:
        # OpenRouter uses these for attribution/rankings; both are optional.
        headers["HTTP-Referer"] = os.environ.get(
            "OPENAI_SITE_URL", "https://github.com/benostreamdb"
        )
        headers["X-Title"] = os.environ.get("OPENAI_APP_NAME", "BenoStreamDB PR Review")
    raw = _request(
        "POST",
        f"{api_base}/chat/completions",
        data=payload,
        extra_headers=headers,
    )
    body = json.loads(raw)
    content = body["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"findings": [], "summary": content}


def _format_comment(result: dict) -> str:
    findings = result.get("findings", [])
    summary = result.get("summary", "")
    if not findings:
        return f"### 🤖 AI security review\n\n{summary or 'No security-relevant changes.'}"
    lines = ["### 🤖 AI security review", "", summary, ""]
    for f in findings:
        sev = str(f.get("severity", "?")).upper()
        loc = f"{f.get('file', '?')}:{f.get('line', '?')}"
        lines.append(f"- **[{sev}]** `{loc}` — {f.get('issue', '')}")
        if f.get("recommendation"):
            lines.append(f"  - _Recommendation:_ {f['recommendation']}")
    lines.append("")
    lines.append("_AI-generated; verify before acting. Not a substitute for review._")
    return "\n".join(lines)


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set — nothing to do.", file=sys.stderr)
        return 1

    api_base = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini")

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr = os.environ.get("PR_NUMBER")
    diff_file = os.environ.get("PR_DIFF_FILE")

    if diff_file:
        diff = Path(diff_file).read_text(encoding="utf-8", errors="replace")
    elif repo and pr and token:
        diff = _fetch_pr_diff(repo, pr, token)
    else:
        diff = _local_diff()

    if not diff.strip():
        print("Empty diff — nothing to review.")
        return 0
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated] ..."

    try:
        result = _review(diff, api_base, model, api_key)
    except urllib.error.HTTPError as e:
        print(f"LLM request failed: {e.code} {e.read()[:500]!r}", file=sys.stderr)
        return 1

    comment = _format_comment(result)
    if token and repo and pr:
        _request(
            "POST",
            f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
            token=token,
            data=json.dumps({"body": comment}).encode(),
        )
        print("Posted review comment to the PR.")
    print(comment)

    high = [f for f in result.get("findings", []) if str(f.get("severity", "")).lower() == "high"]
    if high:
        print(f"\n{len(high)} high-severity finding(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())