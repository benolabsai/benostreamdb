#!/usr/bin/env python3
"""Fail on documentation links that cannot work when the site is published.

The published artifact contains only the Sphinx output (plus rustdoc under
/rust/). Repository files and pages in another documentation directory are not
reachable with a relative path from a flattened page, so the rules are:

1. A relative link must resolve on disk (case-sensitive, like GitHub).
2. A relative link may only target a file in the same directory as the source
   document. Anything crossing a directory boundary must be an absolute URL:
   the docs site for a documentation page, a GitHub blob URL for source.

This is enforced because `myst.xref_missing` is suppressed in conf.py, so a
broken link produces a green build and an inert span instead of a failure.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "docs", ROOT / "plans", ROOT / "contrib" / "benostreamdb-search" / "docs"]
EXTRA = [ROOT / "README.md"]
SKIP = ("http://", "https://", "mailto:", "#", "file://")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")


def documents():
    for base in SOURCES:
        if not base.exists():
            continue
        for p in base.rglob("*.md"):
            if "docs/source" in p.as_posix():
                continue
            yield p
    for p in EXTRA:
        if p.exists():
            yield p


def main() -> int:
    errors = []
    for p in documents():
        try:
            text = p.read_text(errors="ignore")
        except OSError as exc:
            errors.append(f"{p.relative_to(ROOT)}: unreadable ({exc})")
            continue
        for match in LINK.finditer(text):
            target = match.group(1)
            if target.startswith(SKIP):
                continue
            path_part = LINE_SUFFIX.sub("", target.split("#")[0])
            if not path_part:
                continue
            try:
                resolved = (p.parent / path_part).resolve()
            except OSError:
                continue
            if not resolved.exists():
                errors.append(f"{p.relative_to(ROOT)}: broken link -> {target}")
                continue
            if resolved.parent != p.resolve().parent:
                errors.append(
                    f"{p.relative_to(ROOT)}: cross-directory relative link -> {target} "
                    "(use an absolute docs-site or GitHub URL)"
                )

    for e in errors:
        print(e)
    print(f"{len(errors)} documentation link problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
