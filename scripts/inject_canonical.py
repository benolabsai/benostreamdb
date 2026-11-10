#!/usr/bin/env python3
"""Add a canonical URL to every built page.

The same HTML is served from two hosts: www.benostreamdb.org (GitHub Pages,
site root) and www.benolabsai.com/docs/ (BenoLabs, imported copy). Without a
canonical tag search engines index both. This makes the Pages site canonical,
which is also the host the cross-directory documentation links point at.
"""

import pathlib
import re
import sys

BASE = "https://www.benostreamdb.org/"
SKIP_PREFIXES = ("rust/", "_modules/")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inject_canonical.py <build-html-dir>")
        return 2
    out = pathlib.Path(sys.argv[1])
    if not out.is_dir():
        print(f"not a directory: {out}")
        return 2

    count = 0
    for page in out.rglob("*.html"):
        rel = page.relative_to(out).as_posix()
        if rel.startswith(SKIP_PREFIXES):
            continue
        text = page.read_text(errors="ignore")
        if 'rel="canonical"' in text:
            continue
        href = BASE if rel == "index.html" else BASE + rel
        tag = f'<link rel="canonical" href="{href}">'
        updated = re.sub(r"(<head[^>]*>)", r"\1\n" + tag.replace("\\", "\\\\"), text, count=1)
        if updated != text:
            page.write_text(updated)
            count += 1
    print(f"canonical URL injected into {count} page(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
