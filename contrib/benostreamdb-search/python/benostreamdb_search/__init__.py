# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
"""BenoStreamDB Search - OpenSearch and Qdrant REST compatibility gateway."""

import os
import shutil
import subprocess
import sys

__version__ = "0.12.0"


def find_binary() -> str:
    """Locate the bsdb-search binary."""
    # Check adjacent directory in installed wheel
    pkg_dir = os.path.dirname(__file__)
    candidate = os.path.join(pkg_dir, "..", "..", "bin", "bsdb-search")
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # Check PATH
    which_bin = shutil.which("bsdb-search")
    if which_bin:
        return which_bin

    return "bsdb-search"


def main():
    """CLI entrypoint for bsdb-search."""
    binary = find_binary()
    try:
        sys.exit(subprocess.call([binary] + sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
