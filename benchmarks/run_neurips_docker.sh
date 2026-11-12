#!/bin/sh
# Entrypoint for the NeurIPS BigANN harness image.
#
# Installs the BenoStreamDB abi3 wheel if one is mounted at /wheels (so the image
# stays engine-agnostic), then execs the harness with whatever arguments compose
# passed. `--hardware d8lds --strict-hardware` makes the run fail loudly if the
# container is not actually constrained to the evaluation envelope.
set -eu

if ls /wheels/*.whl >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages --force-reinstall --no-deps --no-index --find-links /wheels benostreamdb
else
    echo "[warn] no wheel found in /wheels; using whatever benostreamdb is installed" >&2
fi

BENCH_SCRIPT="${BENCH_SCRIPT:-/opt/bench/run_neurips_bigann.py}"
exec python3 "$BENCH_SCRIPT" "$@"
