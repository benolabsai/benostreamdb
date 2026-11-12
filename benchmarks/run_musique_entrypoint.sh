#!/bin/sh
# Entrypoint for the MuSiQue Multi-Hop GraphRAG benchmark container.
#
# Installs the BenoStreamDB abi3 wheel mounted at /wheels, then execs the
# MuSiQue benchmark harness with whatever arguments were passed.
set -eu

if ls /wheels/*.whl >/dev/null 2>&1; then
    echo "[musique-docker] Installing BenoStreamDB wheel from /wheels..."
    python3 -m pip install --break-system-packages --force-reinstall --no-deps --no-index --find-links /wheels benostreamdb
else
    echo "[musique-docker: warn] No wheel found in /wheels; using pre-installed benostreamdb if available" >&2
fi

# The benchmark script is selectable so the same image can run the passage-level
# GraphRAG harness or the entity-level Vector Graph RAG harness without a rebuild.
BENCH_SCRIPT="${BENCH_SCRIPT:-run_musique_graphrag.py}"
exec python3 "/opt/bench/$BENCH_SCRIPT" "$@"

