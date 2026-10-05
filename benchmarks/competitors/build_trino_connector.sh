#!/usr/bin/env bash
# Build the BenoStreamDB Trino connector image for the latest Trino container.
#
# The connector needs two artifacts:
#   1. the plugin ZIP (`connector-artifacts/trino-benostream-java-17.zip`), and
#   2. `libbenostreamdb.so` on `java.library.path`.
#
# The native lib must be built for the *Trino image's* glibc (Debian, ~2.34),
# not the host's. A host `cargo build` links against the host glibc and fails to
# load with `version 'GLIBC_2.4x' not found`. So we extract the `.so` from the
# manylinux wheel (built with `maturin --zig --compatibility manylinux_2_28`).
#
# Usage:
#   benchmarks/competitors/build_trino_connector.sh [--build-wheel]
#
#   --build-wheel   Rebuild the manylinux wheel first (slow).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ART="$REPO/connector-artifacts"

BUILD_WHEEL=0
[[ "${1:-}" == "--build-wheel" ]] && BUILD_WHEEL=1

if [[ "$BUILD_WHEEL" == "1" ]]; then
  echo "building manylinux_2_28 wheel ..."
  ( cd "$REPO" && maturin build --release --zig --compatibility manylinux_2_28 -o dist )
fi

# 1. Plugin ZIP.
if [[ ! -f "$ART/trino-benostream-java-17.zip" ]]; then
  echo "error: $ART/trino-benostream-java-17.zip not found." >&2
  echo "Build it with: (cd trino-benostreamdb && mvn clean package -DskipTests)" >&2
  exit 1
fi

# 2. Native lib, extracted from the manylinux wheel.
wheel="$(ls -1 "$REPO"/dist/benostreamdb-*-manylinux_2_28_x86_64.whl 2>/dev/null | head -1 || true)"
if [[ -z "$wheel" ]]; then
  echo "error: no manylinux_2_28 wheel in dist/. Run with --build-wheel." >&2
  exit 1
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
( cd "$tmp" && unzip -q "$wheel" )
so="$(find "$tmp" -name '*.so' | head -1)"
cp "$so" "$ART/libbenostreamdb.so"
echo "staged native lib: $ART/libbenostreamdb.so"

# 3. Build the Trino image.
docker compose -f "$HERE/docker-compose.bench.yml" build trino
echo "built bsdb-trino:latest"
