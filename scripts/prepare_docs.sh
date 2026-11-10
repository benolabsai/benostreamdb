#!/bin/sh
# Materialise the Sphinx source tree from docs/.
#
# Copies, not symlinks. Sphinx 9 resolves inline `.md` links against the source
# directory, and a symlinked source file resolves outside it, so the link loses
# its anchor and renders as an inert `<span class="xref myst">`. Copying makes
# each page a real document and the links resolve to generated `.html`.
#
# Both names are materialised: the lowercase docname the toctree uses, and the
# original mixed-case filename the sources link to.
#
# This script is the single implementation used by .github/workflows/docs.yml
# and by local builds.
set -eu

cd "$(dirname "$0")/.."

mkdir -p docs/source/guides docs/source/integrations docs/source/tutorials

# docs/*.md -> guides/<lowercase>.md and guides/<original>.md
for f in docs/*.md; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    name=$(echo "${base%.md}" | tr '[:upper:]' '[:lower:]')
    if [ "$name" = "index" ]; then
        continue
    fi
    rm -f "docs/source/guides/$name.md" "docs/source/guides/$base"
    cp -f "$f" "docs/source/guides/$name.md"
    cp -f "$f" "docs/source/guides/$base"
done

# docs/integrations/*.md -> integrations/<lowercase>.md and <original>.md
for f in docs/integrations/*.md; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    name=$(echo "${base%.md}" | tr '[:upper:]' '[:lower:]')
    if [ "$name" = "readme" ]; then
        continue
    fi
    rm -f "docs/source/integrations/$name.md" "docs/source/integrations/$base"
    cp -f "$f" "docs/source/integrations/$name.md"
    cp -f "$f" "docs/source/integrations/$base"
done

# docs/tutorials/*.md -> tutorials/<lowercase>.md and <original>.md
for f in docs/tutorials/*.md; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    name=$(echo "${base%.md}" | tr '[:upper:]' '[:lower:]')
    rm -f "docs/source/tutorials/$name.md" "docs/source/tutorials/$base"
    cp -f "$f" "docs/source/tutorials/$name.md"
    cp -f "$f" "docs/source/tutorials/$base"
done

# ADR_00N_*.md -> guides/adr_00N.md, the docname index.rst references.
for f in docs/architecture/ADR_*.md; do
    [ -f "$f" ] || continue
    num=$(basename "$f" | sed -n 's/^ADR_\([0-9][0-9]*\)_.*/\1/p')
    if [ -z "$num" ]; then
        continue
    fi
    rm -f "docs/source/guides/adr_$num.md"
    cp -f "$f" "docs/source/guides/adr_$num.md"
done

# docs/governance/*.md -> guides/<lowercase>.md and <original>.md. The toctree
# references guides/dependency_risk, so these are published even though they are
# governance documents rather than user guides.
for f in docs/governance/*.md; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    name=$(echo "${base%.md}" | tr '[:upper:]' '[:lower:]')
    rm -f "docs/source/guides/$name.md" "docs/source/guides/$base"
    cp -f "$f" "docs/source/guides/$name.md"
    cp -f "$f" "docs/source/guides/$base"
done

rm -f docs/source/guides/production_readiness.md
cp -f docs/PRODUCTION_READINESS_ASSESSMENT.md docs/source/guides/production_readiness.md
