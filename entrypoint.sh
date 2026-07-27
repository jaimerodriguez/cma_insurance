#!/bin/sh
# Container entrypoint: seed the data directory if it is empty, then serve.
#
# Only does anything when CLAIMS_DATA_DIR points at a mounted volume. In the
# default no-volume deployment the image already ships a populated ./data and
# every copy below is skipped.
set -eu

SEED_DIR=/app/seed
DATA_DIR="${CLAIMS_DATA_DIR:-/app/data}"

mkdir -p "$DATA_DIR"

# Per-file, not a directory-level "is it empty" test: a volume holding a
# half-written world (say incidents.json but no adjusters.json, after an
# interrupted first boot) would pass the directory test and then serve an
# adjuster-less world. Existing files are never overwritten — this seeds, it
# does not reset.
for f in "$SEED_DIR"/*.json; do
    target="$DATA_DIR/$(basename "$f")"
    if [ ! -e "$target" ]; then
        echo "seeding $(basename "$f")"
        cp "$f" "$target"
    fi
done

exec python mcp_server.py "$@"
