#!/bin/sh
# Container entrypoint: prepare the data directory, then serve as `app`.
#
# The image starts as root and drops privileges here rather than declaring
# `USER app` in the Dockerfile. That is not a preference — a mounted volume
# arrives owned by root, so an image that had already dropped to `app` could not
# seed it and crash-looped on `cp: Permission denied`. Seeding needs root;
# serving does not.
set -eu

SEED_DIR=/app/seed
DATA_DIR="${CLAIMS_DATA_DIR:-/app/data}"
APP_UID=10001
APP_GID=10001

mkdir -p "$DATA_DIR"

if [ "$(id -u)" = "0" ]; then
    # chown is refused on some network filesystems — CIFS/SMB, which is exactly
    # what an Azure Files mount is. Not fatal: those mounts are normally already
    # writable via their file_mode/dir_mode options. Let the check below decide
    # instead of failing here on a mount that would have worked.
    chown -R "$APP_UID:$APP_GID" "$DATA_DIR" 2>/dev/null || true
fi

# Per file, not a directory-level "is it empty" test: a volume holding a
# half-written world (say incidents.json but no adjusters.json, after an
# interrupted first boot) would pass the directory test and then serve an
# adjuster-less world. Existing files are never overwritten — this seeds, it
# does not reset.
for f in "$SEED_DIR"/*.json; do
    target="$DATA_DIR/$(basename "$f")"
    if [ ! -e "$target" ]; then
        echo "seeding $(basename "$f")"
        cp "$f" "$target"
        chown "$APP_UID:$APP_GID" "$target" 2>/dev/null || true
    fi
done

if [ "$(id -u)" != "0" ]; then
    exec python mcp_server.py "$@"
fi

# Fail loudly and specifically here rather than letting every tool call return a
# permission error deep into a run, which reads as a bug in the tools.
if ! setpriv --reuid="$APP_UID" --regid="$APP_GID" --clear-groups \
        sh -c "touch '$DATA_DIR/.writetest' && rm -f '$DATA_DIR/.writetest'" 2>/dev/null; then
    echo "FATAL: $DATA_DIR is not writable by uid $APP_UID." >&2
    echo "  The claims tools write on almost every call. If this is a mounted" >&2
    echo "  volume, mount it writable by that uid — for Azure Files, set the" >&2
    echo "  share's file/dir mode to 0777 on the storage definition." >&2
    exit 1
fi

exec setpriv --reuid="$APP_UID" --regid="$APP_GID" --clear-groups \
    python mcp_server.py "$@"
