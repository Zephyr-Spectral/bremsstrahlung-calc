#!/usr/bin/env bash
# Launch the bremsstrahlung-calc web app on the first free port
# starting from START_PORT. Opens the app in the default browser
# once uvicorn is listening.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Config ---------------------------------------------------------------
VENV="${VENV:-$HOME/science_venv}"
HOST="${HOST:-127.0.0.1}"
START_PORT="${START_PORT:-8001}"
END_PORT="${END_PORT:-8100}"
APP="${APP:-server.main:app}"
RELOAD="${RELOAD:-1}"   # set RELOAD=0 to disable --reload

# --- Activate venv --------------------------------------------------------
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: virtualenv not found at $VENV" >&2
    echo "Create it or set VENV=/path/to/venv before running." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# --- Find a free port -----------------------------------------------------
port_in_use() {
    # Returns 0 if something is listening on $1, else 1.
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

PORT=""
for (( p=START_PORT; p<=END_PORT; p++ )); do
    if ! port_in_use "$p"; then
        PORT=$p
        break
    fi
done

if [[ -z "$PORT" ]]; then
    echo "ERROR: no free port in range $START_PORT-$END_PORT" >&2
    echo "Currently listening ports in that range:" >&2
    lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null \
        | awk -v lo="$START_PORT" -v hi="$END_PORT" '
            NR>1 {
                n = split($9, a, ":"); port = a[n] + 0
                if (port >= lo && port <= hi) print "  " $1 " (pid " $2 ") -> " $9
            }' >&2
    exit 1
fi

URL="http://${HOST}:${PORT}"
echo "=========================================="
echo "  bremsstrahlung-calc"
echo "  listening on: $URL"
echo "=========================================="

# --- Open browser once the server is ready --------------------------------
(
    # Poll until the port is actually accepting connections, then open it.
    for _ in $(seq 1 50); do
        if port_in_use "$PORT"; then
            sleep 0.3
            if command -v open >/dev/null 2>&1; then
                open "$URL"
            elif command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$URL"
            fi
            exit 0
        fi
        sleep 0.2
    done
) &

# --- Run uvicorn (foreground) --------------------------------------------
RELOAD_FLAG=()
if [[ "$RELOAD" == "1" ]]; then
    RELOAD_FLAG=(--reload)
fi

exec uvicorn "$APP" --host "$HOST" --port "$PORT" "${RELOAD_FLAG[@]}"
