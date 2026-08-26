#!/usr/bin/env bash
# Start a local SearXNG instance for the agent's web lookup.
#
# SearXNG runs on YOUR machine. The agent only ever talks to localhost, so the
# loopback guard in ollama_client stays intact; SearXNG is what actually
# reaches out to search engines. No API key, no account, no vendor log.
#
# Requires Docker to be running.
#
#     ./scripts/setup_searxng.sh
#     ./scripts/setup_searxng.sh --stop
#     ./scripts/setup_searxng.sh --config-dir ~/.local/share/OffTheWire/searxng-config
#
# This is the POSIX counterpart of setup_searxng.ps1, and deliberately mirrors
# its behaviour decision for decision -- if one changes, change the other.

set -u

NAME="searxng-agent"
PORT=8080
CONFIG_DIR=""
STOP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --stop) STOP=1 ;;
        --port) PORT="$2"; shift ;;
        --config-dir) CONFIG_DIR="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ -z "$CONFIG_DIR" ]; then
    CONFIG_DIR="$(cd "$(dirname "$0")/.." && pwd)/searxng-config"
fi

if [ "$STOP" = 1 ]; then
    echo "Stopping $NAME..."
    docker rm -f "$NAME" >/dev/null 2>&1
    echo "Stopped."
    exit 0
fi

# --- preflight ---------------------------------------------------------------

# Checked before use: a missing docker binary otherwise surfaces as a shell
# "command not found" mid-script rather than a diagnosis.
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed on this machine." >&2
    echo >&2
    echo "Web lookup runs SearXNG in a local Docker container, so it needs" >&2
    echo "Docker: https://docs.docker.com/engine/install/" >&2
    echo >&2
    echo "Everything else works without it -- only --web is affected." >&2
    exit 1
fi

if ! docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
    echo "Docker is installed but not running (or this user lacks permission)." >&2
    echo "Start the Docker daemon, or add this user to the docker group:" >&2
    echo "  sudo usermod -aG docker \$USER   # then log out and back in" >&2
    exit 1
fi
echo "Docker engine $(docker info --format '{{.ServerVersion}}')"

if [ -n "$(docker ps -a --filter "name=$NAME" --format '{{.Names}}')" ]; then
    echo "Removing existing $NAME container..."
    docker rm -f "$NAME" >/dev/null
fi

# --- config ------------------------------------------------------------------
# SearXNG serves JSON only if 'json' is listed under search.formats. It is NOT
# enabled by default, and omitting it is the single most common reason the
# agent's search calls fail with a 403 or an HTML body.

mkdir -p "$CONFIG_DIR"
SECRET_KEY="$(head -c 32 /dev/urandom | od -A n -t x1 | tr -d ' \n')"

cat > "$CONFIG_DIR/settings.yml" <<EOF
use_default_settings: true

server:
  secret_key: "$SECRET_KEY"
  limiter: false
  image_proxy: false

search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json

engines:
  - name: google
    disabled: false
  - name: duckduckgo
    disabled: false
  - name: stackoverflow
    disabled: false
EOF
echo "Wrote $CONFIG_DIR/settings.yml (json format enabled)"

# --- run ---------------------------------------------------------------------

echo "Starting SearXNG on http://localhost:$PORT ..."

# Bind to loopback explicitly. A bare "-p 8080:8080" publishes on 0.0.0.0,
# which makes this an open search relay for anyone on the same network --
# their queries would leave from this machine's address.
if ! docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    -p "127.0.0.1:${PORT}:8080" \
    -v "${CONFIG_DIR}:/etc/searxng:rw" \
    -e "SEARXNG_BASE_URL=http://localhost:${PORT}/" \
    searxng/searxng:latest >/dev/null; then
    echo "docker run failed." >&2
    exit 1
fi

echo "Waiting for it to come up..."
ok=0
for _ in $(seq 1 30); do
    sleep 2
    if curl -fsS --max-time 5 "http://localhost:${PORT}/search?q=test&format=json" >/dev/null 2>&1; then
        ok=1
        break
    fi
done

if [ "$ok" = 1 ]; then
    echo
    echo "SearXNG is up and returning JSON."
    echo "Run the agent with web lookup:"
    # The invocation differs by where this copy of the script lives, and
    # printing the wrong one is worse than printing nothing.
    if [ -x "$(dirname "$0")/OffTheWire" ]; then
        echo "  OffTheWire --web"
    else
        echo "  .venv/bin/python src/agent.py <workspace> --web"
    fi
    echo
    echo "The container restarts with Docker, so this is a one-time setup."
else
    echo
    echo "Container started but the JSON endpoint did not respond in 60s." >&2
    echo "Check logs with:  docker logs $NAME" >&2
    exit 1
fi
