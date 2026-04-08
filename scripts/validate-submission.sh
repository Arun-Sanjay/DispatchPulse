#!/usr/bin/env bash
# DispatchPulse pre-submission validator.
#
# Runs the same three checks the hackathon's automated grader runs:
#   1. POST /reset to the live HF Space — must return HTTP 200
#   2. docker build of the repo (Dockerfile in root, then server/Dockerfile)
#   3. `openenv validate` against the repo
#
# Usage:
#   ./scripts/validate-submission.sh https://<your-space>.hf.space [path/to/repo]
#
# Exits non-zero on the first failure.
set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
RESET='\033[0m'

stop_at() {
    echo -e "${RED}validation FAILED at: $1${RESET}" >&2
    exit 1
}

ok() {
    echo -e "${GREEN}OK${RESET} $1"
}

warn() {
    echo -e "${YELLOW}WARN${RESET} $1"
}

if [[ -z "$PING_URL" ]]; then
    echo "Usage: $0 <space_url> [repo_dir]" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Check 1: HF Space deploys
# ---------------------------------------------------------------------------
echo
echo "[1/3] HF Space deploys — POST $PING_URL/reset"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" -d '{}' \
    "$PING_URL/reset" --max-time 30 || echo "000")

if [[ "$HTTP_CODE" != "200" ]]; then
    stop_at "HF Space /reset returned $HTTP_CODE (expected 200)"
fi
ok "HF Space /reset returned 200"

# ---------------------------------------------------------------------------
# Check 2: Docker build
# ---------------------------------------------------------------------------
echo
echo "[2/3] Docker build"
DOCKER_CONTEXT=""
if [[ -f "$REPO_DIR/Dockerfile" ]]; then
    DOCKER_CONTEXT="$REPO_DIR"
    echo "  using Dockerfile at: $REPO_DIR/Dockerfile"
elif [[ -f "$REPO_DIR/server/Dockerfile" ]]; then
    DOCKER_CONTEXT="$REPO_DIR/server"
    echo "  using Dockerfile at: $REPO_DIR/server/Dockerfile"
else
    stop_at "no Dockerfile found in $REPO_DIR or $REPO_DIR/server"
fi

if ! command -v docker >/dev/null 2>&1; then
    warn "docker CLI not found locally — skipping docker build (graders will run it)"
else
    if ! timeout 600 docker build "$DOCKER_CONTEXT" >/tmp/dispatchpulse-docker-build.log 2>&1; then
        echo "--- last 40 lines of docker build log ---" >&2
        tail -40 /tmp/dispatchpulse-docker-build.log >&2
        stop_at "docker build failed"
    fi
    ok "docker build succeeded (log: /tmp/dispatchpulse-docker-build.log)"
fi

# ---------------------------------------------------------------------------
# Check 3: openenv validate
# ---------------------------------------------------------------------------
echo
echo "[3/3] openenv validate"
if ! command -v openenv >/dev/null 2>&1; then
    warn "openenv CLI not found locally — skipping (graders will run it)"
else
    if ! (cd "$REPO_DIR" && openenv validate); then
        stop_at "openenv validate failed"
    fi
    ok "openenv validate passed"
fi

echo
echo -e "${GREEN}All checks passed.${RESET}"
