#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

KEY_PATH="${KEY_PATH:-<PATH_TO_SSH_KEY>}"
SERVER_IP="${SERVER_IP:-<SERVER_IP>}"
SERVER_USER="${SERVER_USER:-ubuntu}"
APP_ROOT="${APP_ROOT:-/opt/chessez}"
SERVICE_NAME="${SERVICE_NAME:-chessez-fastapi}"
NGINX_SITE_NAME="${NGINX_SITE_NAME:-chessez-fastapi}"
APP_PORT="${APP_PORT:-8010}"
KEEP_RELEASES="${KEEP_RELEASES:-5}"
BOOTSTRAP="${BOOTSTRAP:-0}"
RELOAD_NGINX="${RELOAD_NGINX:-0}"
REMOTE_ENV_FILE="${REMOTE_ENV_FILE:-/etc/chessez/chessez-fastapi.env}"

RELEASE_ID="$(date +"%Y%m%d-%H%M%S")"
REMOTE_RELEASE_DIR="$APP_ROOT/releases/$RELEASE_ID"
REMOTE_VENV_DIR="$APP_ROOT/shared/venv"
REMOTE_LOG_DIR="$APP_ROOT/shared/logs"
REMOTE_LINK_PATH="$APP_ROOT/current"
REMOTE_HOST="$SERVER_USER@$SERVER_IP"

SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=accept-new)
RSYNC_EXCLUDES=(
  --exclude ".git/"
  --exclude ".cursor/"
  --exclude "venv/"
  --exclude ".venv/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude "*.pyo"
  --exclude "searchtime_raw_fastapi/"
  --exclude "searchtime_experiment_checkpoint_fastapi.json"
  --exclude "searchtime_mae_results_fastapi.json"
)

SAMPLE_FEN="8/8/pB1p4/P7/4kp2/2K1p3/3r4/4R3 w - - 6 57"

log() {
  printf '[deploy] %s\n' "$1"
}

fail() {
  printf '[deploy] ERROR: %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

run_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
}

require_command ssh
require_command rsync
require_command curl

[[ -f "$KEY_PATH" ]] || fail "SSH key not found at $KEY_PATH"
[[ -f "$PROJECT_ROOT/requirements.txt" ]] || fail "requirements.txt not found"
[[ -f "$PROJECT_ROOT/main.py" ]] || fail "main.py not found"

log "Preparing remote directories"
run_remote "
  set -euo pipefail
  sudo mkdir -p '$APP_ROOT/releases' '$REMOTE_LOG_DIR' '/etc/chessez'
  sudo chown -R '$SERVER_USER:$SERVER_USER' '$APP_ROOT'
  mkdir -p '$REMOTE_RELEASE_DIR'
"

log "Uploading release to $REMOTE_RELEASE_DIR"
rsync -az --delete "${RSYNC_EXCLUDES[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "$PROJECT_ROOT"/ "$REMOTE_HOST:$REMOTE_RELEASE_DIR/"

log "Installing runtime dependencies and Python packages"
run_remote "
  set -euo pipefail
  if ! command -v python3 >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip
  fi
  if ! command -v stockfish >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y stockfish
  fi
  if [ ! -d '$REMOTE_VENV_DIR' ]; then
    python3 -m venv '$REMOTE_VENV_DIR'
  fi
  '$REMOTE_VENV_DIR/bin/pip' install --upgrade pip
  '$REMOTE_VENV_DIR/bin/pip' install -r '$REMOTE_RELEASE_DIR/requirements.txt'
"

log "Updating current symlink"
run_remote "
  set -euo pipefail
  ln -sfn '$REMOTE_RELEASE_DIR' '$REMOTE_LINK_PATH'
"

if [[ "$BOOTSTRAP" == "1" ]]; then
  log "Installing systemd unit and nginx template"
  run_remote "
    set -euo pipefail
    sudo cp '$REMOTE_RELEASE_DIR/config/systemd/chessez-fastapi.service' '/etc/systemd/system/$SERVICE_NAME.service'
    sudo cp '$REMOTE_RELEASE_DIR/config/nginx/chessez-fastapi.conf' '/etc/nginx/sites-available/$NGINX_SITE_NAME'
    if [ ! -L '/etc/nginx/sites-enabled/$NGINX_SITE_NAME' ]; then
      sudo ln -s '/etc/nginx/sites-available/$NGINX_SITE_NAME' '/etc/nginx/sites-enabled/$NGINX_SITE_NAME'
    fi
    if [ ! -f '$REMOTE_ENV_FILE' ]; then
      sudo cp '$REMOTE_RELEASE_DIR/config/systemd/chessez-fastapi.env.example' '$REMOTE_ENV_FILE'
      sudo chmod 640 '$REMOTE_ENV_FILE'
      echo 'Created $REMOTE_ENV_FILE from example and updated current release link. Update it and rerun deploy.'
      exit 20
    fi
    sudo systemctl daemon-reload
    sudo systemctl enable '$SERVICE_NAME'
    sudo nginx -t
  "
fi

log "Restarting FastAPI service"
run_remote "
  set -euo pipefail
  sudo systemctl daemon-reload
  sudo systemctl restart '$SERVICE_NAME'
  sudo systemctl --no-pager --full status '$SERVICE_NAME' | sed -n '1,15p'
"

if [[ "$RELOAD_NGINX" == "1" || "$BOOTSTRAP" == "1" ]]; then
  log "Reloading nginx"
  run_remote "
    set -euo pipefail
    sudo nginx -t
    sudo systemctl reload nginx
  "
fi

log "Running health checks"
run_remote "
  set -euo pipefail
  curl --fail --silent 'http://127.0.0.1:$APP_PORT/api/health' >/dev/null
  curl --fail --silent 'http://127.0.0.1:$APP_PORT/api/ai/status' >/dev/null
  curl --fail --silent -X POST 'http://127.0.0.1:$APP_PORT/api/ai/evaluate' \
    -H 'Content-Type: application/json' \
    -d '{\"fen\":\"$SAMPLE_FEN\",\"searchTimeMs\":50}' >/dev/null
"

log "Pruning old releases (keeping latest $KEEP_RELEASES)"
run_remote "
  set -euo pipefail
  cd '$APP_ROOT/releases'
  ls -1dt */ 2>/dev/null | tail -n +$((KEEP_RELEASES + 1)) | xargs -r rm -rf
"

log "Deployment completed successfully"
