# Chessez FastAPI Deploy Guide

## Overview

This project is deployed to a single Ubuntu EC2 instance with:

- `Nginx` handling public `80/443`
- `systemd` managing the FastAPI app
- `Redis` as a local runtime dependency
- `Stockfish` installed on the server

The deployment flow keeps the existing `./deploy.sh` one-command experience, but replaces `nohup + pkill` with a safer `systemd` restart.

## Target Layout

```text
/opt/chessez/
  releases/
    20260330-120000/
    20260331-101500/
  current -> /opt/chessez/releases/<latest>/
  shared/
    venv/
    logs/
```

## Runtime Files

- `config/systemd/chessez-fastapi.service`
- `config/systemd/chessez-fastapi.env.example`
- `config/nginx/chessez-fastapi.conf`
- `deploy.sh`

## First-Time Server Bootstrap

1. Install runtime packages on EC2.

```bash
sudo apt-get update -y
sudo apt-get install -y nginx redis-server stockfish python3 python3-venv python3-pip rsync
```

2. Run the first deploy with bootstrap enabled.

```bash
BOOTSTRAP=1 RELOAD_NGINX=1 ./deploy.sh
```

If `/etc/chessez/chessez-fastapi.env` does not exist yet, the script copies the example file there, updates `current` to the uploaded release, and exits so you can edit the env file safely before the service starts.

3. Edit the environment file and rerun the bootstrap deploy.

```bash
sudo nano /etc/chessez/chessez-fastapi.env
BOOTSTRAP=1 RELOAD_NGINX=1 ./deploy.sh
```

This bootstrap flow installs:

- `/etc/systemd/system/chessez-fastapi.service`
- `/etc/nginx/sites-available/chessez-fastapi`
- `/etc/nginx/sites-enabled/chessez-fastapi` symlink

4. If the server still has an old site file that points to Spring, disable or replace it after verifying the new config.

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Normal Redeploy

After bootstrap, redeploys are just:

```bash
./deploy.sh
```

The script will:

1. Create a new release directory
2. Upload the project
3. Reuse or update `/opt/chessez/shared/venv`
4. Move `current` to the new release
5. Restart `chessez-fastapi`
6. Run health checks against `127.0.0.1:8010`

## Environment Variables

The deploy script supports overrides:

```bash
KEY_PATH=~/.ssh/your-key.pem \
SERVER_IP=<SERVER_IP> \
SERVER_USER=ubuntu \
APP_ROOT=/opt/chessez \
SERVICE_NAME=chessez-fastapi \
APP_PORT=8010 \
./deploy.sh
```

## Health Checks

`deploy.sh` verifies:

- `GET /api/health`
- `GET /api/ai/status`
- `POST /api/ai/evaluate`

If any check fails, the deploy exits non-zero so the current release can be inspected before traffic changes.

## Nginx Routing

The current FastAPI app owns all public paths:

- `/`
- `/js`
- `/css`
- `/images`
- `/sounds`
- `/api/*`
- `/ws`

Important migration note:

- old config expected Spring on `8080`
- old WebSocket route was `/ws-chess`
- current app uses `/ws`

## systemd Commands

```bash
sudo systemctl daemon-reload
sudo systemctl enable chessez-fastapi
sudo systemctl restart chessez-fastapi
sudo systemctl status chessez-fastapi --no-pager
journalctl -u chessez-fastapi -n 100 --no-pager
```

## Rollback

1. List releases.

```bash
ls -1dt /opt/chessez/releases/*/
```

2. Repoint `current` to the previous release.

```bash
ln -sfn /opt/chessez/releases/<previous-release> /opt/chessez/current
sudo systemctl restart chessez-fastapi
```

3. If the issue is caused by the Nginx switch, restore the previous site config and reload Nginx.

```bash
sudo nginx -t
sudo systemctl reload nginx
```

During the first cutover, keep the old Spring/legacy AI process available until the new FastAPI app is verified externally.

## Post-Deploy Experiment Order

Once deployment is stable:

1. Run a small sanity check against `/api/ai/evaluate`
2. Re-run `scripts/run_searchtime_experiment_fastapi.py --sample 500`
3. Reconfirm `S/M/L`
4. Run the main MAE comparison
5. Run `scripts/locust_eval_load_fastapi.py`
