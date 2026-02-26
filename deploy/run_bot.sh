#!/bin/bash
# run_bot.sh — Auto-restart wrapper for live_bot.py
# Usage: nohup ./deploy/run_bot.sh &
# Or use with systemd (see deploy/polymarket-bot.service)

# Always run from project root
cd "$(dirname "$0")/.."
LOG="logs/bot_supervisor.log"
mkdir -p logs

while true; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] Starting live_bot.py..." >> "$LOG"
    python bot/live_bot.py 2>&1 | tee -a "logs/bot_stdout_$(date -u '+%Y%m%d').log"
    EXIT_CODE=$?
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] Bot exited with code $EXIT_CODE, restarting in 5s..." >> "$LOG"
    sleep 5
done
