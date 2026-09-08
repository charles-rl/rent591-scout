#!/usr/bin/env bash
# No-cron host fallback: runs the incoming ingest on a fixed interval.
# Use when the host has neither systemd timers nor a cron daemon (containers).
#
#   nohup scripts/run_scheduler.sh >> data/logs/scheduler.log 2>&1 &
#
# RENT591_SCHED_INTERVAL (default 1800 = 30 min, matches the systemd timer).
set -u
cd "$(dirname "$0")/.."

INTERVAL="${RENT591_SCHED_INTERVAL:-1800}"
mkdir -p data/logs
echo "scheduler started (pid $$): ingest every ${INTERVAL}s"
while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- run ---"
  scripts/run_incoming.sh
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] run finished (rc=$?)"
  sleep "$INTERVAL"
done
