#!/bin/bash
# Simple file-based lock for claude_agents_task_board.md
# Usage:
#   source taskboard_lock.sh
#   taskboard_lock    # acquire lock (waits up to 10s)
#   # ... edit the file ...
#   taskboard_unlock  # release lock

LOCKFILE="/data/repos/parameter-golf-fork/records/track_10min_16mb/2026-03-20_PiyushDattaSubmission/.taskboard.lock"

taskboard_lock() {
    local max_wait=10
    local waited=0
    while [ -f "$LOCKFILE" ]; do
        if [ $waited -ge $max_wait ]; then
            echo "WARN: Lock stale after ${max_wait}s, breaking it"
            rm -f "$LOCKFILE"
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "$$" > "$LOCKFILE"
}

taskboard_unlock() {
    rm -f "$LOCKFILE"
}
