#!/usr/bin/env bash
# Watchdog protecting the other users of vent from the annotation jobs. Every 5 s it reads the free RAM (MemAvailable):
#   below WARN_GB  -> logs a warning with this user's processes' RSS and the GPU state;
#   below FLOOR_GB -> stops this user's annotation tmux sessions (SESSIONS) at once, so the kernel never has to kill
#                     anyone's processes; their work is saved and resumes when restarted.
# Heartbeat every 10 minutes. Run in its own tmux session:
#   tmux new -d -s guard "bash scripts/preprocess/annotations/vent/guard.sh >> ~/adele_tools/logs/guard.log 2>&1"
# Stop it: tmux kill-session -t guard
WARN_GB=${WARN_GB:-5}
FLOOR_GB=${FLOOR_GB:-3}
SESSIONS=${SESSIONS:-"prep smoke adele"}
warn_kb=$((WARN_GB * 1024 * 1024))
floor_kb=$((FLOOR_GB * 1024 * 1024))
n=0
echo "$(date '+%F %T') guard started: warn below ${WARN_GB} GB free, stop sessions [${SESSIONS}] below ${FLOOR_GB} GB"
while true; do
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  mine=$(ps -u "$USER" -o rss= | awk '{s += $1} END {print s + 0}')
  gpu=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  ts=$(date '+%F %T')
  if [ "$avail" -lt "$floor_kb" ]; then
    echo "$ts ALERT free RAM $((avail / 1024)) MB below ${FLOOR_GB} GB: stopping sessions (my RSS $((mine / 1024)) MB, gpu ${gpu} MiB)"
    for s in $SESSIONS; do
      tmux kill-session -t "$s" 2>/dev/null && echo "$ts STOPPED tmux session $s"
    done
  elif [ "$avail" -lt "$warn_kb" ]; then
    echo "$ts WARN free RAM $((avail / 1024)) MB, my RSS $((mine / 1024)) MB, gpu ${gpu} MiB"
  elif [ $((n % 120)) -eq 0 ]; then
    echo "$ts OK free RAM $((avail / 1024)) MB, my RSS $((mine / 1024)) MB, gpu ${gpu} MiB, sessions: $(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
  fi
  n=$((n + 1))
  sleep 5
done
