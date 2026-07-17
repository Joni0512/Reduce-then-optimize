#!/bin/bash
# 2026-07-16: Auto-resubmit watcher fuer die 50/10-Seeds. Laeuft dauerhaft
# (nicht als SLURM-Job, sondern als leichtgewichtige Schleife auf dem
# Login-Node - nur squeue/find/sleep, kein Rechenaufwand) und schiebt
# automatisch den naechsten noch nicht fertigen Seed nach, sobald ein
# Job-Slot frei wird (Account-Limit: max. 2 gleichzeitige Jobs). Nutzt
# dieselbe Resume-Logik wie die einzelnen Skripte selbst - kein Fortschritt
# geht bei einem Timeout verloren, es wird einfach automatisch weitergemacht.
# Stoppt von selbst, sobald alle 5 Seeds (42,1,2,3,4) 36/36 Ergebnisse haben.
set -u

SEEDS="42 1 2 3 4"
OUT_BASE="$HOME/Reduce-then-optimize/outputs/experiment_window_50_10_quick"
MAX_JOBS=2
POLL_INTERVAL=300  # 5 Minuten zwischen Checks
TARGET_PER_SEED=36  # 12 Instanzen x (Baseline + thr0.4 + thr0.5)

cd "$HOME/Reduce-then-optimize" || exit 1

seed_job_name() {
  echo "rp50s$1"
}

seed_complete_count() {
  local seed="$1"
  find "$OUT_BASE" -path "*_seed${seed}/*" -iname "results.json" 2>/dev/null | wc -l | tr -d ' '
}

seed_job_active() {
  local seed="$1"
  local jobname
  jobname=$(seed_job_name "$seed")
  squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$jobname"
}

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

log "=== auto-resubmit watcher gestartet (PID $$) ==="

while true; do
  all_done=true
  status_line=""
  for seed in $SEEDS; do
    n=$(seed_complete_count "$seed")
    status_line="${status_line}seed${seed}=${n}/${TARGET_PER_SEED} "
    if [ "$n" -lt "$TARGET_PER_SEED" ]; then
      all_done=false
    fi
  done
  log "Status: $status_line"

  if [ "$all_done" = true ]; then
    log "Alle 5 Seeds vollstaendig (36/36) - Watcher beendet sich."
    break
  fi

  current_jobs=$(squeue -u "$USER" -h -t pending,running 2>/dev/null | wc -l | tr -d ' ')
  if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
    submitted_this_round=false
    for seed in $SEEDS; do
      n=$(seed_complete_count "$seed")
      if [ "$n" -ge "$TARGET_PER_SEED" ]; then
        continue
      fi
      if seed_job_active "$seed"; then
        continue
      fi
      jobname=$(seed_job_name "$seed")
      log "Slot frei (aktuell $current_jobs/$MAX_JOBS) - submitte Seed $seed (Stand $n/$TARGET_PER_SEED), job-name=$jobname"
      if [ "$seed" = "42" ]; then
        sbatch --job-name="$jobname" submit_50_10_quick.sbatch
      else
        sbatch --job-name="$jobname" submit_50_10_seed.sbatch "$seed"
      fi
      submitted_this_round=true
      break
    done
    if [ "$submitted_this_round" = false ]; then
      log "Slot frei, aber kein Seed uebrig zum Submitten (alle aktiv oder fertig) - warte."
    fi
  fi

  sleep "$POLL_INTERVAL"
done
