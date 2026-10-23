#!/bin/bash
#SBATCH -J oopsie-daily
#SBATCH -A IRI23005
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH -o /corral/utexas/IRI26024/logs/oopsie-daily-%j.log
#SBATCH -e /corral/utexas/IRI26024/logs/oopsie-daily-%j.log

set -euo pipefail
umask 007
SECONDS=0

export HF_HUB_DISABLE_XET=1

BACKEND=/work/09666/abahety/ls6/oopsie-backend
DATA_DIR=/corral/utexas/IRI26024/oopsiedata
STATS_OUT=/work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json
LOG_DIR=/corral/utexas/IRI26024/logs
UV=/home1/09666/abahety/.local/bin/uv

mkdir -p "$LOG_DIR" "$DATA_DIR" "$(dirname "$STATS_OUT")" "$BACKEND/reports"

if [[ -f "$HOME/.hf_token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: set HF_TOKEN or create \$HOME/.hf_token" >&2
  exit 1
fi

cd "$BACKEND"

run_step() {
  local name=$1
  shift
  local step_start=$SECONDS
  echo "=== $(date -Is) START: ${name} ==="
  local rc=0
  "$@" || rc=$?
  if (( rc != 0 )); then
    echo "=== $(date -Is) FAILED: ${name} exited ${rc} after $((SECONDS - step_start))s ==="
    return "$rc"
  fi
  echo "=== $(date -Is) SUCCESS: ${name} completed in $((SECONDS - step_start))s ==="
}

# Pin numpy before 2.4's x86-v2 baseline requirement. Need this for running on vm-small.
run_step setup_numpy bash -c "\"$UV\" add 'numpy<2.4' && \"$UV\" sync --reinstall-package numpy"

# Datasets that still fail after retries are skipped, and the sync exits non-zero.
# Keep going so stats cover everything that did sync; the job is marked failed at the end.
sync_rc=0
run_step sync_hf_org "$UV" run python -m oopsie_backend.scripts.sync_hf_org \
  --org OopsieData-Submissions \
  --local-dir "$DATA_DIR" \
  --group G-829069 \
  --verbose || sync_rc=$?

# Note: This step is kept as a safety net, but is not needed anymore.
# run_step fix_data_permissions bash -c "
#   find \"$DATA_DIR\" -user \"\$USER\" ! -group G-829069 -exec chgrp G-829069 {} +
#   find \"$DATA_DIR\" -user \"\$USER\" -type d -exec chmod 2770 {} +
#   find \"$DATA_DIR\" -user \"\$USER\" -type f -exec chmod 660 {} +
#   echo \"Set group G-829069 and permissions (dirs 770, files 660) under $DATA_DIR\"
# "

run_step compute_local_stats "$UV" run python -m oopsie_backend.scripts.compute_local_stats \
  --data-dir "$DATA_DIR" \
  --docs-output "$STATS_OUT"

chmod g+s /corral/utexas/IRI26024/logs
chgrp G-829069 /corral/utexas/IRI26024/logs/*.log
chmod g+r /corral/utexas/IRI26024/logs/*.log

if (( sync_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but sync_hf_org exited ${sync_rc}; see failed datasets above (job ${SLURM_JOB_ID}) ==="
  exit "$sync_rc"
fi

echo "=== $(date -Is) SUCCESS: oopsie-daily completed in ${SECONDS}s (job ${SLURM_JOB_ID}) ==="
