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
umask 002
SECONDS=0

BACKEND=/work/09666/abahety/ls6/oopsie-backend
DATA_DIR=/corral/utexas/IRI26024/oopsiedata
DOCS_STATS_OUT=/work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json
LOG_DIR=/corral/utexas/IRI26024/logs
UV=/home1/09666/abahety/.local/bin/uv

mkdir -p "$LOG_DIR" "$DATA_DIR" "$(dirname "$DOCS_STATS_OUT")" "$BACKEND/reports"
chgrp G-829069 "$DATA_DIR"
chmod g+rwxs "$DATA_DIR"

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
  "$@"
  echo "=== $(date -Is) SUCCESS: ${name} completed in $((SECONDS - step_start))s ==="
}

run_step sync_hf_org "$UV" run python -m oopsie_backend.scripts.sync_hf_org \
  --org OopsieData-Submissions \
  --local-dir "$DATA_DIR" \
  --group G-829069 \
  --verbose

run_step compute_local_stats "$UV" run python -m oopsie_backend.scripts.compute_local_stats \
  --data-dir "$DATA_DIR" \
  --docs-output "$DOCS_STATS_OUT"

echo "=== $(date -Is) SUCCESS: oopsie-daily completed in ${SECONDS}s (job ${SLURM_JOB_ID}) ==="
