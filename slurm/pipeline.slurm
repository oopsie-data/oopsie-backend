#!/bin/bash
#SBATCH -J oopsie-pipeline
#SBATCH -A IRI23005
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o /corral/utexas/IRI26024/logs/oopsie-pipeline-%j.log
#SBATCH -e /corral/utexas/IRI26024/logs/oopsie-pipeline-%j.log

# Run pipeline targets on a compute node, typically the manual ones the nightly job leaves out.
# Submit from the staged checkout; the arguments go to `oopsie pipeline run`:
#
#   cd $SCRATCH/oopsie/oopsie-backend
#   sbatch slurm/pipeline.slurm training --set training_recipe=configs/training/droid_joint_velocity.yaml
#   sbatch slurm/pipeline.slurm imports
#   sbatch slurm/pipeline.slurm sync --hf-token hf_... --set data_dir=/path/to/mirror
#
# --hf-token replaces HF_TOKEN and the token in tokens.yaml.

set -euo pipefail
umask 007
SECONDS=0
export HF_HUB_DISABLE_XET=1

args=()
while (( $# )); do
  case $1 in
    --hf-token) export HF_TOKEN=$2; shift 2 ;;
    --hf-token=*) export HF_TOKEN=${1#*=}; shift ;;
    *) args+=("$1"); shift ;;
  esac
done
set -- ${args[@]+"${args[@]}"}

BACKEND=${OOPSIE_BACKEND:-${SLURM_SUBMIT_DIR:-$PWD}}
# shellcheck source=slurm/env.sh
source "$BACKEND/slurm/env.sh"

if (( $# == 0 )); then
  echo "usage: sbatch slurm/pipeline.slurm TARGET [--hf-token TOKEN] [--set KEY=VALUE ...]" >&2
  exit 2
fi

rc=0
run_step "pipeline $*" "$UV" run oopsie pipeline run "${PIPELINE_SETTINGS[@]}" --explain "$@" || rc=$?

for dir in "$INDEX_DIR" "$ROOT/training" "$DATA_DIR/imports"; do
  if [[ -d "$dir" ]]; then
    chgrp -R G-829069 "$dir" 2>/dev/null || true
    find "$dir" -user "$USER" -type d -exec chmod 2770 {} +
    find "$dir" -user "$USER" -type f -exec chmod 660 {} +
  fi
done
exit "$rc"
