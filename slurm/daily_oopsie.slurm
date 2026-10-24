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

# The scron entry (daily.scron) fetches this repo's main into $SCRATCH/oopsie/oopsie-backend and
# runs this script from there.
BACKEND=${OOPSIE_BACKEND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck source=slurm/env.sh
source "$BACKEND/slurm/env.sh"

# Stage the docs site that receives stats.json. Without it, stats.json goes to the reports
# directory and is not published.
stage_docs() {
  if [[ -d "$DOCS/.git" ]]; then
    git -C "$DOCS" fetch -q --depth 1 origin "$DOCS_BRANCH" && git -C "$DOCS" reset -q --hard FETCH_HEAD
  else
    git clone -q --depth 1 --branch "$DOCS_BRANCH" "$DOCS_REMOTE" "$DOCS"
  fi
}
docs_rc=0
run_step stage_docs stage_docs || docs_rc=$?
if (( docs_rc == 0 )); then
  STATS_OUT=$DOCS/$STATS_REL
else
  STATS_OUT=$REPORTS_DIR/stats.json
fi

# Pin numpy before 2.4's x86-v2 baseline requirement. Need this for running on vm-small.
run_step setup_numpy bash -c "\"$UV\" add 'numpy<2.4' && \"$UV\" sync --reinstall-package numpy"

# Datasets that still fail after retries are skipped, and the sync exits non-zero.
# Keep going so stats cover everything that did sync; the job is marked failed at the end.
sync_rc=0
run_step sync_hf_org "$UV" run python -m oopsie_backend.ingest.sync \
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

# The full pipeline (configs/pipeline.yaml), recomputing only what changed. QA copies the
# episodes it checks to $EPISODE_STAGE first. Manual stages (imports, training) never run here;
# see pipeline.slurm.
pipeline_rc=0
run_step pipeline "$UV" run oopsie pipeline run all "${PIPELINE_SETTINGS[@]}" --explain \
  --set stats_docs_output="$STATS_OUT" || pipeline_rc=$?

# Publish the website stats: commit stats.json to oopsie-docs when it changed.
publish_stats() {
  git -C "$DOCS" add "$STATS_REL"
  if git -C "$DOCS" diff --cached --quiet; then
    echo "stats.json unchanged"
    return 0
  fi
  git "${GIT_AUTHOR[@]}" -C "$DOCS" commit -q -m "Update dashboard stats $(date -I)"
  git "${GIT_AUTHOR[@]}" -C "$DOCS" pull -q --rebase origin "$DOCS_BRANCH"
  git -C "$DOCS" push -q origin "HEAD:$DOCS_BRANCH"
}
publish_rc=0
if (( pipeline_rc == 0 && docs_rc == 0 )); then
  run_step publish_stats publish_stats || publish_rc=$?
fi

# The index, pipeline cache and reports are shared with the group, like the synced data.
for dir in "$INDEX_DIR" "$REPORTS_DIR"; do
  chgrp -R G-829069 "$dir"
  find "$dir" -type d -exec chmod 2770 {} +
  find "$dir" -type f -exec chmod 660 {} +
done

chmod g+s "$LOG_DIR"
chgrp G-829069 "$LOG_DIR"/*.log
chmod g+r "$LOG_DIR"/*.log

if (( sync_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but sync_hf_org exited ${sync_rc}; see failed datasets above (job ${SLURM_JOB_ID}) ==="
  exit "$sync_rc"
fi
if (( pipeline_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but the pipeline exited ${pipeline_rc} (job ${SLURM_JOB_ID}) ==="
  exit "$pipeline_rc"
fi
if (( docs_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but staging oopsie-docs exited ${docs_rc}; stats.json is in $REPORTS_DIR (job ${SLURM_JOB_ID}) ==="
  exit "$docs_rc"
fi
if (( publish_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but publishing stats.json exited ${publish_rc} (job ${SLURM_JOB_ID}) ==="
  exit "$publish_rc"
fi

echo "=== $(date -Is) SUCCESS: oopsie-daily completed in ${SECONDS}s (job ${SLURM_JOB_ID}) ==="
